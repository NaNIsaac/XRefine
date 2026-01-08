"""
Refinement Pipeline
===================
Multi-stage Refinement Workflow

Usage:
    from xrefine.refine import run_real_inference
    
    result = run_real_inference(
        checkpoint_path="path/to/model.pt",
        data_path="path/to/data.dat",
        config=RefinementConfig()  # Optional custom config
    )
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from scipy.signal import find_peaks

from xrefine.config import XRefineConfig, CONFIG
from xrefine.models import XRR_Fourier_PINN
from xrefine.physics_utils import tth2q
from xrefine.core.structures import ParamSet
from xrefine.logger import logger

from xrefine.refine.config import RefinementConfig, StageConfig, DEFAULT_CONFIG
from xrefine.refine.optimizer import optimize_step
from xrefine.refine.losses import CORRELATION


def load_real_data(file_path: str, config: XRefineConfig) -> torch.Tensor:
    """실제 데이터를 로드하고 유효 q 영역을 추출합니다."""
    df = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python', dtype=str)
    for col in df.columns:
        df[col] = df[col].str.replace('(', '', regex=False).str.replace(')', '', regex=False).astype(float)
    
    x_raw, y_raw = df.values[:, 0], df.values[:, 1]
    x_raw = tth2q(x_raw, config.instrument.wavelength)
    
    norm_factor = y_raw.max()
    y_norm = y_raw / (norm_factor + 1e-12)
    log_y_raw = np.log10(np.clip(y_norm, 1e-11, 1.0))
    
    q_target = config.q_grid.numpy()
    log_y_interp = np.interp(q_target, x_raw, log_y_raw)
    log_y_interp[(q_target < x_raw.min()) | (q_target > x_raw.max())] = -15.0
    
    return torch.from_numpy(log_y_interp).float().view(1, -1)


def grid_search_thickness(
    model, 
    p_init: ParamSet, 
    log_r_obs: torch.Tensor, 
    xrr_config: XRefineConfig,
    refine_config: RefinementConfig,
    stage_config: StageConfig
) -> float:
    """Grid Search를 사용한 두께 탐색"""
    device = xrr_config.device
    q_grid = xrr_config.q_grid.to(device)
    log_r_obs_gpu = log_r_obs.to(device)
    
    # Q-range mask
    q_max = stage_config.q_max or refine_config.q_cutoff
    mask = (log_r_obs_gpu > -14.0) & (q_grid > 0.05) & (q_grid < q_max)
    target_pattern = log_r_obs_gpu[mask]
    target_mean = target_pattern.mean()
    
    # Grid 설정
    d_min, d_max = stage_config.grid_range
    d_grid = np.linspace(d_min, d_max, stage_config.grid_points)
    
    # 임시 파라미터 설정 (Oxide/Roughness 제거)
    p_scan = deepcopy(p_init)
    p_scan.d_ox.fill_(0.0)
    p_scan.sig_ox.fill_(0.0)
    p_scan.rho_ox.fill_(0.0)
    p_scan.sig_f.fill_(2.0)
    
    best_corr_loss = 1e9
    best_d = p_init.d_f.item()
    scores = []
    
    for d_val in d_grid:
        p_scan.d_f.fill_(d_val)
        
        with torch.no_grad():
            _, r_sim, _ = model.forward_with_params(p_scan)
            r_sim_norm = r_sim / (r_sim.max() + 1e-12)
            log_r_sim = torch.log10(torch.clamp(r_sim_norm, min=1e-12))
            
            sim_pattern = log_r_sim[mask]
            vx = sim_pattern - sim_pattern.mean()
            vy = target_pattern - target_mean
            corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-9)
            
            c_loss = 1.0 - corr.item()
            scores.append(c_loss)
            
            if c_loss < best_corr_loss:
                best_corr_loss = c_loss
                best_d = d_val
    
    # Valley Detection
    peaks, _ = find_peaks(-np.array(scores), prominence=0.005)
    
    # Critical Angle 기반 필터링
    r_obs_linear = 10 ** log_r_obs.squeeze().cpu().numpy()
    peak_idx = np.argmax(r_obs_linear)
    q_critical = xrr_config.q_grid.numpy()[peak_idx]
    min_d_threshold = refine_config.min_thickness_ratio / (q_critical + 1e-9)
    
    logger.info(f"🛡️ Dynamic Thickness Filter: d < {min_d_threshold:.1f}A 제외 (Qc={q_critical:.4f})")
    
    candidates = []
    for idx in peaks:
        d_cand = d_grid[idx]
        loss_cand = scores[idx]
        if d_cand < min_d_threshold:
            logger.info(f"⚠️ 후보 제외 (Too Thin): {d_cand:.1f}A (Loss={loss_cand:.4f})")
            continue
        candidates.append((d_cand, loss_cand))
    
    if not candidates:
        logger.warning("🚫 적절한 Local Minima 없음. Global Min 사용.")
        return best_d
    
    candidates.sort(key=lambda x: x[1])
    best_d = candidates[0][0]
    logger.info(f"✅ 후보군 Top 3: {[f'{d:.1f}A({l:.4f})' for d, l in candidates[:3]]}")
    logger.info(f"🎯 최종 선택: d_f = {best_d:.1f} A")
    
    return best_d


def run_refinement(
    model,
    p_init: ParamSet,
    log_r_obs: torch.Tensor,
    xrr_config: XRefineConfig,
    refine_config: RefinementConfig = None
) -> ParamSet:
    """
    Multi-stage Refinement 실행
    
    Args:
        model: XRR_Fourier_PINN 모델
        p_init: AI 초기 추정값
        log_r_obs: 관측된 반사율 (log10)
        xrr_config: XRR 설정
        refine_config: Refinement 설정 (None이면 기본값 사용)
    
    Returns:
        최적화된 ParamSet
    """
    if refine_config is None:
        refine_config = DEFAULT_CONFIG
    
    p_current = p_init
    fixed_thickness = None  # Grid Search에서 찾은 두께
    
    for i, stage in enumerate(refine_config.stages):
        logger.info(f"{'─'*20}")
        logger.info(f"📌 [{i}단계] {stage.name}")
        
        # Grid Search 스테이지
        if stage.use_grid_search:
            fixed_thickness = grid_search_thickness(
                model, p_current, log_r_obs, xrr_config, refine_config, stage
            )
            # 두께 설정
            p_current = deepcopy(p_current)
            p_current.d_f.fill_(fixed_thickness)
            p_current.d_ox.fill_(0.0)
            p_current.sig_ox.fill_(0.0)
            p_current.rho_ox.fill_(0.0)
            p_current.sig_f.fill_(3.0)
            continue
        
        # 일반 최적화 스테이지
        stage_with_fixed = deepcopy(stage)
        
        # 두께 고정 적용 (아직 fine-tuning 스테이지가 아닌 경우)
        if fixed_thickness is not None and 'd_f' not in stage.optimize_keys:
            stage_with_fixed.fixed_params['d_f'] = fixed_thickness
        
        # Fine-tuning 스테이지인 경우 두께 범위 제한
        if fixed_thickness is not None and 'd_f' in stage.optimize_keys:
            original_d_range = xrr_config.ranges.d_f
            xrr_config.ranges.d_f = (
                max(10.0, fixed_thickness - refine_config.thickness_fine_range),
                fixed_thickness + refine_config.thickness_fine_range
            )
            logger.info(f"🎯 두께 미세 조정: d_f ≈ {fixed_thickness:.1f} ± {refine_config.thickness_fine_range}A")
        
        # Q max 기본값 설정
        if stage_with_fixed.q_max is None:
            stage_with_fixed.q_max = refine_config.q_cutoff
        
        # 최적화 실행
        p_current = optimize_step(
            model, p_current, log_r_obs, xrr_config, stage_with_fixed,
            verbose=refine_config.verbose
        )
        
        # 범위 복원
        if fixed_thickness is not None and 'd_f' in stage.optimize_keys:
            xrr_config.ranges.d_f = original_d_range
    
    return p_current


@torch.no_grad()
def run_real_inference(
    checkpoint_path: str, 
    data_path: str, 
    output_dir: str = "./results",
    refine_config: RefinementConfig = None
) -> ParamSet:
    """
    실제 데이터 피팅 실행
    
    Args:
        checkpoint_path: 모델 체크포인트 경로
        data_path: 데이터 파일 경로
        output_dir: 결과 저장 디렉토리
        refine_config: Refinement 설정 (None이면 기본값 사용)
    
    Returns:
        최적화된 ParamSet
    """
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except:
        pathlib.PosixPath = temp
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    finally:
        pathlib.PosixPath = temp
    
    xrr_config = XRefineConfig(**ckpt['config'])
    xrr_config.ranges = CONFIG.ranges  # 최신 Range 적용
    logger.info(f"🔄 Range Config Overridden: sig_f max = {xrr_config.ranges.sig_f[1]}")
    
    model = XRR_Fourier_PINN(config=xrr_config).to(xrr_config.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    data_name = Path(data_path).stem
    input_tensor = load_real_data(data_path, xrr_config)
    
    # AI 초기 추정
    p_ai, r_ai, _ = model(input_tensor.to(xrr_config.device))
    
    # Refinement 실행
    p_refined = run_refinement(model, p_ai, input_tensor, xrr_config, refine_config)
    
    # 최종 결과
    _, r_final, _ = model.forward_with_params(p_refined.to(xrr_config.device))
    
    # 시각화 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / f"{data_name}_fit_{datetime.now().strftime('%H%M%S')}.png"
    
    r_final_norm = r_final / (r_final.max() + 1e-12)
    r_ai_norm = r_ai / (r_ai.max() + 1e-12)
    
    _save_plot(xrr_config, input_tensor[0], r_ai_norm[0], r_final_norm[0], data_name, fig_path, p_refined)
    
    logger.info(f"✨ 완료. 결과 저장됨: {fig_path}")
    print("Refined Params:", p_refined)
    
    return p_refined


def _save_plot(config, y_meas_log, r_ai, r_ref, title, save_path, p_refined=None):
    q = config.q_grid.numpy()
    y_meas = y_meas_log.cpu().numpy()
    
    valid_mask = y_meas > -14.0
    q_valid = q[valid_mask]
    y_valid = 10 ** y_meas[valid_mask]
    r_ai_valid = r_ai.cpu().numpy()[valid_mask]
    r_ref_valid = r_ref.cpu().numpy()[valid_mask]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(q_valid, y_valid, 'ko', ms=2, alpha=0.3, label='Measured')
    ax.semilogy(q_valid, r_ai_valid, 'b--', alpha=0.6, label='AI Init')
    ax.semilogy(q_valid, r_ref_valid, 'r-', lw=1.5, label='Refined')
    ax.set_xlabel("Q (Å⁻¹)")
    ax.set_ylabel("Reflectivity")
    ax.set_title(f"XRefine: {title}")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    
    # 주요 파라미터 표시
    if p_refined is not None:
        param_text = (
            f"Film:  d={p_refined.d_f.item():.1f}Å  σ={p_refined.sig_f.item():.1f}Å  ρ={p_refined.rho_f.item():.1f}\n"
            f"Oxide: d={p_refined.d_ox.item():.1f}Å  σ={p_refined.sig_ox.item():.1f}Å  ρ={p_refined.rho_ox.item():.1f}"
        )
        ax.text(0.02, 0.02, param_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="XRefine: XRR Data Refinement Tool")
    parser.add_argument("data", help="Path to XRR data file (.dat)")
    parser.add_argument("-c", "--checkpoint", 
                        default=r"D:\03_Resources\Data\XRR_AI\xrefine\v1\best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("-o", "--output", default="./results",
                        help="Output directory for results")
    parser.add_argument("--preset", choices=["default", "quick", "high_quality"],
                        default="default", help="Refinement preset")
    
    args = parser.parse_args()
    
    # Preset 선택
    if args.preset == "quick":
        config = RefinementConfig.quick_thickness()
    elif args.preset == "high_quality":
        config = RefinementConfig.high_quality()
    else:
        config = None  # Default
    
    run_real_inference(args.checkpoint, args.data, args.output, config)
