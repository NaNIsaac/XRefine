"""
Refinement Pipeline
===================
Multi-stage Refinement Workflow with Smart Cutoff

Usage:
    from xrefine.refine import run_real_inference

    # 1. 자동 감지 (권장)
    run_real_inference(ckpt, data, q_cutoff=None)

    # 2. 수동 지정
    run_real_inference(ckpt, data, q_cutoff=0.45)
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


def auto_detect_cutoff(q: np.ndarray, log_y: np.ndarray, window: int = 15, slope_threshold: float = -1.5) -> float:
    """
    [Smart Cutoff] 유효 데이터(> -14.0) 내에서만 신호 끝자락을 탐색합니다.
    """
    # 1. [안전장치] 이미 -15.0으로 마스킹된 데이터(측정 범위 밖)는 탐색에서 제외
    valid_indices = log_y > -14.0
    
    # 유효 데이터가 너무 적으면 그냥 Max 반환
    if valid_indices.sum() < window * 2:
        return q.max()
        
    # 유효한 구간만 추출
    q_valid = q[valid_indices]
    y_valid = log_y[valid_indices]
    
    max_q = q_valid.max()
    
    # 2. 유효 데이터 끝에서부터(High Q) -> 앞쪽으로(Low Q) 스캔
    for i in range(len(q_valid) - window, 0, -2):
        q_seg = q_valid[i : i+window]
        y_seg = y_valid[i : i+window]
        
        # 선형 회귀로 기울기 계산
        slope, _ = np.polyfit(q_seg, y_seg, 1)
        
        # 기울기가 임계값보다 가파르면(음수) 신호 영역 진입으로 판단
        if slope < slope_threshold:
            # 발견된 지점 (여기가 신호의 끝)
            found_q = q_valid[i + window]
            return min(found_q + 0.02, max_q) # 0.02 여유 마진

    return max_q


def load_real_data(file_path: str, config: XRefineConfig) -> torch.Tensor:
    """
    순수 데이터 로드 함수 (전처리 및 마스킹 없음)
    """
    df = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python', dtype=str)
    for col in df.columns:
        df[col] = df[col].str.replace('(', '', regex=False).str.replace(')', '', regex=False).astype(float)

    x_raw, y_raw = df.values[:, 0], df.values[:, 1]
    x_raw = tth2q(x_raw, config.instrument.wavelength)

    # 정규화 (Peak=1.0)
    norm_factor = y_raw.max()
    y_norm = y_raw / (norm_factor + 1e-12)
    log_y_raw = np.log10(np.clip(y_norm, 1e-11, 1.0))

    # Grid Interpolation
    q_target = config.q_grid.numpy()
    log_y_interp = np.interp(q_target, x_raw, log_y_raw)
    
    # 측정 범위 밖 기본 마스킹
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

    # 임계각 감지
    r_obs_linear = 10 ** log_r_obs_gpu.squeeze()
    peak_idx = int(torch.argmax(r_obs_linear).item())
    q_critical = q_grid[peak_idx].item()

    # 탐색 범위: 임계각 직후 ~ Config에 설정된 Cutoff까지
    q_start = q_critical + 0.01
    q_end = refine_config.q_cutoff  # [중요] 설정된 Cutoff 활용

    # 요동 감지 (Oscillation Detection)
    log_r_np = log_r_obs_gpu.squeeze().cpu().numpy()
    grad = np.abs(np.gradient(log_r_np))
    window = 20
    grad_smooth = np.convolve(grad, np.ones(window)/window, mode='same')
    threshold = np.mean(grad_smooth) * 1.5
    high_oscillation = grad_smooth > threshold
    high_osc_mask = torch.from_numpy(high_oscillation).to(device)

    # 최종 마스크
    mask = (log_r_obs_gpu > -14.0) & (q_grid > q_start) & (q_grid < q_end) & high_osc_mask

    if mask.sum() < 30:
        logger.warning(f"Grid Search 유효 포인트 부족. 기본 범위({q_start:.2f}~{q_end:.2f}) 사용")
        mask = (log_r_obs_gpu > -14.0) & (q_grid > q_start) & (q_grid < q_end)

    logger.info(f"Grid Search Range: {q_start:.3f} ~ {q_end:.3f} A^-1 ({mask.sum().item()} pts)")

    target_pattern = log_r_obs_gpu[mask]
    target_mean = target_pattern.mean()

    # Grid 설정
    d_min, d_max = stage_config.grid_range
    d_grid = np.linspace(d_min, d_max, stage_config.grid_points)

    p_scan = deepcopy(p_init)
    p_scan.d_ox.fill_(0.0)
    p_scan.sig_ox.fill_(0.0)
    p_scan.rho_ox.fill_(0.0)
    p_scan.sig_f.fill_(2.0)

    best_corr_loss = 1e9
    best_d = p_init.d_f.item()
    c_scores = []s
    m_scores = []

    for d_val in d_grid:
        p_scan.d_f.fill_(d_val)
        with torch.no_grad():
            _, r_sim, _ = model.forward_with_params(p_scan)
            r_sim_norm = r_sim / (r_sim.max() + 1e-12)
            log_r_sim = torch.log10(torch.clamp(r_sim_norm, min=1e-12))

            sim_pattern = log_r_sim[mask]

            # Correlation
            vx = sim_pattern - sim_pattern.mean()
            vy = target_pattern - target_mean
            corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-9)
            c_loss = 1.0 - corr.item()
            c_scores.append(c_loss)

            # MSE
            mse = torch.mean((sim_pattern - target_pattern)**2).item()
            m_scores.append(mse)

            if c_loss < best_corr_loss:
                best_corr_loss = c_loss
                best_d = d_val

    # Peak Finding & Candidate Selection
    peaks_c, _ = find_peaks(-np.array(c_scores), prominence=0.01)
    min_d_threshold = refine_config.min_thickness_ratio / (q_critical + 1e-9)

    def get_candidates(peak_indices, score_list):
        cands = []
        for idx in peak_indices:
            if d_grid[idx] >= min_d_threshold:
                cands.append((d_grid[idx], score_list[idx]))
        
        # Fallback to global search if no peaks
        if not cands:
            arr = np.array(score_list)
            indices = np.argsort(arr)
            for idx in indices:
                if d_grid[idx] >= min_d_threshold:
                    cands.append((d_grid[idx], score_list[idx]))
                    if len(cands) >= 3: break
        
        cands.sort(key=lambda x: x[1])
        return cands

    candidates = get_candidates(peaks_c, c_scores)
    if not candidates:
        peaks_m, _ = find_peaks(-np.array(m_scores), prominence=0.01)
        candidates = get_candidates(peaks_m, m_scores)

    if not candidates:
        return best_d

    # Two-Pass Check for Top 3
    top_cands = candidates[:3]
    if len(top_cands) > 1:
        logger.info(f"Two-Pass 검증: {[f'{d:.1f}A' for d,_ in top_cands]}")
        final_losses = []
        for d_cand, _ in top_cands:
            p_test = deepcopy(p_init)
            p_test.d_f.fill_(d_cand)
            p_test.sig_f.fill_(3.0)
            p_test.d_ox.fill_(0.0)

            # Quick Fit
            qs = StageConfig(name="Quick", optimize_keys={'sig_f','rho_f'}, fixed_params={'d_f':d_cand}, loss_fn=stage_config.loss_fn, maxiter=30, popsize=15)
            p_opt = optimize_step(model, p_test, log_r_obs, xrr_config, qs, verbose=False)
            
            with torch.no_grad():
                _, r_s, _ = model.forward_with_params(p_opt)
                r_s_n = r_s / (r_s.max() + 1e-12)
                l_s = torch.log10(torch.clamp(r_s_n, min=1e-12))[mask]
                vx = l_s - l_s.mean()
                vy = target_pattern - target_mean
                cr = torch.sum(vx*vy) / (torch.sqrt(torch.sum(vx**2))*torch.sqrt(torch.sum(vy**2)) + 1e-9)
                final_losses.append((d_cand, 1.0 - cr.item()))
        
        final_losses.sort(key=lambda x: x[1])
        best_d = final_losses[0][0]
    else:
        best_d = top_cands[0][0]

    logger.info(f"🎯 Grid Search Selected: d_f = {best_d:.1f} A")
    return best_d


def run_refinement(
    model,
    p_init: ParamSet,
    log_r_obs: torch.Tensor,
    xrr_config: XRefineConfig,
    refine_config: RefinementConfig = None
) -> ParamSet:
    """Multi-stage Refinement 실행"""
    if refine_config is None:
        refine_config = DEFAULT_CONFIG

    p_current = p_init
    fixed_thickness = None

    logger.info(f"\n{'Stage':<20} | {'d_f':>7} {'σ_f':>6} {'ρ_f':>5} | {'d_ox':>5} {'σ_ox':>5} {'ρ_ox':>5}")
    logger.info("-" * 70)

    def log_params(stage_name: str, p: ParamSet):
        logger.info(
            f"{stage_name:<20} | "
            f"{p.d_f.item():>7.1f} {p.sig_f.item():>6.1f} {p.rho_f.item():>5.1f} | "
            f"{p.d_ox.item():>5.1f} {p.sig_ox.item():>5.1f} {p.rho_ox.item():>5.1f}"
        )

    log_params("AI Init", p_init)

    for i, stage in enumerate(refine_config.stages):
        # 1. Grid Search
        if stage.use_grid_search:
            fixed_thickness = grid_search_thickness(
                model, p_current, log_r_obs, xrr_config, refine_config, stage
            )
            p_current = deepcopy(p_current)
            p_current.d_f.fill_(fixed_thickness)
            p_current.d_ox.fill_(0.0)
            p_current.sig_ox.fill_(0.0)
            p_current.rho_ox.fill_(0.0)
            p_current.sig_f.fill_(3.0)
            log_params(f"[{i}] {stage.name[:15]}", p_current)
            continue

        # 2. Regular Optimization
        stage_with_fixed = deepcopy(stage)

        # 두께 고정 논리
        if fixed_thickness is not None and 'd_f' not in stage.optimize_keys:
            stage_with_fixed.fixed_params['d_f'] = fixed_thickness

        if fixed_thickness is not None and 'd_f' in stage.optimize_keys:
            original_d_range = xrr_config.ranges.d_f
            xrr_config.ranges.d_f = (
                max(10.0, fixed_thickness - refine_config.thickness_fine_range),
                fixed_thickness + refine_config.thickness_fine_range
            )

        # [중요] Config의 Cutoff를 모든 스테이지에 전파
        if stage_with_fixed.q_max is None:
            stage_with_fixed.q_max = refine_config.q_cutoff

        p_current = optimize_step(
            model, p_current, log_r_obs, xrr_config, stage_with_fixed, verbose=False
        )

        log_params(f"[{i}] {stage.name[:15]}", p_current)

        # 범위 복구
        if fixed_thickness is not None and 'd_f' in stage.optimize_keys:
            xrr_config.ranges.d_f = original_d_range

    logger.info("-" * 70)
    return p_current


@torch.no_grad()
def run_real_inference(
    checkpoint_path: str, 
    data_path: str, 
    output_dir: str = "./results",
    refine_config: RefinementConfig = None,
    q_cutoff: float = None  # None이면 자동 감지
) -> ParamSet:
    """실제 데이터 추론 및 피팅 (Smart Cutoff 포함)"""
    # 1. 모델 로드
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
    xrr_config.ranges = CONFIG.ranges
    
    model = XRR_Fourier_PINN(config=xrr_config).to(xrr_config.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    if refine_config is None:
        refine_config = DEFAULT_CONFIG

    # 2. 데이터 로드 (순수 데이터)
    input_tensor = load_real_data(data_path, xrr_config)
    q_grid_np = xrr_config.q_grid.numpy()
    
    # 3. [핵심] 피팅 시작 전 Cutoff 감지 및 데이터 마스킹
    #    AI는 전체 데이터를 보고 힌트를 얻지만, Optimizer는 노이즈를 보지 말아야 함.
    
    # (A) Cutoff 결정
    if q_cutoff is None:
        # 자동 감지
        log_y_np = input_tensor.squeeze().cpu().numpy()

        detected_cutoff = auto_detect_cutoff(q_grid_np, log_y_np)
        actual_cutoff = detected_cutoff
        logger.info(f"🤖 Auto-Cutoff Detected: Q > {actual_cutoff:.3f} A^-1 (Noise Floor Removed)")
    else:
        # 수동 지정
        actual_cutoff = q_cutoff
        logger.info(f"✂️ Manual Cutoff Applied: Q > {actual_cutoff:.3f} A^-1")

    # (B) Config에 반영 (모든 스테이지가 이 값을 참조)
    refine_config.q_cutoff = actual_cutoff

    # (C) 텐서 마스킹 (물리적으로 값 제거 -> -15.0)
    #     이렇게 하면 Optimizer나 Loss Function이 이 영역을 무시함
    mask_indices = q_grid_np > actual_cutoff
    masked_input = input_tensor.clone()
    masked_input[0, mask_indices] = -15.0

    # 4. AI 추론
    p_ai, r_ai, _ = model(masked_input.to(xrr_config.device))

    # 5. Refinement 실행 (마스킹된 텐서 전달)
    p_refined = run_refinement(model, p_ai, masked_input, xrr_config, refine_config)

    # 6. 최종 결과 저장
    _, r_final, _ = model.forward_with_params(p_refined.to(xrr_config.device))
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_name = Path(data_path).stem
    fig_path = output_path / f"{data_name}_fit_{datetime.now().strftime('%H%M%S')}.png"

    r_final_norm = r_final / (r_final.max() + 1e-12)
    r_ai_norm = r_ai / (r_ai.max() + 1e-12)

    # 시각화 시에는 원본 데이터(input_tensor)와 Cutoff 라인을 같이 표시
    _save_plot(xrr_config, input_tensor[0], r_ai_norm[0], r_final_norm[0], 
        data_name, fig_path, p_refined, q_cutoff=actual_cutoff)

    logger.info(f"✨ 완료. 결과 저장됨: {fig_path}")
    print("Refined Params:", p_refined)

    return p_refined


def _save_plot(config, y_meas_log, r_ai, r_ref, title, save_path, p_refined=None, q_cutoff=None):
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
    
    if q_cutoff:
        ax.axvline(x=q_cutoff, color='gray', linestyle='--', alpha=0.6, label=f'Cutoff ({q_cutoff:.2f})')
        # Cutoff 이후 영역 음영 처리
        ax.axvspan(q_cutoff, q.max(), color='gray', alpha=0.1)

    ax.set_xlabel("Q (Å⁻¹)")
    ax.set_ylabel("Reflectivity")
    ax.set_title(f"XRefine: {title}")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

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
    parser.add_argument("--cutoff", type=float, default=None, 
                        help="Manual Q-Cutoff (Default: Auto Detect)")

    args = parser.parse_args()

    if args.preset == "quick":
        config = RefinementConfig.quick_thickness()
    elif args.preset == "high_quality":
        config = RefinementConfig.high_quality()
    else:
        config = None  # Default

    run_real_inference(args.checkpoint, args.data, args.output, config, q_cutoff=args.cutoff)