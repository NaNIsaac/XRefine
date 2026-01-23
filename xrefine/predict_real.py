from pathlib import Path
from datetime import datetime
from copy import deepcopy
from typing import Set, Dict, Optional

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks
from scipy.stats import qmc

from xrefine.config import CONFIG, XRefineConfig
from xrefine.models import XRR_Fourier_PINN
from xrefine.physics_utils import tth2q
from xrefine.logger import logger
from xrefine.core.structures import ParamSet

def calculate_r2(y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor) -> float:
    """
    R² (결정 계수) 계산 함수
    """
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    if len(y_true_masked) == 0:
        return -1.0
        
    ss_res = torch.sum((y_true_masked - y_pred_masked) ** 2)
    ss_tot = torch.sum((y_true_masked - y_true_masked.mean()) ** 2)
    
    # 분모가 0이 되는 것을 방지
    r2 = 1 - (ss_res / (ss_tot + 1e-12))
    return float(r2.item())

def load_real_data(file_path: str, config: XRefineConfig):
    """
    Load and Normalize Data using configurable normalization strategy.
    
    Normalization mode is controlled by config.data.normalization_mode.
    """
    from xrefine.data.normalization import apply_normalization
    
    # 데이터 로드
    df = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python', dtype=str)
    for col in df.columns:
        df[col] = df[col].str.replace('(', '', regex=False).str.replace(')', '', regex=False).astype(float)

    x_raw_tth, y_raw = df.values[:, 0], df.values[:, 1]
    
    # 2theta -> q 변환
    x_raw_q = tth2q(x_raw_tth, config.instrument.wavelength)

    # Configurable Normalization
    y_norm = apply_normalization(y_raw, config, q_array=x_raw_q, tth_array=x_raw_tth)
    log_y_raw = np.log10(np.maximum(y_norm, 1e-11))

    # Grid Interpolation
    q_target = config.q_grid.numpy()
    log_y_interp = np.interp(q_target, x_raw_q, log_y_raw)
    
    # 측정 범위 밖 마스킹 (-15.0)
    log_y_interp[(q_target < x_raw_q.min()) | (q_target > x_raw_q.max())] = -15.0
    
    return torch.from_numpy(log_y_interp).float().view(1, -1)

def _optimize_subset(
    model, p_init: ParamSet, log_r_obs: torch.Tensor, config: XRefineConfig,
    optimize_keys: Set[str], forced_params: Dict[str, float] = None,
    q_min: Optional[float] = None, q_max: Optional[float] = None, 
    use_gradient_loss: bool = False, use_linear_loss: bool = False,
    use_r2_loss: bool = False,
    maxiter: int = 100, popsize: int = 20
) -> ParamSet:
    """
    부분 파라미터 최적화 엔진 (Scipy DE 기반)
    """
    target_device = next(model.parameters()).device
    range_config = config.ranges
    
    # 타겟 데이터 및 마스크 준비
    log_r_obs_gpu = log_r_obs.to(target_device)
    mask = (log_r_obs_gpu > -14.0)
    
    # Q-Range Cutting
    q_grid = config.q_grid.to(target_device)
    if q_max is not None: mask &= (q_grid < q_max).unsqueeze(0)
    if q_min is not None: mask &= (q_grid > q_min).unsqueeze(0)

    # 파라미터 Flattening 및 범위 설정
    opt_param_info = []
    opt_guess_norm = []
    fixed_params = {}
    
    for k, v in p_init.items():
        if forced_params and k in forced_params:
            # 강제 고정 (예: Roughness Locking)
            fixed_params[k] = torch.tensor([[forced_params[k]]], dtype=torch.float32, device=target_device)
            continue
            
        p_min, p_max = getattr(range_config, k)
        val_flat = v.detach().cpu().numpy().flatten()
        
        if k in optimize_keys:
            # 최적화 대상
            opt_param_info.append({'key': k, 'size': val_flat.size, 'min': p_min, 'max': p_max})
            # [DEBUG] Check bounds
            logger.info(f"🔎 Param {k}: Bounds [{p_min:.2f}, {p_max:.2f}]")
            
            norm_val = (val_flat - p_min) / (p_max - p_min + 1e-9)
            opt_guess_norm.extend(np.clip(norm_val, 0.0, 1.0).tolist())
        else:
            # 고정 대상 (AI 초기값 유지)
            fixed_params[k] = torch.tensor(val_flat, dtype=torch.float32, device=target_device).view(1, -1)

    total_dim = len(opt_guess_norm)
    if total_dim == 0: return p_init

    # 목적 함수 정의
    def objective(trial_vector):
        try:
            p_dict = dict(fixed_params)
            cursor = 0
            for info in opt_param_info:
                k, size, p_min, p_max = info['key'], info['size'], info['min'], info['max']
                phys_val = trial_vector[cursor : cursor + size] * (p_max - p_min) + p_min
                p_dict[k] = torch.tensor(phys_val, dtype=torch.float32, device=target_device).view(1, size)
                cursor += size

            p_set = ParamSet(**p_dict)

            with torch.no_grad():
                _, r_sim, penalty = model.forward_with_params(p_set)
                log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))

                # [LOSS 1] 기본 MSE (Log or Linear)
                if use_linear_loss:
                    # Linear R 비교 (프린지 진폭에 민감)
                    r_obs_linear = 10 ** log_r_obs_gpu
                    mse_loss = torch.mean(((r_sim - r_obs_linear)**2) * mask)
                else:
                    mse_loss = torch.mean(((log_r_sim - log_r_obs_gpu)**2) * mask)
                loss = mse_loss

                # [LOSS 2] Gradient Loss (Fringe 형상 보존)
                if use_gradient_loss:
                    diff_sim = torch.diff(log_r_sim, dim=-1)
                    diff_obs = torch.diff(log_r_obs_gpu, dim=-1)
                    diff_mask = mask[:, 1:]
                    grad_loss = torch.mean(((diff_sim - diff_obs)**2) * diff_mask)
                    loss += grad_loss * 5.0 

                # [LOSS 3] R2 Score (선택적)
                if use_r2_loss:
                    r2 = calculate_r2(log_r_obs_gpu.squeeze(), log_r_sim.squeeze(), mask.squeeze())
                    # R2 최대화 = -R2 최소화
                    loss = -r2

                # [Penalty] 물리적 제약 (너무 큰 Roughness 방지 제거됨)
                r_pen = 0.0
                # 기존 sig > 5.0 페널티 제거 (Damping 맞추기 위해 더 높은 Roughness 허용)
                
                # 2. [Physical Constraint] Roughness < Thickness / 2.0
                # 거칠기가 두께의 절반을 넘으면 물리적으로 층이 유지될 수 없음
                if p_set.sig_f.item() > (p_set.d_f.item() / 2.0):
                    r_pen += 100.0 + (p_set.sig_f.item() - p_set.d_f.item()/2.0) * 10.0
                
                if p_set.sig_ox.item() > (p_set.d_ox.item() / 2.0):
                    r_pen += 100.0 + (p_set.sig_ox.item() - p_set.d_ox.item()/2.0) * 10.0

                return float((loss + penalty + r_pen).item())
                
        except Exception as e:
            # 디버깅용 로그
            # logger.error(f"Objective Error: {e}") 
            return 1e9

    # Scipy DE 실행
    sampler = qmc.LatinHypercube(d=total_dim)
    init_pop = sampler.random(n=popsize)
    init_pop[0] = np.array(opt_guess_norm) # AI 초기값 주입

    result = differential_evolution(
        objective, bounds=[(0, 1)]*total_dim, strategy='best1bin',
        maxiter=maxiter, popsize=popsize, mutation=(0.5, 1.0), recombination=0.7,
        init=init_pop, disp=False, workers=1
    )

    # 결과 복원
    final_dict = {k: v.cpu() for k, v in fixed_params.items()}
    cursor = 0
    for info in opt_param_info:
        phys_val = result.x[cursor : cursor + info['size']] * (info['max'] - info['min']) + info['min']
        final_dict[info['key']] = torch.tensor(phys_val, dtype=torch.float32).view(1, info['size'])
        cursor += info['size']

    return ParamSet(**final_dict)

def three_stage_refinement(model, p_ai: ParamSet, log_r_obs: torch.Tensor, config: XRefineConfig):
    """
    ALD 특화 3단계 최적화 전략
    """
    Q_CUTOFF = 0.45  # 노이즈가 심해지는 고각도 영역 커팅
    
    # 임계각 위치 감지 (피크 위치)
    r_obs_linear = 10 ** log_r_obs.squeeze().cpu().numpy()
    q_grid = config.q_grid.numpy()
    peak_idx = np.argmax(r_obs_linear)
    q_critical = q_grid[peak_idx]
    logger.info(f"📍 임계각 위치 감지: q_c = {q_critical:.4f} Å⁻¹ (index {peak_idx})")

    # [0단계] 풋프린트 & 배경 (Low Q)
    logger.info("👣 [0단계] 풋프린트 & Background (Q < 0.15)...")
    p_stage0 = _optimize_subset(
        model, p_ai, log_r_obs, config,
        optimize_keys={'beam_width', 'L', 'i0', 'bkg'},
        q_max=0.15, maxiter=50, popsize=20
    )

    # [1단계] 두께 Grid Search (Global Search)
    # Gradient Descent 대신 1D Grid Scan을 통해 Global Minimum을 확실히 찾음
    # Metric: 1 - Correlation (패턴 매칭)
    logger.info("📐 [1단계] Thickness Grid Search (Correlation 기반, 10~1200A)...")

    d_grid = np.linspace(20, 1200, 500) # 20A ~ 1200A, 약 2.4A 간격
    best_corr_loss = 1e9
    best_d = p_stage0.d_f.item()

    # Grid Search용 임시 파라미터 (Stage0 기반)
    p_scan = deepcopy(p_stage0)
    # Oxide/Roughness 제거하여 메인 주기만 확인
    p_scan.d_ox.fill_(0.0)
    p_scan.sig_ox.fill_(0.0)
    p_scan.rho_ox.fill_(0.0)
    p_scan.sig_f.fill_(2.0) # 기본 Roughness

    # Q-Mask for Correlation (0.05 ~ cutoff)
    q_grid_gpu = config.q_grid.to(config.device)
    log_r_obs_gpu = log_r_obs.to(config.device)
    mask_corr = (log_r_obs_gpu > -14.0) & (q_grid_gpu > 0.05) & (q_grid_gpu < Q_CUTOFF)
    target_pattern = log_r_obs_gpu[mask_corr]
    target_mean = target_pattern.mean()
    
    scores = []
    
    for d_val in d_grid:
        p_scan.d_f.fill_(d_val)
        
        with torch.no_grad():
            _, r_sim, _ = model.forward_with_params(p_scan)
            r_sim_norm = r_sim / (r_sim.max() + 1e-12)
            log_r_sim = torch.log10(torch.clamp(r_sim_norm, min=1e-12))
            
            sim_pattern = log_r_sim[mask_corr]
            
            # Pearson Correlation Calculation
            vx = sim_pattern - sim_pattern.mean()
            vy = target_pattern - target_mean
            corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-9)
            
            # Loss = 1 - Correlation
            c_loss = 1.0 - corr.item()
            scores.append(c_loss)
            
            if c_loss < best_corr_loss:
                best_corr_loss = c_loss
                best_d = d_val

    # [Smart Selection] Valley Detection to find true minima
    peaks, _ = find_peaks(-np.array(scores), prominence=0.005) 

    # [Dynamic Filter] Critical Angle 기반 최소 두께 설정
    # 경험적으로 Artifact는 2.0/Qc 이하에서 발생 (e.g. Qc=0.04 -> 50A)
    # q_critical은 three_stage_refinement 함수 초반에 계산됨
    min_d_threshold = 2.0 / (q_critical + 1e-9)
    logger.info(f"🛡️ Dynamic Thickness Filter: d < {min_d_threshold:.1f}A 제외 (based on Qc={q_critical:.4f})")
    
    candidates = []
    for idx in peaks:
        d_cand = d_grid[idx]
        loss_cand = scores[idx]
        
        if d_cand < min_d_threshold:
            logger.info(f"⚠️ 후보 제외 (Too Thin): {d_cand:.1f}A (Loss={loss_cand:.4f})")
            continue
            
        candidates.append((d_cand, loss_cand))
    
    # 만약 적절한 후보가 없으면 그냥 전체 Min 사용
    if not candidates:
        logger.warning("🚫 적절한 Local Minima를 찾지 못함. Global Min 사용.")
        final_best_d = best_d
    else:
        # Loss 순 정렬 후 1등 선택
        candidates.sort(key=lambda x: x[1])
        final_best_d = candidates[0][0]
        logger.info(f"✅ 후보군 Top 3: {[f'{d:.1f}A({l:.4f})' for d, l in candidates[:3]]}")

    logger.info(f"🎯 최종 선택: d_f = {final_best_d:.1f} A")
    
    # 찾은 Best Thickness로 Stage 1 결과 설정
    p_stage1 = deepcopy(p_stage0)
    p_stage1.d_f.fill_(final_best_d)
    # Oxide/Roughness는 일단 0으로 (Stage 2에서 복원)
    p_stage1.d_ox.fill_(0.0)
    p_stage1.sig_ox.fill_(0.0)
    p_stage1.rho_ox.fill_(0.0)
    p_stage1.sig_f.fill_(3.0) 
    
    # [2단계] Linear R 기반 Roughness/Damping 최적화
    logger.info(f"� [2단계] Linear R 피팅 (Fringe 진폭 맞추기, d_f={final_best_d:.1f}A 고정)...")
    
    p_combined = p_stage1
    # AI 초기값에서 Oxide/Roughness 복원
    p_combined.sig_f = torch.tensor([[3.0]], device=p_ai.d_f.device)
    p_combined.d_ox = p_ai.d_ox
    p_combined.sig_ox = p_ai.sig_ox
    p_combined.rho_ox = p_ai.rho_ox
    
    fixed_stage2 = {'d_f': final_best_d}
    
    p_stage2 = _optimize_subset(
        model, p_combined, log_r_obs, config,
        optimize_keys={'sig_f', 'rho_f', 'd_ox', 'sig_ox', 'rho_ox', 'L', 'i0', 'bkg', 'dq'},
        forced_params=fixed_stage2,
        q_max=Q_CUTOFF, 
        use_linear_loss=True,  # Linear R로 프린지 진폭 맞추기
        maxiter=100, popsize=30
    )
    
    # [3단계] Log R 기반 최종 미세 조정
    logger.info("🔬 [3단계] Log R 미세 조정...")
    
    p_stage3 = _optimize_subset(
        model, p_stage2, log_r_obs, config,
        optimize_keys={'sig_f', 'rho_f', 'd_ox', 'sig_ox', 'rho_ox', 'L', 'i0', 'bkg', 'dq'},
        forced_params=fixed_stage2,
        q_max=Q_CUTOFF,
        use_linear_loss=False,  # Log R로 마무리
        maxiter=100, popsize=30
    )
    
    # [4단계] 두께 미세 조정 (±50A 범위)
    logger.info(f"🎯 [4단계] 두께 미세 조정 (d_f ≈ {final_best_d:.1f} ± 50A)...")
    
    # d_f 범위를 좁은 구간으로 제한하기 위해 임시로 config 변경
    original_d_range = config.ranges.d_f
    config.ranges.d_f = (max(10.0, final_best_d - 50.0), final_best_d + 50.0)
    
    p_final = _optimize_subset(
        model, p_stage3, log_r_obs, config,
        # 이제 d_f도 포함 (좁은 범위에서만)
        optimize_keys={'d_f', 'sig_f', 'rho_f', 'd_ox', 'sig_ox', 'rho_ox', 'L', 'i0', 'bkg', 'dq'},
        q_max=Q_CUTOFF,
        use_linear_loss=False,
        maxiter=50, popsize=20
    )
    
    # 원래 범위 복원
    config.ranges.d_f = original_d_range
    
    return p_final

@torch.no_grad()
def run_real_inference(checkpoint_path: str, data_path: str, output_dir: str = "./results"):
    # Checkpoint 로드 (Windows Path 호환성 처리 포함)
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

    # Load Config from Checkpoint but OVERRIDE ranges from current code
    # (Checking point may have old narrow ranges like sig_f < 5.0)
    loaded_config_dict = ckpt['config']
    config = XRefineConfig(**loaded_config_dict)
    
    # [FIXED] 체크포인트 config를 그대로 사용 (ranges 포함)
    # 기존: config.ranges = CONFIG.ranges (하드코딩 제거됨)
    
    model = XRR_Fourier_PINN(config=config).to(config.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    data_name = Path(data_path).stem
    
    # 1. AI 초기 추측
    input_tensor = load_real_data(data_path, config)
    p_ai, r_ai, _ = model(input_tensor.to(config.device))

    # 2. 3단계 최적화
    p_refined = three_stage_refinement(model, p_ai, input_tensor, config)

    # 3. 최종 결과
    p_final, r_final, _ = model.forward_with_params(p_refined.to(config.device))

    # 정규화 (시각화용)
    r_final_norm = r_final / (r_final.max() + 1e-12)
    r_ai_norm = r_ai / (r_ai.max() + 1e-12)

    # 4. 시각화 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / f"{data_name}_fit_{datetime.now().strftime('%H%M%S')}.png"

    _save_plot(config, input_tensor[0], r_ai_norm[0], r_final_norm[0], 
               data_name, calculate_r2(input_tensor.to(config.device).squeeze(), 
                                     torch.log10(torch.clamp(r_final_norm, min=1e-12)).squeeze(), 
                                     (input_tensor.to(config.device)>-14).squeeze()), 
               fig_path)

    logger.info(f"✨ 완료. 결과 저장됨: {fig_path}")
    print("Refined Params:", p_final)

def _save_plot(config, y_meas_log, r_ai, r_ref, title, r2, save_path):
    q = config.q_grid.numpy()
    y_meas = y_meas_log.cpu().numpy()
    
    # 유효 데이터 영역만 표시 (마스킹 영역 제외)
    valid_mask = y_meas > -14.0
    q_valid = q[valid_mask]
    y_valid = 10 ** y_meas[valid_mask]
    r_ai_valid = r_ai.cpu().numpy()[valid_mask]
    r_ref_valid = r_ref.cpu().numpy()[valid_mask]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(q_valid, y_valid, 'ko', ms=2, alpha=0.3, label='Measured')
    ax.semilogy(q_valid, r_ai_valid, 'b--', alpha=0.6, label='AI Init')
    ax.semilogy(q_valid, r_ref_valid, 'r-', lw=2, label=f'Refined (R²={r2:.3f})')
    ax.set_xlabel("Q (Å⁻¹)")
    ax.set_ylabel("Reflectivity")
    ax.set_title(f"XRefine: {title}")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.savefig(save_path)
    plt.close(fig)

if __name__ == "__main__":
    # 경로 설정
    CHECKPOINT = r"D:\03_Resources\Data\XRR_AI\xrefine\v1\best.pt"
    # DATA_FILE = r"D:\03_Resources\Data\XRR_AI\XRR_data\KBJ\ALD_A_no_XRR.dat"
    DATA_FILE = r"D:\03_Resources\Data\XRR_AI\XRR_data\XRR_data\#1_xrr.dat"
    
    run_real_inference(CHECKPOINT, DATA_FILE)