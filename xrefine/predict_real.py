import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from typing import Dict, Set
from datetime import datetime

from xrefine.config import CONFIG, XRefineConfig
from xrefine.models import XRR_Fourier_PINN
from xrefine.physics_utils import tth2q
from xrefine.logger import logger
from xrefine.core.structures import ParamSet


def load_real_data(file_path: str, config: XRefineConfig):
    """실제 데이터를 로드하고 정규화를 수행합니다."""
    df = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python', dtype=str)
    for col in df.columns:
        df[col] = df[col].str.replace('(', '', regex=False).str.replace(')', '', regex=False).astype(float)

    x_raw, y_raw = df.values[:, 0], df.values[:, 1]
    x_raw = tth2q(x_raw, config.instrument.wavelength)

    # 최대값 = 1.0으로 정규화
    norm_factor = y_raw.max()
    y_norm = y_raw / (norm_factor + 1e-12)
    log_y_raw = np.log10(np.clip(y_norm, 1e-11, 1.0))

    q_target = config.q_grid.numpy()
    log_y_interp = np.interp(q_target, x_raw, log_y_raw)
    log_y_interp[(q_target < x_raw.min()) | (q_target > x_raw.max())] = -15.0
    
    return torch.from_numpy(log_y_interp).float().view(1, -1)


def calculate_r2(y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor) -> float:
    """R² (결정 계수) 계산"""
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    ss_res = torch.sum((y_true_masked - y_pred_masked) ** 2)
    ss_tot = torch.sum((y_true_masked - y_true_masked.mean()) ** 2)
    
    r2 = 1 - (ss_res / (ss_tot + 1e-12))
    return float(r2.item())


def three_stage_refinement(model, p_ai: ParamSet, log_r_obs: torch.Tensor, config: XRefineConfig):
    """
    3단계 최적화:
    0단계 - 풋프린트 파라미터 (L, beam_width) - 임계각 이전 영역만 사용
    1단계 - 기타 Instrumental (bkg) - 전체 영역
    2단계 - Physics 파라미터 - 전체 영역
    """
    
    FOOTPRINT_PARAMS = {'L', 'beam_width'}
    OTHER_INSTRUMENTAL = {'bkg', 'dq', 'tth_offset'}
    PHYSICS_PARAMS = {'d_f', 'sig_f', 'rho_f', 'd_ox', 'sig_ox', 'rho_ox'}
    FIXED_PARAMS = {'i0': 1.0}
    
    # 임계각 위치 감지 (피크 위치)
    r_obs_linear = 10 ** log_r_obs.squeeze().cpu().numpy()
    q_grid = config.q_grid.numpy()
    peak_idx = np.argmax(r_obs_linear)
    q_critical = q_grid[peak_idx]
    
    logger.info(f"📍 임계각 위치 감지: q_c = {q_critical:.4f} Å⁻¹ (index {peak_idx})")
    
    # 0단계: 풋프린트 파라미터 최적화 (임계각 이전 영역만)
    logger.info("👣 [0단계] 풋프린트 최적화 (q < q_critical 영역)...")
    p_stage0 = _optimize_subset(
        model, p_ai, log_r_obs, config,
        optimize_keys=FOOTPRINT_PARAMS,
        forced_params=FIXED_PARAMS,
        q_max=q_critical,
        use_r2_loss=True,
        maxiter=80, popsize=15
    )
    
    # 1단계: 기타 instrumental 최적화 (전체 영역)
    logger.info("🔧 [1단계] Background 최적화 (전체 영역)...")
    p_stage1 = _optimize_subset(
        model, p_stage0, log_r_obs, config,
        optimize_keys=OTHER_INSTRUMENTAL,
        forced_params=FIXED_PARAMS,
        use_r2_loss=True,
        maxiter=50, popsize=10
    )
    
    # 2단계: Physics 파라미터 최적화
    logger.info("🧬 [2단계] Physics 파라미터 최적화...")
    p_final = _optimize_subset(
        model, p_stage1, log_r_obs, config,
        optimize_keys=PHYSICS_PARAMS,
        forced_params=FIXED_PARAMS,
        use_r2_loss=False,
        maxiter=200, popsize=20
    )
    
    return p_final


def _optimize_subset(
    model, p_init: ParamSet, log_r_obs: torch.Tensor, config: XRefineConfig,
    optimize_keys: Set[str], forced_params: Dict[str, float] = None,
    q_max: float = None, use_r2_loss: bool = False, 
    maxiter: int = 150, popsize: int = 20
) -> ParamSet:
    """특정 파라미터 그룹만 최적화."""
    target_device = next(model.parameters()).device
    range_config = config.ranges
    
    log_r_obs_gpu = log_r_obs.to(target_device)
    mask = (log_r_obs_gpu > -14.0)
    
    # q_max가 지정되면 해당 영역만 마스크에 포함
    if q_max is not None:
        q_grid = config.q_grid.to(target_device)
        q_mask = (q_grid < q_max).unsqueeze(0)
        mask = mask & q_mask
        logger.info(f"   → 마스크 영역: q < {q_max:.4f} ({mask.sum().item()} points)")

    opt_param_info = []
    opt_guess_norm = []
    fixed_params = {}
    
    for k, v in p_init.items():
        if forced_params and k in forced_params:
            fixed_params[k] = torch.tensor([[forced_params[k]]], dtype=torch.float32, device=target_device)
            continue
            
        p_min, p_max = getattr(range_config, k)
        val_flat = v.detach().cpu().numpy().flatten()
        
        if k in optimize_keys:
            opt_param_info.append({'key': k, 'size': val_flat.size, 'min': p_min, 'max': p_max})
            norm_val = (val_flat - p_min) / (p_max - p_min + 1e-9)
            opt_guess_norm.extend(np.clip(norm_val, 0.0, 1.0).tolist())
        else:
            fixed_params[k] = torch.tensor(val_flat, dtype=torch.float32, device=target_device).view(1, -1)

    total_dim = len(opt_guess_norm)
    if total_dim == 0:
        logger.warning("최적화할 파라미터가 없습니다.")
        return p_init

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
                
                # 피크 정규화
                r_sim_peak = r_sim.max()
                r_sim_normalized = r_sim / (r_sim_peak + 1e-12)
                log_r_sim = torch.log10(torch.clamp(r_sim_normalized, min=1e-12))
                
                if use_r2_loss:
                    r2 = calculate_r2(log_r_obs_gpu.squeeze(), log_r_sim.squeeze(), mask.squeeze())
                    loss = -r2
                else:
                    loss = torch.mean(((log_r_sim - log_r_obs_gpu)**2) * mask)
                
                total_val = loss + (penalty * config.train.phys_penalty_w)
                return float(total_val.item()) if not use_r2_loss else float(total_val)
        
        except Exception as e:
            logger.error(f"❌ Objective 오류: {e}")
            raise e

    # Latin Hypercube 초기화
    sampler = qmc.LatinHypercube(d=total_dim)
    init_pop = sampler.random(n=popsize)
    init_pop[0] = np.array(opt_guess_norm)

    result = differential_evolution(
        objective,
        bounds=[(0.0, 1.0)] * total_dim,
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        tol=0.001,
        atol=1e-8,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,
        init=init_pop,
        disp=False,
        workers=1
    )
    
    metric_name = "R²" if use_r2_loss else "MSE"
    metric_val = -result.fun if use_r2_loss else result.fun
    logger.info(f"✅ 완료: {result.nit} iterations, {metric_name} = {metric_val:.6f}")

    # 결과 조립
    final_dict = {k: v.cpu() for k, v in fixed_params.items()}
    cursor = 0
    for info in opt_param_info:
        phys_val = result.x[cursor : cursor + info['size']] * (info['max'] - info['min']) + info['min']
        final_dict[info['key']] = torch.tensor(phys_val, dtype=torch.float32).view(1, info['size'])
        cursor += info['size']
        
    return ParamSet(**final_dict)


@torch.no_grad()
def run_real_inference(checkpoint_path: str, data_path: str, output_dir: str = "./results"):
    """실제 XRR 데이터에 대한 추론 및 피팅 실행"""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = XRefineConfig(**ckpt['config'])
    
    model = XRR_Fourier_PINN(config=config).to(config.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    data_name = Path(data_path).stem
    
    # 1. AI 초기 추측
    input_tensor = load_real_data(data_path, config)
    p_ai, r_ai, _ = model(input_tensor.to(config.device))

    # 2. 3단계 최적화 실행
    p_refined = three_stage_refinement(model, p_ai, input_tensor, config)
    
    # 3. 최종 결과 생성 (피크 정규화 적용)
    p_final, r_final, _ = model.forward_with_params(p_refined.to(config.device))
    r_final_normalized = r_final / (r_final.max() + 1e-12)
    r_ai_normalized = r_ai / (r_ai.max() + 1e-12)
    
    # 4. 통계 계산
    log_r_obs = input_tensor.to(config.device)
    mask = (log_r_obs > -14.0)
    log_r_final = torch.log10(torch.clamp(r_final_normalized, min=1e-12))
    log_r_ai = torch.log10(torch.clamp(r_ai_normalized, min=1e-12))
    
    r2_final = calculate_r2(log_r_obs.squeeze(), log_r_final.squeeze(), mask.squeeze())
    r2_ai = calculate_r2(log_r_obs.squeeze(), log_r_ai.squeeze(), mask.squeeze())
    mse_final = float(torch.mean(((log_r_final - log_r_obs)**2) * mask).item())
    mse_ai = float(torch.mean(((log_r_ai - log_r_obs)**2) * mask).item())

    # 5. 결과 출력
    print("\n" + "="*60)
    print(f"📊 XRefine 피팅 결과: {data_name}")
    print("="*60)
    print(f"\n{'메트릭':<20} {'AI 초기값':<15} {'최적화 후':<15}")
    print("-"*50)
    print(f"{'R²':<20} {r2_ai:<15.4f} {r2_final:<15.4f}")
    print(f"{'MSE (log scale)':<20} {mse_ai:<15.6f} {mse_final:<15.6f}")
    
    print(f"\n{'파라미터':<15} {'AI 초기값':<15} {'최적화 후':<15}")
    print("-"*50)
    for k in p_ai.keys():
        ai_val = getattr(p_ai, k).item()
        final_val = getattr(p_final, k).item()
        print(f"{k:<15} {ai_val:<15.4f} {final_val:<15.4f}")
    print("="*60 + "\n")

    # 6. 그래프 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / f"{data_name}_fit_{datetime.now().strftime('%H%M%S')}.png"
    
    _save_plot(config, input_tensor[0], r_ai_normalized[0], r_final_normalized[0], 
               data_name, r2_final, fig_path)
    
    logger.info(f"📈 그래프 저장: {fig_path}")


def _save_plot(config, y_meas_log, r_ai, r_ref, title, r2, save_path):
    q = config.q_grid.numpy()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.semilogy(q, 10**y_meas_log.cpu().numpy(), 'ko', ms=3, alpha=0.3, label='Measured Data')
    ax.semilogy(q, r_ai.cpu().numpy(), 'b--', alpha=0.5, label='AI Initial Guess')
    ax.semilogy(q, r_ref.cpu().numpy(), 'r-', lw=2, label=f'Optimized Fit (R²={r2:.4f})')
    
    ax.set_title(f"XRefine Result: {title}", fontsize=14)
    ax.set_xlabel("q (Å⁻¹)", fontsize=12)
    ax.set_ylabel("Reflectivity", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    CHECKPOINT = r"D:\02_Projects\Dev\X-ray_AI\XRefine\checkpoints_test\TestRun\best.pt"
    DATA_FILE = r"D:\03_Resources\Data\XRR_AI\XRR_data\XRR_data2\#4.dat"
    run_real_inference(CHECKPOINT, DATA_FILE)