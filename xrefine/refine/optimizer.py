"""
Optimizer Step
==============
단일 최적화 Step 실행 (Differential Evolution 기반)
"""
import torch
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from typing import Set, Dict, Optional

from xrefine.core.structures import ParamSet
from xrefine.config import XRefineConfig
from xrefine.refine.config import StageConfig
from xrefine.refine.losses import LossFn, log_mse_loss
from xrefine.logger import logger


def optimize_step(
    model,
    p_init: ParamSet,
    log_r_obs: torch.Tensor,
    xrr_config: XRefineConfig,
    stage_config: StageConfig,
    verbose: bool = True
) -> ParamSet:
    """
    단일 최적화 스텝 실행
    
    Args:
        model: XRR_Fourier_PINN 모델
        p_init: 초기 파라미터
        log_r_obs: 관측된 반사율 (log10 scale)
        xrr_config: XRR 설정 (q_grid, ranges 등)
        stage_config: 스테이지별 최적화 설정
        verbose: 로그 출력 여부
    
    Returns:
        최적화된 ParamSet
    """
    target_device = xrr_config.device
    range_config = xrr_config.ranges
    q_grid = xrr_config.q_grid.to(target_device)
    log_r_obs_gpu = log_r_obs.to(target_device)
    
    # Q-Range 마스킹
    q_min = stage_config.q_min or 0.0
    q_max = stage_config.q_max or xrr_config.q_grid.max().item()
    mask = (log_r_obs_gpu > -14.0) & (q_grid > q_min) & (q_grid < q_max)
    
    # 파라미터 분류: 최적화 대상 vs 고정
    optimize_keys = stage_config.optimize_keys
    fixed_params = {}
    opt_param_info = []
    opt_guess_norm = []
    
    for k, v in p_init.items():
        # 강제 고정 파라미터
        if k in stage_config.fixed_params:
            fixed_val = stage_config.fixed_params[k]
            fixed_params[k] = torch.tensor([[fixed_val]], dtype=torch.float32, device=target_device)
            continue
        
        p_min, p_max = getattr(range_config, k)
        val_flat = v.detach().cpu().numpy().flatten()
        
        if k in optimize_keys:
            opt_param_info.append({'key': k, 'size': val_flat.size, 'min': p_min, 'max': p_max})
            if verbose:
                logger.debug(f"🔎 Param {k}: Bounds [{p_min:.2f}, {p_max:.2f}]")
            norm_val = (val_flat - p_min) / (p_max - p_min + 1e-9)
            opt_guess_norm.extend(np.clip(norm_val, 0.0, 1.0).tolist())
        else:
            fixed_params[k] = torch.tensor(val_flat, dtype=torch.float32, device=target_device).view(1, -1)
    
    total_dim = len(opt_guess_norm)
    if total_dim == 0:
        return p_init
    
    # Loss 함수 가져오기
    loss_fn: LossFn = stage_config.loss_fn
    
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
                
                # 1급 객체 Loss 함수 호출
                loss = loss_fn(r_sim, log_r_obs_gpu, mask)
                
                # ========== Minimal Physical Penalties ==========
                phys_penalty = 0.0
                sig_f = p_set.sig_f.item()
                sig_ox = p_set.sig_ox.item()
                d_ox = p_set.d_ox.item()
                
                # [1] Roughness 상한 페널티 (σ_f > 15Å 이면 페널티)
                if sig_f > 15.0:
                    phys_penalty += 1.0 * (sig_f - 15.0) ** 2
                
                # [2] Oxide roughness는 oxide 두께의 50% 넘으면 비물리적
                if sig_ox > d_ox * 0.5 and d_ox > 5.0:
                    phys_penalty += 10.0
                
                return float((loss + penalty + phys_penalty).item())
                
        except Exception as e:
            return 1e9
    
    # Scipy DE 실행
    sampler = qmc.LatinHypercube(d=total_dim)
    init_pop = sampler.random(n=stage_config.popsize)
    init_pop[0] = np.array(opt_guess_norm)
    
    result = differential_evolution(
        objective, 
        bounds=[(0, 1)] * total_dim, 
        strategy='best1bin',
        maxiter=stage_config.maxiter, 
        popsize=stage_config.popsize, 
        mutation=(0.5, 1.0), 
        recombination=0.7,
        init=init_pop, 
        disp=False, 
        workers=1
    )
    
    # 결과 복원
    final_dict = {k: v.cpu() for k, v in fixed_params.items()}
    cursor = 0
    for info in opt_param_info:
        phys_val = result.x[cursor : cursor + info['size']] * (info['max'] - info['min']) + info['min']
        final_dict[info['key']] = torch.tensor(phys_val, dtype=torch.float32).view(1, info['size'])
        cursor += info['size']
    
    return ParamSet(**final_dict)
