"""
Loss Functions (Added Gradient Loss)
====================================
"""
import torch
from torch import Tensor
from typing import Callable

LossFn = Callable[[Tensor, Tensor, Tensor], Tensor]

def log_mse_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
    return torch.mean(((log_r_sim - log_r_obs) ** 2) * mask)

def linear_mse_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    r_obs = 10 ** log_r_obs
    return torch.mean(((r_sim - r_obs) ** 2) * mask)

def correlation_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
    sim_masked = log_r_sim[mask]
    obs_masked = log_r_obs[mask]
    vx = sim_masked - sim_masked.mean()
    vy = obs_masked - obs_masked.mean()
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-9)
    return 1.0 - corr

# [NEW] Gradient (Sobolev) Loss
# 그래프의 "기울기"를 비교합니다. 
# 시뮬레이션이 밋밋하게 펴져있으면(기울기 0), 실제 데이터(기울기 급격함)와 차이가 커서 Loss가 폭발합니다.
def log_gradient_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
    
    # 1차 미분 (Central Difference approximation or simple diff)
    # shape: (B, Q-1)
    diff_sim = log_r_sim[:, 1:] - log_r_sim[:, :-1]
    diff_obs = log_r_obs[:, 1:] - log_r_obs[:, :-1]
    mask_diff = mask[:, 1:] & mask[:, :-1] # 두 포인트 모두 유효해야 함
    
    # 미분값의 MSE + 원래 값의 MSE (Hybrid)
    grad_loss = torch.mean(((diff_sim - diff_obs) ** 2) * mask_diff)
    val_loss = torch.mean(((log_r_sim - log_r_obs) ** 2) * mask)
    
    # 기울기 형상에 10배 가중치
    return val_loss + 10.0 * grad_loss

# Factory functions
def make_fringe_weighted_loss(base_loss, fringe_q_range=(0.03, 0.12), fringe_weight=20.0, high_q_damping=0.1):
    def weighted_loss(r_sim, log_r_obs, mask):
        log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
        error = (log_r_sim - log_r_obs) ** 2
        
        n_points = mask.shape[-1]
        q_estimated = torch.linspace(0.0, 0.5, n_points, device=mask.device)
        q_min, q_max = fringe_q_range
        
        weights = torch.ones_like(q_estimated)
        weights[(q_estimated >= q_min) & (q_estimated <= q_max)] = fringe_weight
        weights[q_estimated > q_max] = high_q_damping
        
        return torch.sum(error * weights * mask) / (torch.sum(mask * weights) + 1e-9)
    return weighted_loss

def hybrid_linear_log_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    # 1. Linear Component (Intensity & Critical Angle)
    r_obs = 10 ** log_r_obs
    loss_lin = torch.mean(((r_sim - r_obs) ** 2) * mask)
    
    # 2. Log Component (Fringes & Dynamic Range)
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
    loss_log = torch.mean(((log_r_sim - log_r_obs) ** 2) * mask)
    
    # 3. Cross Balancing (Linear:Log = 20:1)
    return (20.0 * loss_lin) + loss_log

# Presets
LOG_MSE = log_mse_loss
LINEAR_MSE = linear_mse_loss
CORRELATION = correlation_loss
LOG_GRADIENT = log_gradient_loss # Export
FRINGE_WEIGHTED_LOG = make_fringe_weighted_loss(log_mse_loss)
HYBRID_LOSS = hybrid_linear_log_loss