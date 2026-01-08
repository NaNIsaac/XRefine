"""
Loss Functions for XRR Refinement
=================================
1급 객체(함수)로 정의된 Loss Functions

Usage:
    from xrefine.refine.losses import log_mse_loss, linear_mse_loss, correlation_loss
    
    loss_fn = log_mse_loss
    loss = loss_fn(r_sim, r_obs, mask)
"""
import torch
from torch import Tensor
from typing import Callable, Optional

# Type alias for loss function signature
# (r_sim, log_r_obs, mask) -> scalar loss
LossFn = Callable[[Tensor, Tensor, Tensor], Tensor]


def log_mse_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    """
    Log10 공간에서의 MSE Loss
    
    Args:
        r_sim: Simulated reflectivity (linear scale)
        log_r_obs: Observed reflectivity (log10 scale)
        mask: Valid data mask
    
    Returns:
        Scalar loss tensor
    """
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
    return torch.mean(((log_r_sim - log_r_obs) ** 2) * mask)


def linear_mse_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    """
    Linear R 공간에서의 MSE Loss (프린지 진폭에 민감)
    
    Args:
        r_sim: Simulated reflectivity (linear scale)
        log_r_obs: Observed reflectivity (log10 scale)
        mask: Valid data mask
    
    Returns:
        Scalar loss tensor
    """
    r_obs = 10 ** log_r_obs
    return torch.mean(((r_sim - r_obs) ** 2) * mask)


def correlation_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    """
    Pearson Correlation Loss (1 - corr)
    두께 검색에 적합 (패턴 매칭)
    
    Args:
        r_sim: Simulated reflectivity (linear scale)
        log_r_obs: Observed reflectivity (log10 scale)
        mask: Valid data mask
    
    Returns:
        1 - correlation (0 = perfect match)
    """
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
    
    sim_masked = log_r_sim[mask]
    obs_masked = log_r_obs[mask]
    
    vx = sim_masked - sim_masked.mean()
    vy = obs_masked - obs_masked.mean()
    
    corr = torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-9
    )
    
    return 1.0 - corr


def gradient_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor, weight: float = 5.0) -> Tensor:
    """
    Gradient Loss (프린지 형상 보존)
    
    Args:
        r_sim: Simulated reflectivity (linear scale)
        log_r_obs: Observed reflectivity (log10 scale)
        mask: Valid data mask
        weight: Gradient loss weight
    
    Returns:
        Weighted gradient loss
    """
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
    
    diff_sim = torch.diff(log_r_sim, dim=-1)
    diff_obs = torch.diff(log_r_obs, dim=-1)
    diff_mask = mask[:, 1:] if mask.dim() > 1 else mask[1:]
    
    return weight * torch.mean(((diff_sim - diff_obs) ** 2) * diff_mask)


def combined_loss(
    r_sim: Tensor, 
    log_r_obs: Tensor, 
    mask: Tensor,
    base_loss: LossFn = log_mse_loss,
    add_gradient: bool = True,
    gradient_weight: float = 5.0
) -> Tensor:
    """
    Combined Loss (Base + Gradient)
    
    Args:
        r_sim: Simulated reflectivity
        log_r_obs: Observed reflectivity (log10)
        mask: Valid data mask
        base_loss: Base loss function
        add_gradient: Whether to add gradient loss
        gradient_weight: Weight for gradient loss
    
    Returns:
        Combined loss
    """
    loss = base_loss(r_sim, log_r_obs, mask)
    
    if add_gradient:
        loss = loss + gradient_loss(r_sim, log_r_obs, mask, gradient_weight)
    
    return loss


# Factory function to create custom combined losses
def make_combined_loss(
    base: LossFn = log_mse_loss,
    gradient: bool = False,
    gradient_weight: float = 5.0
) -> LossFn:
    """
    커스텀 Combined Loss 생성 팩토리
    
    Usage:
        my_loss = make_combined_loss(linear_mse_loss, gradient=True, gradient_weight=3.0)
    """
    def loss_fn(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
        return combined_loss(r_sim, log_r_obs, mask, base, gradient, gradient_weight)
    
    return loss_fn


# Preset losses for convenience
LOG_MSE = log_mse_loss
LINEAR_MSE = linear_mse_loss
CORRELATION = correlation_loss
LOG_MSE_WITH_GRAD = make_combined_loss(log_mse_loss, gradient=True)
LINEAR_MSE_WITH_GRAD = make_combined_loss(linear_mse_loss, gradient=True)
