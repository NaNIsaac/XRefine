"""
Loss Functions (Added Gradient Loss)
====================================
"""

from collections.abc import Callable

import torch
from torch import Tensor

LossFn = Callable[[Tensor, Tensor, Tensor], Tensor]


def log_mse_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
    return torch.mean(((log_r_sim - log_r_obs) ** 2) * mask)


def linear_mse_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    r_obs = 10**log_r_obs
    return torch.mean(((r_sim - r_obs) ** 2) * mask)


def correlation_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
    sim_masked = log_r_sim[mask]
    obs_masked = log_r_obs[mask]
    vx = sim_masked - sim_masked.mean()
    vy = obs_masked - obs_masked.mean()
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-9)
    return 1.0 - corr


# Gradient (Sobolev) loss: compares the slopes of the curves.
# A flat simulation (slope 0) differs strongly from real data with steep slopes, so the loss grows
# large.
def log_gradient_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))

    # shape: (B, Q-1)
    diff_sim = log_r_sim[:, 1:] - log_r_sim[:, :-1]
    diff_obs = log_r_obs[:, 1:] - log_r_obs[:, :-1]
    mask_diff = mask[:, 1:] & mask[:, :-1]

    grad_loss = torch.mean(((diff_sim - diff_obs) ** 2) * mask_diff)
    val_loss = torch.mean(((log_r_sim - log_r_obs) ** 2) * mask)

    return val_loss + 10.0 * grad_loss


# Factory functions
def make_fringe_weighted_loss(
    base_loss, fringe_q_range=(0.03, 0.12), fringe_weight=20.0, high_q_damping=0.1
):
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
    # linear part: intensity and critical angle
    r_obs = 10**log_r_obs
    loss_lin = torch.mean(((r_sim - r_obs) ** 2) * mask)

    # log part: fringes and dynamic range
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
    loss_log = torch.mean(((log_r_sim - log_r_obs) ** 2) * mask)

    # weights linear:log = 20:1
    return (20.0 * loss_lin) + loss_log


# Presets
LOG_MSE = log_mse_loss
LINEAR_MSE = linear_mse_loss
CORRELATION = correlation_loss
LOG_GRADIENT = log_gradient_loss  # Export
FRINGE_WEIGHTED_LOG = make_fringe_weighted_loss(log_mse_loss)
HYBRID_LOSS = hybrid_linear_log_loss
