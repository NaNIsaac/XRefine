"""
XRR Augmentation Module (Cleaned & Optimized)
=============================================
GPU-accelerated physics augmentation without CPU synchronization overhead.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from xrefine.config import XRefineConfig
from xrefine.core.structures import ParamSet


class XRRAugmentor(nn.Module):
    def __init__(self, config: XRefineConfig):
        super().__init__()
        self.cfg = config
        self.aug_cfg = config.augmentations
        self.q_grid = config.q_grid.to(config.device)
        self.total_angle = self.cfg.q_max_tth - self.cfg.q_min_tth

    def apply_smearing(self, R: torch.Tensor, res_sigma: torch.Tensor) -> torch.Tensor:
        """
        Batch-wise Gaussian Smearing (Convolution).
        Assumes uniform q-grid for kernel generation (approximation for speed).
        """
        B, Q = R.shape
        # dq as Tensor
        dq = (self.q_grid[-1] - self.q_grid[0]) / (Q - 1)

        # Calculate Kernel Size (Scalar calc on CPU is okay here for setup)
        # But to be safe for compilation, use tensor ops or fixed max size
        max_sigma_val = self.aug_cfg.res_sigma_range[1]

        # We use a fixed dq estimate for kernel sizing to avoid dynamic shapes
        dq_est = (self.cfg.q_max_tth - self.cfg.q_min_tth) * 0.0408 / (Q - 1)  # approx
        max_sigma_px = max_sigma_val / dq_est
        kernel_size = int(max_sigma_px * 6)

        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            return R

        padding = kernel_size // 2

        x = torch.linspace(-padding, padding, kernel_size, device=R.device)

        # sigma_px: (B, 1)
        sigma_px = res_sigma / (dq + 1e-9)
        kernel = torch.exp(-0.5 * (x.view(1, kernel_size) / sigma_px) ** 2)
        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-9)

        # Input: (1, B, Q)
        # Weight: (B, 1, K)
        inp = R.unsqueeze(1)
        inp_padded = F.pad(inp, (padding, padding), mode="replicate")
        kernel = kernel.unsqueeze(1)

        # conv1d groups=B applies separate kernel to each batch item
        smeared = F.conv1d(inp_padded.view(1, B, -1), kernel, bias=None, groups=B)

        return smeared.view(B, Q)

    def apply_resampling(self, R: torch.Tensor) -> torch.Tensor:
        """
        Variable Density Resampling (CPU-Async Optimized).
        Uses numpy random for size calculation to avoid GPU sync.
        """
        B, Q = R.shape

        # 20% Chance to skip
        if np.random.rand() < 0.2:
            return R

        step_min, step_max = self.aug_cfg.measure_step_range

        # Split batch into groups for diversity
        num_groups = 4 if B >= 4 else 1
        splits = torch.chunk(R, num_groups, dim=0)
        results = []

        for split in splits:
            # [Optimization] Generate random step on CPU -> Float immediately
            # This avoids cudaStreamSynchronize caused by tensor.item()
            step_size = np.random.uniform(step_min, step_max)

            # Calculate target length (Python int)
            target_len = int(self.total_angle / step_size)
            target_len = max(min(target_len, Q), 10)

            split_in = split.unsqueeze(1)

            # downsample then upsample back to Q points; the round trip loses information
            split_coarse = F.interpolate(
                split_in, size=target_len, mode="linear", align_corners=True
            )

            split_recon = F.interpolate(split_coarse, size=Q, mode="linear", align_corners=True)
            results.append(split_recon.squeeze(1))

        return torch.cat(results, dim=0)

    @torch.no_grad()
    def forward(self, q: torch.Tensor, R: torch.Tensor, params: ParamSet) -> torch.Tensor:
        """Pipeline: Resampling -> Scaling -> Noise -> Masking"""
        B, Q = R.shape
        device = R.device
        n_q = Q

        R = torch.nan_to_num(R, nan=1e-11, posinf=1.0, neginf=1e-11)

        R_aug = self.apply_resampling(R)

        R_aug = R_aug * params.i0 + torch.pow(10.0, params.bkg)
        R_aug = torch.nan_to_num(R_aug, nan=1e-11).clamp(min=0.0)

        noise_factor = 1.0 / (self.aug_cfg.intensity_noise_scale + 1e-6)
        s_rand = torch.pow(10.0, torch.rand(B, 1, device=device) * 2 + 5) * noise_factor

        poisson_rate = torch.clamp(R_aug * s_rand, min=0.0)
        poisson_rate = torch.nan_to_num(poisson_rate, nan=0.0)

        R_aug = torch.poisson(poisson_rate) / (s_rand + 1e-9)

        # (A) Low-Q Masking
        low_q_cut = torch.rand(B, 1, device=device) * 0.015 + 0.005
        mask_low = q < low_q_cut
        R_aug[mask_low] = 1e-15

        # (B) High-Q Smart/Random Cutoff
        use_smart_cutoff = torch.rand(B, 1, device=device) < 0.8

        # Smart Logic
        floor_threshold = torch.pow(10.0, params.bkg) * 5.0
        is_below_floor = R_aug < floor_threshold
        first_floor_idx = torch.argmax(is_below_floor.int(), dim=1).unsqueeze(1)

        floor_not_found = first_floor_idx == 0
        smart_base_idx = torch.where(
            floor_not_found, torch.tensor(n_q, device=device), first_floor_idx
        )

        margin = torch.randint(50, 150, (B, 1), device=device)
        smart_indices = torch.clamp(smart_base_idx + margin, min=int(n_q * 0.4), max=n_q)

        # Random Logic
        random_indices = torch.randint(int(n_q * 0.4), n_q + 1, (B, 1), device=device)

        # Combine
        cutoff_indices = torch.where(use_smart_cutoff, smart_indices, random_indices)

        indices = torch.arange(n_q, device=device).unsqueeze(0)
        mask_high = indices >= cutoff_indices
        R_aug[mask_high] = 1e-15

        return torch.clamp(R_aug, min=1e-15)
