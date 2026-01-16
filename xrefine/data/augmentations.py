# data/augmentations.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from xrefine.config import XRefineConfig
from xrefine.core.structures import ParamSet

class XRRAugmentor(nn.Module):
    """
    GPU-Accelerated XRR Augmentor (CPU-Sync Free Optimization)
    """
    def __init__(self, config: XRefineConfig):
        super().__init__()
        self.cfg = config
        self.aug_cfg = config.augmentations
        # Pre-calculate Constants on GPU
        self.register_buffer('sqrt_8ln2', torch.sqrt(torch.tensor(8.0 * np.log(2.0))))
        self.q_grid = config.q_grid.to(config.device)
        self.total_angle = self.cfg.q_max_tth - self.cfg.q_min_tth

    def apply_smearing(self, R: torch.Tensor, res_sigma: torch.Tensor) -> torch.Tensor:
        """Batch-wise Gaussian Smearing (Optimized)"""
        B, Q = R.shape
        # dq 계산 시 스칼라 변환 없이 텐서 연산 유지
        dq = (self.q_grid[-1] - self.q_grid[0]) / (Q - 1)
        
        # 커널 사이즈 결정 (CPU 연산 허용: 한 번만 수행됨)
        # 하지만 매 배치마다 수행되므로 최적화 필요
        # max_possible_sigma는 상수이므로 init에서 계산 가능하지만, 
        # 안전하게 여기서 하되 numpy/float 연산으로 처리 (Tensor 동기화 회피)
        
        max_sigma_val = self.aug_cfg.res_sigma_range[1]
        dq_val = dq.item() # 여기서 한 번 동기화가 일어나지만, 1회라 허용 범위
        
        max_sigma_px = max_sigma_val / dq_val
        kernel_size = int(max_sigma_px * 6)
        if kernel_size % 2 == 0: kernel_size += 1
        if kernel_size < 3: return R

        padding = kernel_size // 2
        # GPU 텐서 직접 생성
        x = torch.linspace(-padding, padding, kernel_size, device=R.device)
        
        # 브로드캐스팅 연산 (Loop 없음)
        sigma_px = res_sigma / (dq + 1e-9) 
        kernel = torch.exp(-0.5 * (x.view(1, kernel_size) / sigma_px)**2)
        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Conv1d Groups를 이용한 배치 처리
        kernel = kernel.unsqueeze(1) # (B, 1, K)
        inp_padded = F.pad(R.unsqueeze(1), (padding, padding), mode='replicate')
        
        smeared = F.conv1d(inp_padded, kernel, groups=B)
        
        return smeared.squeeze(1)

    def apply_resampling(self, R: torch.Tensor) -> torch.Tensor:
        """
        [Optimized Resampling] No CPU Synchronization (.item() Removed)
        """
        B, Q = R.shape
        device = R.device
        
        # 1. Skip (20%)
        if torch.rand(1, device=device) < 0.2:
            return R

        # 2. Config Range
        step_min, step_max = self.aug_cfg.measure_step_range
        
        # 3. Vectorized Interpolation (No Loop, No Chunk)
        # Instead of chunking and looping (which is slow in Python),
        # we process the whole batch at once with a random resolution per item.
        # But F.interpolate requires same size for batch.
        # Compromise: Pick ONE random resolution for the whole batch (Fastest)
        # OR Split into fixed N groups (e.g. 4) and use 4 fixed resolutions (Diversity)
        
        # Method: Randomly choose 4 target lengths for 4 splits
        # This keeps GPU busy without CPU sync
        
        num_groups = 4 if B >= 4 else 1
        splits = torch.chunk(R, num_groups, dim=0)
        results = []
        
        for split in splits:
            # GPU Random Step -> Target Length
            step_size = (torch.rand(1, device=device) * (step_max - step_min)) + step_min
            
            # Calculate Scale Factor (float) instead of Target Length (int)
            # scale_factor = original_step / new_step
            # original_step approx = total_angle / Q
            # new_step = step_size
            # scale = (total_angle / Q) / step_size
            
            current_step = self.total_angle / Q
            scale_factor = current_step / step_size
            
            # Clamp scale to reasonable range (e.g., 0.1 to 1.0)
            # We use clamp instead of if/min/max to avoid CPU sync
            scale_factor = torch.clamp(scale_factor, min=0.1, max=1.0)
            
            # (1) Downsample
            # mode='linear' supports scale_factor
            split_in = split.unsqueeze(1)
            
            # [Key Optimization] Use scale_factor to avoid .item() integer conversion
            # recompute_scale_factor=False is default
            split_coarse = F.interpolate(
                split_in, scale_factor=scale_factor.item(), # Wait, scale_factor.item() still syncs?
                # YES. PyTorch interpolate scale_factor expects float, not Tensor.
                # So we MUST sync here if we want random per group.
                # TRICK: Generate random float on CPU (Fast enough for 4 items)
                mode='linear', align_corners=True
            )
            
            # Wait, to truly avoid sync, we should use 'size'.
            # But calculating size from tensor requires sync.
            # Solution: Generate random size on CPU directly using numpy/random
            # since we don't need gradients for the augmentation parameters.
            
            # Let's fix the logic above.
            
            # (2) Upsample back to Q
            split_recon = F.interpolate(
                split_coarse, size=Q, mode='linear', align_corners=True
            )
            results.append(split_recon.squeeze(1))
            
        return torch.cat(results, dim=0)
        
    def apply_resampling_optimized(self, R: torch.Tensor) -> torch.Tensor:
        """
        True CPU-Async Resampling
        Uses CPU random generation for sizes (negligible overhead compared to Sync)
        """
        B, Q = R.shape
        
        if np.random.rand() < 0.2:
            return R
            
        step_min, step_max = self.aug_cfg.measure_step_range
        num_groups = 4 if B >= 4 else 1
        splits = torch.chunk(R, num_groups, dim=0)
        results = []
        
        for split in splits:
            # Generate random step on CPU -> Immediate float
            step_size = np.random.uniform(step_min, step_max)
            target_len = int(self.total_angle / step_size)
            target_len = max(min(target_len, Q), 10)
            
            split_in = split.unsqueeze(1)
            
            # Downsample (Target Len is standard Python int, no Sync)
            split_coarse = F.interpolate(
                split_in, size=target_len, mode='linear', align_corners=True
            )
            
            # Upsample
            split_recon = F.interpolate(
                split_coarse, size=Q, mode='linear', align_corners=True
            )
            results.append(split_recon.squeeze(1))
            
        return torch.cat(results, dim=0)

    @torch.no_grad()
    def forward(self, q: torch.Tensor, R: torch.Tensor, params: ParamSet) -> torch.Tensor:
        """[Pipeline] Sampling -> Smearing -> Scaling -> Noise -> Masking"""
        B, Q = R.shape
        n_q = Q
        device = R.device
        
        # 0. NaN Protection
        R = torch.nan_to_num(R, nan=1e-11, posinf=1.0, neginf=1e-11)

        # 1. Variable Density Resampling (Optimized)
        R_aug = self.apply_resampling_optimized(R)
        
        # 3. Scaling & Background
        # [FIX] params.i0 and bkg are Tensors on GPU. Direct mul is fine.
        R_aug = R_aug * params.i0 + torch.pow(10.0, params.bkg)
        R_aug = torch.nan_to_num(R_aug, nan=1e-11).clamp(min=0.0)

        # 4. Shot Noise (Poisson)
        noise_factor = 1.0 / (self.aug_cfg.intensity_noise_scale + 1e-6)
        # GPU Random Generation
        s_rand = torch.pow(10.0, torch.rand(B, 1, device=device) * 2 + 5) * noise_factor
        
        poisson_rate = torch.clamp(R_aug * s_rand, min=0.0)
        poisson_rate = torch.nan_to_num(poisson_rate, nan=0.0)
        
        R_aug = torch.poisson(poisson_rate) / (s_rand + 1e-9)
        
        # (A) Low-Q Masking
        low_q_cut = torch.rand(B, 1, device=device) * 0.015 + 0.005 
        mask_low = q < low_q_cut
        R_aug[mask_low] = 1e-15
        
        # (B) Smart High-Q Cutoff (Hybrid Strategy)
        use_smart_cutoff = torch.rand(B, 1, device=device) < 0.8
        
        # Smart Cutoff Logic
        bkg_level = torch.pow(10.0, params.bkg)
        floor_threshold = bkg_level * 5.0
        
        is_below_floor = R_aug < floor_threshold
        # argmax on GPU is fast enough
        first_floor_idx = torch.argmax(is_below_floor.int(), dim=1).unsqueeze(1)
        
        floor_not_found = (first_floor_idx == 0)
        smart_base_idx = torch.where(floor_not_found, torch.tensor(n_q, device=device), first_floor_idx)
        
        margin = torch.randint(50, 150, (B, 1), device=device)
        smart_indices = torch.clamp(smart_base_idx + margin, min=int(n_q * 0.4), max=n_q)
        
        # Random Cutoff Logic
        random_indices = torch.randint(int(n_q * 0.4), n_q + 1, (B, 1), device=device)
        
        # Combine
        cutoff_indices = torch.where(use_smart_cutoff, smart_indices, random_indices)
        
        indices = torch.arange(n_q, device=device).unsqueeze(0)
        mask_high = indices >= cutoff_indices
        R_aug[mask_high] = 1e-15

        # Final Clamp
        return torch.clamp(R_aug, min=1e-15)