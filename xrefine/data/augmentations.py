import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from xrefine.config import XRefineConfig
from xrefine.core.structures import ParamSet

class XRRAugmentor(nn.Module):
    """
    GPU 기반 고속 XRR 물리 증강 레이어 (최적화 버전).
    
    [New Features]
    - Low-Q Masking: 실제 실험에서 측정 불가능한 저각도 영역 제거
    - High-Q Random Cutoff: 데이터 부족 상황(Extrapolation) 학습
    """
    def __init__(self, config: XRefineConfig):
        super().__init__()
        self.cfg = config
        self.aug_cfg = config.augmentations
        self.register_buffer('sqrt_8ln2', torch.sqrt(torch.tensor(8.0 * np.log(2.0))))

    def apply_smearing(self, R: torch.Tensor, res_sigma: torch.Tensor) -> torch.Tensor:
        """배치 단위 가우시안 스미어링 (고정 커널 최적화 버전)"""
        B, Q = R.shape
        q_grid = self.cfg.q_grid.to(R.device)
        dq = (q_grid[-1] - q_grid[0]) / (Q - 1)
        
        max_possible_sigma = self.aug_cfg.res_sigma_range[1]
        max_sigma_px = max_possible_sigma / dq
        kernel_size = int(max_sigma_px * 6)
        if kernel_size % 2 == 0: kernel_size += 1
        if kernel_size < 3: return R

        padding = kernel_size // 2
        x = torch.linspace(-padding, padding, kernel_size, device=R.device)
        sigma_px = res_sigma / (dq + 1e-9) # (B, 1)
        
        kernel = torch.exp(-0.5 * (x.view(1, kernel_size) / sigma_px)**2)
        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-9)
        kernel = kernel.unsqueeze(1) 

        inp = R.unsqueeze(1) # (B, 1, Q)
        inp_padded = F.pad(inp, (padding, padding), mode='replicate')
        
        smeared = F.conv1d(inp_padded.view(1, B, -1), kernel, bias=None, groups=B)
        
        return smeared.view(B, Q)

    def apply_resampling(self, R: torch.Tensor) -> torch.Tensor:
        """
        [Improved Resampling] Grouped Downsampling
        배치를 여러 그룹으로 나누어 서로 다른 해상도를 적용하여 학습 다양성 확보.
        
        개선 사항:
        1. 배치를 4개 그룹으로 나눠 각기 다른 해상도 적용 (Batch Diversity)
        2. 20% 확률로 원본 유지 (Skip Chance)
        3. target_len이 Q를 초과하지 않도록 안전 처리
        """
        B, Q = R.shape
        device = R.device
        
        # 1. 원본 유지 확률 (20% Skip)
        # 고해상도 데이터 학습 능력 유지
        if torch.rand(1, device=device) < 0.2:
            return R

        # 2. Config 범위
        step_min, step_max = self.aug_cfg.measure_step_range
        total_angle = self.cfg.q_max_tth - self.cfg.q_min_tth
        
        # 3. 그룹 나누기 (배치가 작으면 통으로 처리)
        num_groups = 4 if B >= 4 else 1
        chunks = torch.chunk(R, num_groups, dim=0)
        results = []

        for chunk in chunks:
            # 그룹별 랜덤 스텝 사이즈
            step_size = (torch.rand(1, device=device) * (step_max - step_min)) + step_min
            
            # 목표 길이 계산 (Q 길이를 초과하지 않도록 min 처리)
            target_len = int((total_angle / step_size).item())
            target_len = max(min(target_len, Q), 10)  # 10 < len <= Q

            # (1) Downsample (정보 손실 유도)
            chunk_in = chunk.unsqueeze(1)
            chunk_coarse = F.interpolate(
                chunk_in, size=target_len, mode='linear', align_corners=True
            )

            # (2) Upsample (모델 입력 크기 복원)
            chunk_recon = F.interpolate(
                chunk_coarse, size=Q, mode='linear', align_corners=True
            )
            
            results.append(chunk_recon.squeeze(1))

        # 다시 하나의 배치로 합치기
        return torch.cat(results, dim=0)

    @torch.no_grad()
    def forward(self, q: torch.Tensor, R: torch.Tensor, params: ParamSet) -> torch.Tensor:
        """[Pipeline] Sampling -> Smearing -> Scaling -> Noise -> Masking"""
        B, Q = R.shape
        n_q = Q
        device = R.device
        
        # 0. NaN 방어
        R = torch.nan_to_num(R, nan=1e-11, posinf=1.0, neginf=1e-11)

        # 1. Variable Density Resampling (New!)
        # 실제 측정 포인트 수의 다양성을 모사 (Coarse Sampling -> Linear Interpolation)
        R_aug = self.apply_resampling(R)
        
        # 2. Q-Shift Augmentation
        # [REMOVED] Incorrect additive noise implementation. 
        # Physical q-shift is already handled in 'generate_synthetic_batch' via tth_offset.

        # 3. Scaling & Background
        R_aug = R_aug * params.i0 + torch.pow(10.0, params.bkg)
        R_aug = torch.nan_to_num(R_aug, nan=1e-11).clamp(min=0.0)

        # 4. Shot Noise (Poisson)
        noise_factor = 1.0 / (self.aug_cfg.intensity_noise_scale + 1e-6)
        s_rand = torch.pow(10.0, torch.rand(B, 1, device=device) * 2 + 5) * noise_factor
        
        poisson_rate = torch.clamp(R_aug * s_rand, min=0.0)
        poisson_rate = torch.nan_to_num(poisson_rate, nan=0.0)
        
        R_aug = torch.poisson(poisson_rate) / (s_rand + 1e-9)
        
        # (A) Low-Q Masking (Safe Zone)
        # q_c for Si is approx 0.032. We must preserve data around and after q_c.
        # Cutoff range: 0.005 ~ 0.020
        low_q_cut = torch.rand(B, 1, device=device) * 0.015 + 0.005 
        mask_low = q < low_q_cut
        R_aug[mask_low] = 1e-15
        
        # (B) Smart High-Q Cutoff (Hybrid Strategy)
        # Strategy 1 (80%): Smart Cutoff - 신호가 Noise Floor에 도달하면 약간의 마진 후 종료
        # Strategy 2 (20%): Random Cutoff - 다양성 확보를 위해 무작위 길이로 잘림
        
        use_smart_cutoff = torch.rand(B, 1, device=device) < 0.8
        
        # --- Smart Cutoff Logic ---
        bkg_level = torch.pow(10.0, params.bkg)  # (B, 1)
        floor_threshold = bkg_level * 5.0        # Floor보다 약간 위에서 감지 시작
        
        # Threshold 이하로 떨어지는 첫 지점 탐색
        is_below_floor = R_aug < floor_threshold # (B, Q)
        # argmax는 첫 번째 True의 인덱스를 반환 (없으면 0)
        first_floor_idx = torch.argmax(is_below_floor.int(), dim=1).unsqueeze(1)
        
        # Floor에 도달 못한 경우(0)는 n_q(전체)로 설정
        floor_not_found = (first_floor_idx == 0)
        smart_base_idx = torch.where(floor_not_found, torch.tensor(n_q, device=device), first_floor_idx)
        
        # Margin 추가 (Noise Floor를 50~150 포인트 정도 보여줌)
        margin = torch.randint(50, 150, (B, 1), device=device)
        smart_indices = torch.clamp(smart_base_idx + margin, min=int(n_q * 0.4), max=n_q)
        
        # --- Random Cutoff Logic ---
        random_indices = torch.randint(int(n_q * 0.4), n_q + 1, (B, 1), device=device)
        
        # --- Combine ---
        cutoff_indices = torch.where(use_smart_cutoff, smart_indices, random_indices)
        
        indices = torch.arange(n_q, device=device).unsqueeze(0)
        mask_high = indices >= cutoff_indices
        R_aug[mask_high] = 1e-15

        # 6. 최종 클램핑
        # 최소값을 1e-15로 고정하여 Log 변환 시 정확히 -15.0이 되도록 함
        return torch.clamp(R_aug, min=1e-15)