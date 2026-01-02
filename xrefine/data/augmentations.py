import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from xrefine.config import XRefineConfig

class XRRAugmentor(nn.Module):
    """
    GPU 기반 고속 XRR 물리 증강 레이어.
    
    개선 사항:
    1. Context Switching 제거: 모든 연산이 GPU에서 배차 단위(Vectorized)로 수행됨.
    2. 물리적 일관성: CONFIG의 wavelength 및 instrument 설정을 직접 참조.
    3. 메모리 효율: 중복된 NumPy 변환 없이 텐서 흐름 유지.
    """
    def __init__(self, config: XRefineConfig):
        super().__init__()
        self.cfg = config
        self.device = config.device
        
        # 가우시안 커널용 상수
        self.register_buffer('sqrt_2', torch.sqrt(torch.tensor(2.0)))
        self.register_buffer('sqrt_8ln2', torch.sqrt(torch.tensor(8.0 * np.log(2.0))))

    def apply_footprint(self, q: torch.Tensor, R: torch.Tensor, 
                        beam_width: torch.Tensor, sample_len: torch.Tensor) -> torch.Tensor:
        """
        Differentiable Gaussian Footprint Correction.
        q: (B, Q), beam_width: (B, 1), sample_len: (B, 1)
        """
        # q -> theta (rad) 변환
        wavelength = self.cfg.instrument.wavelength
        sin_alpha = torch.clamp(q * wavelength / (4 * np.pi), -1.0, 1.0)
        
        # FWHM to Sigma conversion
        sigma_b = beam_width / self.sqrt_8ln2
        s_half = sample_len / 2.0
        
        # Footprint factor calculation: erf( s*sin(alpha) / (sqrt2 * sigma_b) )
        arg = (s_half * sin_alpha) / (self.sqrt_2 * sigma_b + 1e-9)
        footprint_factor = torch.erf(arg)
        
        return R * footprint_factor

    def apply_smearing(self, R: torch.Tensor, res_sigma: torch.Tensor) -> torch.Tensor:
        """
        Batch-wise Gaussian Resolution Smearing 최적화 버전.
        R: (B, Q), res_sigma: (B, 1)
        """
        B, Q = R.shape
        dq_avg = (self.cfg.q_grid[-1] - self.cfg.q_grid[0]) / (Q - 1)
        
        # 커널 사이즈 결정
        max_sigma_px = (res_sigma.max() / dq_avg).item()
        kernel_size = int(max_sigma_px * 6)
        if kernel_size % 2 == 0: kernel_size += 1
        if kernel_size < 3: return R

        padding = kernel_size // 2

        # 1. 가우시안 커널 생성 (B, 1, K)
        x = torch.linspace(-padding, padding, kernel_size, device=R.device)
        sigma_px = res_sigma / dq_avg # (B, 1)
        
        # 브로드캐스팅을 이용한 배치별 커널 생성
        kernel = torch.exp(-0.5 * (x.view(1, kernel_size) / sigma_px)**2)
        kernel = kernel / kernel.sum(dim=-1, keepdim=True)
        kernel = kernel.unsqueeze(1) # (B, 1, K)

        # 2. [핵심 수정] 입력 텐서 재구성
        # (B, Q) -> (1, B, Q) 로 변경하여 배치를 채널로 취급
        inp = R.unsqueeze(0) 
        inp_padded = F.pad(inp, (padding, padding), mode='replicate')

        # 3. Conv1d 실행 (groups=B 설정을 통해 각 채널별로 대응하는 커널 적용)
        # input: (1, B, Q_padded), weight: (B, 1, K)
        smeared = F.conv1d(inp_padded, kernel, groups=B)
        
        # 4. 다시 (B, Q) 형태로 복원
        return smeared.squeeze(0)

    def apply_q_shift(self, q: torch.Tensor, R: torch.Tensor, q_shift: torch.Tensor) -> torch.Tensor:
        """
        Q-axis misalignment simulation (Linear Interpolation).
        """
        # torch.roll은 정수 단위이므로, 미세한 shift는 interp1d 로직 필요
        # 여기서는 단순화를 위해 q_grid를 직접 조정하는 방식 권장 (Generator 단계)
        # 또는 간단한 shift 연산
        return R # 실제 구현 시에는 Generator에서 q_grid + shift로 Simulator에 주입하는 것이 가장 깔끔함

    @torch.no_grad()
    def forward(self, q: torch.Tensor, R: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Main augmentation pipeline.
        q: (B, Q), R: (B, Q) - 선형 스케일 Reflectivity
        """
        B, Q = R.shape
        device = R.device
        
        # 1. Random Parameter Sampling (Batch-wise)
        # Intensity Scale (I0)
        i0_min = 1.0 - self.cfg.augmentations.intensity_noise_scale
        i0_max = 1.0 + self.cfg.augmentations.intensity_noise_scale
        i0 = torch.rand(B, 1, device=device) * (i0_max - i0_min) + i0_min
        
        # Background (Log scale)
        bkg_val = torch.rand(B, 1, device=device) * \
                  (self.cfg.augmentations.bg_range[1] - self.cfg.augmentations.bg_range[0]) + \
                  self.cfg.augmentations.bg_range[0]
        bkg = torch.pow(10.0, bkg_val)
        
        # Footprint params
        bw = torch.rand(B, 1, device=device) * \
             (self.cfg.beam_width_range[1] - self.cfg.beam_width_range[0]) + \
             self.cfg.beam_width_range[0]
        sl = torch.rand(B, 1, device=device) * \
             (self.cfg.sample_len_range[1] - self.cfg.sample_len_range[0]) + \
             self.cfg.sample_len_range[0]
             
        # Resolution Sigma
        res_sig = torch.rand(B, 1, device=device) * \
                  (self.cfg.augmentations.res_sigma_range[1] - self.cfg.augmentations.res_sigma_range[0]) + \
                  self.cfg.augmentations.res_sigma_range[0]

        # --- 순차적 증강 적용 ---
        R_aug = R.clone()
        
        # [Step 1] Footprint
        R_aug = self.apply_footprint(q, R_aug, bw, sl)
        
        # [Step 2] Smearing
        R_aug = self.apply_smearing(R_aug, res_sig)
        
        # [Step 3] Scale & Background
        R_aug = R_aug * i0 + bkg
        
        # [Step 4] Shot Noise (Poisson)
        # s_rand: 노출 시간/강도에 따른 통계적 노이즈 수준
        s_rand = torch.pow(10.0, torch.rand(B, 1, device=device) * 3 + 5) # 1e5 ~ 1e8
        R_aug = torch.poisson(torch.clamp(R_aug * s_rand, min=0.0)) / s_rand
        
        # 메타데이터 반환 (Loss 계산 및 모니터링용)
        meta = {
            "i0": i0,
            "bkg": bkg_val, # Log10 bkg
            "beam_width": bw,
            "sample_len": sl,
            "res_sigma": res_sig
        }
        
        return torch.clamp(R_aug, min=1e-10), meta