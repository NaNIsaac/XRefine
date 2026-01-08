"""
Abeles Matrix Method Module (Fixed)
===================================
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

class AbelesMatrix(nn.Module):
    def __init__(self, device: torch.device | None = None):
        super().__init__()
        self.device = device

    def forward(self, 
                q: Tensor, 
                thickness: Tensor, 
                roughness: Tensor, 
                sld: Tensor,
                sample_length: Optional[Tensor] = None,
                beam_width: Optional[Tensor] = None,
                wavelength: float = 1.5406) -> Tensor:
        
        if self.device is None: self.device = q.device
        q = q.float().to(self.device)
        thickness = thickness.float().to(self.device)
        roughness = roughness.float().to(self.device)
        sld = sld.float().to(self.device)

        # num_layers = Ambient + Films... (입력된 thickness 컬럼 수)
        # 이제 Ambient가 포함되므로 3이 됨.
        num_layers = thickness.shape[1] 

        q_aug = q.unsqueeze(-1).to(dtype=torch.complex64)
        
        # SLD Complex 변환
        sld_complex = (sld.unsqueeze(1) - sld[:, 0:1].unsqueeze(1)) * 1e-6 + 1e-30j

        # Wave vector (k_z)
        k_n = torch.sqrt((q_aug / 2)**2 - 4 * math.pi * sld_complex)
        k_curr, k_next = k_n[..., :-1], k_n[..., 1:]

        # Phase Shift (Beta)
        # 이제 thickness와 k_curr의 마지막 차원이 둘 다 num_layers로 일치함!
        beta = 1j * thickness.unsqueeze(1) * k_curr
        exp_beta = torch.exp(beta)
        exp_m_beta = torch.exp(-beta)

        # Fresnel
        sig2 = (roughness.unsqueeze(1) ** 2)
        denom = k_curr + k_next + 1e-30
        r_n = ((k_curr - k_next) / denom) * torch.exp(-2 * k_curr * k_next * sig2)

        # Matrix Elements
        m_00 = exp_beta
        m_01 = r_n * exp_beta
        m_10 = r_n * exp_m_beta
        m_11 = exp_m_beta

        # Matrix Multiplication
        m_accum_00, m_accum_01 = m_00[..., 0], m_01[..., 0]
        m_accum_10, m_accum_11 = m_10[..., 0], m_11[..., 0]

        # [FIX] Loop Range: 1부터 num_layers-1(마지막 인덱스)까지
        # num_layers=3일 때 indices는 0, 1, 2
        # 0은 초기화에 썼으므로 루프는 1, 2를 돌아야 함 -> range(1, 3)
        for i in range(1, num_layers):
            a, b, c, d = m_accum_00, m_accum_01, m_accum_10, m_accum_11
            e, f, g, h = m_00[..., i], m_01[..., i], m_10[..., i], m_11[..., i]
            
            m_accum_00 = a * e + b * g
            m_accum_01 = a * f + b * h
            m_accum_10 = c * e + d * g
            m_accum_11 = c * f + d * h

        # Reflectivity
        reflectivity = (m_accum_10 / (m_accum_00 + 1e-30)).abs() ** 2
        
        # Footprint Correction
        if sample_length is not None and beam_width is not None:
            sin_alpha = torch.clamp(q * wavelength / (4.0 * math.pi), -1.0, 1.0)
            sigma_b = beam_width / 2.3548
            arg = (sample_length * sin_alpha) / (2.8284 * sigma_b + 1e-9)
            reflectivity = reflectivity * torch.erf(arg)

        return torch.clamp(reflectivity, min=1e-12, max=1.0)

def abeles(*args, **kwargs):
    model = AbelesMatrix(device=args[0].device if args else None)
    return model(*args, **kwargs)