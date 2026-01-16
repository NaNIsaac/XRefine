"""
Abeles Matrix Method Module (JIT Optimized)
===========================================
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

# [핵심] JIT 컴파일을 위한 함수형 구현
@torch.jit.script
def abeles_jit_core(q: Tensor, 
                    thickness: Tensor, 
                    roughness: Tensor, 
                    sld_real: Tensor,
                    sld_imag: Tensor,
                    wavelength: float) -> Tensor:
    
    # Double Precision 강제
    q = q.double()
    thickness = thickness.double()
    roughness = roughness.double()
    sld_real = sld_real.double()
    sld_imag = sld_imag.double()

    # SLD 복소수 구성 (Absorption)
    # sld 입력은 (B, N+2) 형태라고 가정
    # Ambient(첫번째 층) 기준 상대 굴절률 계산
    
    # (B, 1) Ambient SLD 추출
    amb_real = sld_real[:, 0:1]
    amb_imag = sld_imag[:, 0:1]
    
    # 10^-6 단위 보정 및 상대값 계산
    # sld_complex = (sld - sld_amb) * 1e-6
    delta_real = (sld_real - amb_real) * 1e-6
    delta_imag = (sld_imag - amb_imag) * 1e-6
    
    sld_complex = torch.complex(delta_real, delta_imag)
    
    # Q 벡터 확장 (B, Q, 1)
    q_aug = q.unsqueeze(-1).to(dtype=torch.complex128)
    
    # Wave vector k_z
    # k_n^2 = (q/2)^2 - 4*pi*sld
    k_n = torch.sqrt((q_aug / 2.0)**2 - 4.0 * math.pi * sld_complex)
    
    # Layer shift
    k_curr = k_n[..., :-1] # Layer j
    k_next = k_n[..., 1:]  # Layer j+1

    # Phase Shift (Beta)
    # thickness shape: (B, Layers) -> broadcast with k_curr
    # beta = k * d * 1j
    beta = 1j * thickness.unsqueeze(1) * k_curr
    
    exp_beta = torch.exp(beta)
    exp_m_beta = torch.exp(-beta)

    # Fresnel + Roughness
    sig2 = roughness.unsqueeze(1) ** 2
    
    # Denom safe div
    denom = k_curr + k_next
    # JIT에서는 where 조건이 스칼라여야 하거나 텐서여야 함.
    # 복소수 절대값 비교
    mask = denom.abs() < 1e-16
    # 1e-16+0j로 대체
    denom_safe = torch.where(mask, torch.tensor(1e-16, dtype=torch.complex128, device=q.device), denom)
    
    roughness_factor = torch.exp(-2.0 * k_curr * k_next * sig2)
    r_n = ((k_curr - k_next) / denom_safe) * roughness_factor

    # Matrix Elements
    m_00 = exp_beta
    m_01 = r_n * exp_beta
    m_10 = r_n * exp_m_beta
    m_11 = exp_m_beta

    # Matrix Multiplication Loop
    # 초기값 (Ambient Interface) -> Index 0
    m_accum_00 = m_00[..., 0]
    m_accum_01 = m_01[..., 0]
    m_accum_10 = m_10[..., 0]
    m_accum_11 = m_11[..., 0]

    num_layers = thickness.shape[1]
    
    # JIT가 가장 빛을 발하는 구간 (Python Loop 제거)
    for i in range(1, num_layers):
        a = m_accum_00
        b = m_accum_01
        c = m_accum_10
        d = m_accum_11
        
        e = m_00[..., i]
        f = m_01[..., i]
        g = m_10[..., i]
        h = m_11[..., i]
        
        m_accum_00 = a * e + b * g
        m_accum_01 = a * f + b * h
        m_accum_10 = c * e + d * g
        m_accum_11 = c * f + d * h

    # Reflectivity
    numerator = m_accum_10
    denominator = m_accum_00
    
    # Safe div for final R
    mask_final = denominator.abs() < 1e-16
    denom_final = torch.where(mask_final, torch.tensor(1e-16, dtype=torch.complex128, device=q.device), denominator)
    
    reflectivity = (numerator / denom_final).abs() ** 2
    return reflectivity

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
        
        # SLD 분리 (Real/Imag) -> JIT 함수로 전달
        # sld가 이미 complex라면 real/imag 분리
        if torch.is_complex(sld):
            sld_real = sld.real
            sld_imag = sld.imag
        else:
            sld_real = sld
            sld_imag = torch.zeros_like(sld)

        # 1. 핵심 물리 연산 (JIT 가속)
        # 입력 텐서들을 JIT 함수로 넘김 (JIT 내부에서 double 변환 수행함)
        reflectivity = abeles_jit_core(q, thickness, roughness, sld_real, sld_imag, wavelength)
        
        # 2. Footprint Correction (JIT 밖에서 수행 - Optional 처리 편의상)
        if sample_length is not None and beam_width is not None:
            sample_length = sample_length.double()
            beam_width = beam_width.double()
            q_dbl = q.double()
            
            sin_alpha = torch.clamp(q_dbl * wavelength / (4.0 * math.pi), -1.0, 1.0)
            sigma_b = beam_width / 2.3548
            arg = (sample_length * sin_alpha) / (2.8284 * sigma_b + 1e-9)
            reflectivity = reflectivity * torch.erf(arg)

        return reflectivity.float()

def abeles(*args, **kwargs):
    model = AbelesMatrix(device=args[0].device if args else None)
    return model(*args, **kwargs)