"""
Abeles Matrix Method Module (JIT Stable / Backward Fix)
=======================================================
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

# [핵심] JIT 컴파일 함수
@torch.jit.script
def abeles_jit_core(q: Tensor,          # (B, Q, 1) Complex128
                    thickness: Tensor,  # (B, 1, N) Float64
                    roughness: Tensor,  # (B, 1, N+1) Float64
                    sld: Tensor         # (B, 1, N+2) Complex128
                    ) -> Tensor:
    
    # -----------------------------------------------------------
    # 1. Wave Vector Calculation
    # -----------------------------------------------------------
    # k_z^2 = (q/2)^2 - 4*pi*sld
    k_n = torch.sqrt((q / 2.0)**2 - 4.0 * math.pi * sld)
    
    k_curr = k_n[..., :-1] # Layer j
    k_next = k_n[..., 1:]  # Layer j+1

    # -----------------------------------------------------------
    # 2. Phase Shift (Beta)
    # -----------------------------------------------------------
    # beta = k * d * 1j
    beta = 1j * thickness * k_curr
    
    exp_beta = torch.exp(beta)
    exp_m_beta = torch.exp(-beta)

    # -----------------------------------------------------------
    # 3. Fresnel Coefficient
    # -----------------------------------------------------------
    sig2 = roughness ** 2
    
    # Denom safe div (JIT Safe Logic)
    denom = k_curr + k_next
    
    # Epsilon for safe division (Complex Double)
    epsilon = torch.tensor(1e-16 + 0j, dtype=torch.complex128, device=q.device)
    
    # Soft Protection
    denom_safe = denom + epsilon

    roughness_factor = torch.exp(-2.0 * k_curr * k_next * sig2)
    r_n = ((k_curr - k_next) / denom_safe) * roughness_factor

    # -----------------------------------------------------------
    # 4. Matrix Elements
    # -----------------------------------------------------------
    m_00 = exp_beta
    m_01 = r_n * exp_beta
    m_10 = r_n * exp_m_beta
    m_11 = exp_m_beta

    # -----------------------------------------------------------
    # 5. Loop (JIT Optimized)
    # -----------------------------------------------------------
    m_accum_00 = m_00[..., 0]
    m_accum_01 = m_01[..., 0]
    m_accum_10 = m_10[..., 0]
    m_accum_11 = m_11[..., 0]

    num_layers = thickness.shape[2] 
    
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

    # -----------------------------------------------------------
    # 6. Reflectivity (Backward Fix)
    # -----------------------------------------------------------
    numerator = m_accum_10
    denominator = m_accum_00 + epsilon
    
    # [FIX] JIT Backward Error 해결
    # 기존: reflectivity = (numerator / denominator).abs() ** 2
    # 문제: .abs()의 역전파가 복소수에서 .sign()을 호출하며 에러 발생
    # 해결: 복소수 나눗셈 후, 실수부 제곱 + 허수부 제곱으로 직접 계산
    
    quotient = numerator / denominator
    reflectivity = quotient.real**2 + quotient.imag**2
    
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
        
        # -----------------------------------------------------------
        # Python Side: Prepare Tensors (Float64 & Complex128)
        # -----------------------------------------------------------
        q_dbl = q.double()
        thick_dbl = thickness.double()
        rough_dbl = roughness.double()
        
        # SLD Complex Conversion
        if torch.is_complex(sld):
            sld_c = sld.to(dtype=torch.complex128)
        else:
            sld_c = sld.double().to(dtype=torch.complex128)
            
        # Relative SLD & Epsilon
        amb_sld = sld_c[:, 0:1]
        sld_complex = (sld_c - amb_sld) * 1e-6 + 1j * 1e-16
        
        # Broadcasting Alignment
        q_in = q_dbl.unsqueeze(-1).to(dtype=torch.complex128) # (B, Q, 1)
        sld_in = sld_complex.unsqueeze(1) # (B, 1, N)
        thick_in = thick_dbl.unsqueeze(1) # (B, 1, N)
        rough_in = rough_dbl.unsqueeze(1) # (B, 1, N)

        # -----------------------------------------------------------
        # Call JIT Core
        # -----------------------------------------------------------
        reflectivity = abeles_jit_core(q_in, thick_in, rough_in, sld_in)
        
        # -----------------------------------------------------------
        # Footprint Correction
        # -----------------------------------------------------------
        if sample_length is not None and beam_width is not None:
            sample_length = sample_length.double()
            beam_width = beam_width.double()
            
            sin_alpha = torch.clamp(q_dbl * wavelength / (4.0 * math.pi), -1.0, 1.0)
            sigma_b = beam_width / 2.3548
            arg = (sample_length * sin_alpha) / (2.8284 * sigma_b + 1e-9)
            reflectivity = reflectivity * torch.erf(arg)

        return reflectivity.float()

def abeles(*args, **kwargs):
    model = AbelesMatrix(device=args[0].device if args else None)
    return model(*args, **kwargs)