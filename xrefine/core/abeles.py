"""
Abeles Matrix Method Module (Float32/Complex64 High-Speed JIT)
==============================================================
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

# [핵심] JIT 컴파일 함수 (Float32 버전)
@torch.jit.script
def abeles_jit_core(q: Tensor,          # (B, Q, 1) Complex64
                    thickness: Tensor,  # (B, 1, N) Float32
                    roughness: Tensor,  # (B, 1, N+1) Float32
                    sld: Tensor         # (B, 1, N+2) Complex64
                    ) -> Tensor:
    
    # -----------------------------------------------------------
    # 1. Wave Vector Calculation
    # -----------------------------------------------------------
    # Float32/Complex64 연산
    k_n = torch.sqrt((q / 2.0)**2 - 4.0 * math.pi * sld)
    
    k_curr = k_n[..., :-1] 
    k_next = k_n[..., 1:] 

    # -----------------------------------------------------------
    # 2. Phase Shift (Beta)
    # -----------------------------------------------------------
    beta = 1j * thickness * k_curr
    exp_beta = torch.exp(beta)
    exp_m_beta = torch.exp(-beta)

    # -----------------------------------------------------------
    # 3. Fresnel Coefficient
    # -----------------------------------------------------------
    sig2 = roughness ** 2
    denom = k_curr + k_next
    
    # Epsilon for safe division (Complex Float)
    # [Diff] 1e-16 -> 1e-7 (Float32 한계 고려)
    epsilon = torch.tensor(1e-9 + 0j, dtype=torch.complex64, device=q.device)
    
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
    # 6. Reflectivity
    # -----------------------------------------------------------
    numerator = m_accum_10
    denominator = m_accum_00 + epsilon
    
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
        # Python Side: Prepare Tensors (Float32 & Complex64)
        # -----------------------------------------------------------
        # [Diff] .float() 사용
        q_float = q.float()
        thick_float = thickness.float()
        rough_float = roughness.float()
        
        # SLD Complex Conversion
        if torch.is_complex(sld):
            sld_c = sld.to(dtype=torch.complex64)
        else:
            sld_c = sld.float().to(dtype=torch.complex64)
            
        # Relative SLD
        amb_sld = sld_c[:, 0:1]
        # [Diff] Imaginary Epsilon 1e-9 (Float32 Resolution)
        sld_complex = (sld_c - amb_sld) * 1e-6 + 1j * 1e-9
        
        # Broadcasting Alignment
        q_in = q_float.unsqueeze(-1).to(dtype=torch.complex64) # (B, Q, 1)
        sld_in = sld_complex.unsqueeze(1) # (B, 1, N)
        thick_in = thick_float.unsqueeze(1) # (B, 1, N)
        rough_in = rough_float.unsqueeze(1) # (B, 1, N)

        # -----------------------------------------------------------
        # Call JIT Core
        # -----------------------------------------------------------
        reflectivity = abeles_jit_core(q_in, thick_in, rough_in, sld_in)
        
        # -----------------------------------------------------------
        # Footprint Correction
        # -----------------------------------------------------------
        if sample_length is not None and beam_width is not None:
            sample_length = sample_length.float()
            beam_width = beam_width.float()
            
            sin_alpha = torch.clamp(q_float * wavelength / (4.0 * math.pi), -1.0, 1.0)
            sigma_b = beam_width / 2.3548
            arg = (sample_length * sin_alpha) / (2.8284 * sigma_b + 1e-9)
            reflectivity = reflectivity * torch.erf(arg)

        return reflectivity
    
def abeles(*args, **kwargs):
    model = AbelesMatrix(device=args[0].device if args else None)
    return model(*args, **kwargs)