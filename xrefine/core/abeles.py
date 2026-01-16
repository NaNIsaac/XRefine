"""
Abeles Matrix Method Module (Eager Mode / No JIT)
=================================================
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

# [중요] @torch.jit.script 데코레이터가 없어야 합니다!
def abeles_core(q: Tensor, thickness: Tensor, roughness: Tensor, sld: Tensor) -> Tensor:
    # A100 Eager Mode 최적화 (Pure PyTorch)
    k_n = torch.sqrt((q / 2.0)**2 - 4.0 * math.pi * sld)
    k_curr, k_next = k_n[..., :-1], k_n[..., 1:] 

    beta = 1j * thickness * k_curr
    exp_beta = torch.exp(beta)
    exp_m_beta = torch.exp(-beta)

    sig2 = roughness ** 2
    denom = k_curr + k_next
    epsilon = 1e-9  
    
    r_n = ((k_curr - k_next) / (denom + epsilon)) * torch.exp(-2.0 * k_curr * k_next * sig2)

    m_00, m_01 = exp_beta, r_n * exp_beta
    m_10, m_11 = r_n * exp_m_beta, exp_m_beta

    m_accum_00, m_accum_01 = m_00[..., 0], m_01[..., 0]
    m_accum_10, m_accum_11 = m_10[..., 0], m_11[..., 0]

    num_layers = thickness.shape[2] 
    for i in range(1, num_layers):
        a, b, c, d = m_accum_00, m_accum_01, m_accum_10, m_accum_11
        e, f, g, h = m_00[..., i], m_01[..., i], m_10[..., i], m_11[..., i]
        
        m_accum_00 = a * e + b * g
        m_accum_01 = a * f + b * h
        m_accum_10 = c * e + d * g
        m_accum_11 = c * f + d * h

    quotient = m_accum_10 / (m_accum_00 + epsilon)
    return quotient.real**2 + quotient.imag**2

class AbelesMatrix(nn.Module):
    def __init__(self, device: torch.device | None = None):
        super().__init__()
        self.device = device

    def forward(self, q: Tensor, thickness: Tensor, roughness: Tensor, sld: Tensor, 
                sample_length: Optional[Tensor]=None, beam_width: Optional[Tensor]=None, wavelength: float=1.5406) -> Tensor:
        if self.device is None: self.device = q.device
        
        # 입력 데이터가 이미 GPU에 있다면 변환 최소화
        q_f, t_f, r_f = q.float(), thickness.float(), roughness.float()
        sld_c = sld.to(dtype=torch.complex64) if torch.is_complex(sld) else sld.float().to(dtype=torch.complex64)
        
        # Broadcasting & Complex Setup
        amb_sld = sld_c[:, 0:1]
        sld_comp = (sld_c - amb_sld) * 1e-6 + 1j * 1e-9
        
        q_in = q_f.unsqueeze(-1).to(dtype=torch.complex64)
        sld_in = sld_comp.unsqueeze(1)
        t_in = t_f.unsqueeze(1)
        r_in = r_f.unsqueeze(1)

        reflectivity = abeles_core(q_in, t_in, r_in, sld_in)
        
        if sample_length is not None and beam_width is not None:
            sample_length, beam_width = sample_length.float(), beam_width.float()
            sin_alpha = torch.clamp(q_f * wavelength / (4.0 * math.pi), -1.0, 1.0)
            sigma_b = beam_width / 2.3548
            arg = (sample_length * sin_alpha) / (2.8284 * sigma_b + 1e-9)
            reflectivity = reflectivity * torch.erf(arg)

        return reflectivity

def abeles(*args, **kwargs):
    model = AbelesMatrix(device=args[0].device if args else None)
    return model(*args, **kwargs)