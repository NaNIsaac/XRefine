"""
Abeles Matrix Method Module (Eager Mode - No JIT)
=================================================
A100 최적화: JIT 오버헤드 제거, Pure PyTorch 구현
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

# [핵심] JIT(@torch.jit.script) 제거 -> Eager Mode 실행
# Complex64 호환성 문제 해결 및 디버깅 용이성 확보
def abeles_core(q: Tensor,          # (B, Q, 1)
                thickness: Tensor,  # (B, 1, N)
                roughness: Tensor,  # (B, 1, N+1)
                sld: Tensor         # (B, 1, N+2)
                ) -> Tensor:
    
    # -----------------------------------------------------------
    # 1. Wave Vector Calculation
    # -----------------------------------------------------------
    # k_z^2 = (q/2)^2 - 4*pi*sld
    # A100에서는 Eager Mode도 충분히 빠름 (커널 퓨전 자동 적용)
    k_n = torch.sqrt((q / 2.0)**2 - 4.0 * math.pi * sld)
    
    k_curr = k_n[..., :-1] 
    k_next = k_n[..., 1:] 

    # -----------------------------------------------------------
    # 2. Phase Shift (Beta)
    # -----------------------------------------------------------
    beta = 1j * thickness * k_curr
    # exp 연산은 GPU 병렬처리가 매우 잘됨
    exp_beta = torch.exp(beta)
    exp_m_beta = torch.exp(-beta)

    # -----------------------------------------------------------
    # 3. Fresnel Coefficient
    # -----------------------------------------------------------
    sig2 = roughness ** 2
    denom = k_curr + k_next
    
    # Epsilon (Float32/Complex64)
    epsilon = 1e-9  # 스칼라로 더해도 브로드캐스팅됨
    
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
    # 5. Loop (Manual Unrolling is not needed in PyTorch 2.x)
    # -----------------------------------------------------------
    # 초기값 (Ambient Interface) -> Index 0
    m_accum_00 = m_00[..., 0]
    m_accum_01 = m_01[..., 0]
    m_accum_10 = m_10[..., 0]
    m_accum_11 = m_11[..., 0]

    num_layers = thickness.shape[2] 
    
    # Python Loop지만 연산 자체가 텐서 통연산이라 빠름
    for i in range(1, num_layers):
        a, b, c, d = m_accum_00, m_accum_01, m_accum_10, m_accum_11
        e, f, g, h = m_00[..., i], m_01[..., i], m_10[..., i], m_11[..., i]
        
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
    # 복소수 절대값 제곱 수동 계산 (안전성)
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
        # Prepare Tensors (Float32 & Complex64)
        # -----------------------------------------------------------
        # 입력 데이터가 이미 GPU에 있다면 .to() 호출 최소화
        q_float = q.float()
        thick_float = thickness.float()
        rough_float = roughness.float()
        
        if torch.is_complex(sld):
            sld_c = sld.to(dtype=torch.complex64)
        else:
            sld_c = sld.float().to(dtype=torch.complex64)
            
        amb_sld = sld_c[:, 0:1]
        sld_complex = (sld_c - amb_sld) * 1e-6 + 1j * 1e-9
        
        # Broadcasting Alignment
        # 생성과 동시에 Device 지정하여 Copy 오버헤드 방지
        q_in = q_float.unsqueeze(-1).to(dtype=torch.complex64)
        sld_in = sld_complex.unsqueeze(1)
        thick_in = thick_float.unsqueeze(1)
        rough_in = rough_float.unsqueeze(1)

        # -----------------------------------------------------------
        # Call Pure Core
        # -----------------------------------------------------------
        reflectivity = abeles_core(q_in, thick_in, rough_in, sld_in)
        
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