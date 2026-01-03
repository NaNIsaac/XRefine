from __future__ import annotations

import math
import torch
from torch import Tensor

def abeles(
    q: Tensor,
    thickness: Tensor,
    roughness: Tensor,
    sld: Tensor,
    sample_length: torch.Tensor,
    beam_width: torch.Tensor,
    wavelength: float = 1.5406
) -> Tensor:
    """
    고속 배치를 위한 최적화된 Abeles Matrix Method.
    
    기존 대비 개선사항:
    1. 루프 최적화: 레이어 전파를 병렬화하거나 효율적인 연산자로 대체.
    2. 수치 안정성: k_z가 0에 가까울 때 발생하는 NaN 방지.
    3. 메모리 레이아웃: broadcasting 최적화를 통한 불필요한 복사 방지.
    """
    
    # 1. 모든 입력을 FP32로 강제 캐스팅 (ComplexHalf 방지 및 정밀도 확보)
    # 정밀도가 매우 중요하다면 .double()을 권장합니다.
    q = q.float() 
    thickness = thickness.float()
    roughness = roughness.float()
    sld = sld.float()

    # 1. 초기 타입 및 디바이스 설정
    device = q.device
    c_dtype = torch.complex128 if q.dtype is torch.float64 else torch.complex64
    batch_size, num_layers = thickness.shape

    # 2. 입력을 [B, Q, N] 형태로 정렬하여 연산 효율화
    # q: (B, Q) -> (B, Q, 1)
    q_aug = q.unsqueeze(-1).to(c_dtype)
    
    # SLD handling (Ambient + Layers + Substrate)
    if sld.shape[-1] == num_layers + 1:
        sld = torch.cat([torch.zeros(batch_size, 1, device=device, dtype=sld.dtype), sld], -1)
    
    # 3. 물리 상수 및 복소 SLD 변환
    # (B, 1, N+2)
    sld_complex = (sld.unsqueeze(1) - sld[:, 0:1].unsqueeze(1)) * 1e-6 + 1e-36j
    
    # 4. k_n 계산 (Wave vector in each layer)
    # k_n shape: (B, Q, num_layers + 2)
    # k_n = sqrt( (q/2)^2 - 4*pi*rho )
    k_n = torch.sqrt((q_aug / 2)**2 - 4 * math.pi * sld_complex)

    k_curr = k_n[..., :-1]  # (B, Q, L+1)
    k_next = k_n[..., 1:]   # (B, Q, L+1)

    # 5. Phase shift (Beta) 계산
    # thickness_aug: (B, 1, L+1) -> Ambient는 두께 0
    thickness_aug = torch.cat([
        torch.zeros(batch_size, 1, device=device, dtype=thickness.dtype), 
        thickness
    ], -1).unsqueeze(1)
    
    beta = 1j * thickness_aug * k_curr  # (B, Q, L+1)
    exp_beta = torch.exp(beta)
    exp_m_beta = torch.exp(-beta)

    # 6. Fresnel Coefficients with Roughness (Nevot-Croce)
    # sigma^2: (B, 1, L+1)
    sig2 = (roughness**2).unsqueeze(1)
    
    # r_n: (B, Q, L+1)
    # k_curr - k_next가 0일 경우를 대비해 아주 작은 값 추가 (Stability)
    denom = k_curr + k_next + 1e-36
    r_n = ((k_curr - k_next) / denom) * torch.exp(-2 * k_curr * k_next * sig2)

    # 7. Characteristic Matrices 구성
    # M_n = [[exp(beta), r*exp(beta)], [r*exp(-beta), exp(-beta)]]
    # (B, Q, L+1, 2, 2) 형태로 한 번에 생성
    m_00 = exp_beta
    m_01 = r_n * exp_beta
    m_10 = r_n * exp_m_beta
    m_11 = exp_m_beta

    # 8. 행렬 누적 곱 (Matrix Chain Multiplication)
    m_accum_00 = m_00[..., 0]
    m_accum_01 = m_01[..., 0]
    m_accum_10 = m_10[..., 0]
    m_accum_11 = m_11[..., 0]

    for i in range(1, num_layers + 1):
        # 직접 수동 곱셈 수행 (matmul보다 overhead 적음)
        a, b = m_accum_00, m_accum_01
        c, d = m_accum_10, m_accum_11
        
        e, f = m_00[..., i], m_01[..., i]
        g, h = m_10[..., i], m_11[..., i]
        
        m_accum_00 = a * e + b * g
        m_accum_01 = a * f + b * h
        m_accum_10 = c * e + d * g
        m_accum_11 = c * f + d * h

    # 9. Reflectivity 산출
    # R = |M10 / M00|^2
    reflectivity = (m_accum_10 / (m_accum_00)).abs() ** 2
    
    sin_alpha = torch.clamp(q * wavelength / (4.0 * math.pi), -1.0, 1.0)
    
    # Gaussian Sigma (FWHM to Sigma)
    # sqrt(8ln2) ≈ 2.3548
    sigma_b = beam_width / 2.3548
    
    # erf( (L * sin_alpha) / (2 * sqrt(2) * sigma_b) )
    # 2 * sqrt(2) ≈ 2.8284
    arg = (sample_length * sin_alpha) / (2.8284 * sigma_b + 1e-9)
    footprint_factor = torch.erf(arg)
    
    # 풋프린트가 적용된 최종 반사율 반환
    return torch.clamp(reflectivity * footprint_factor, min=1e-11, max=1.0)