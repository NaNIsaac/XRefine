"""
Abeles Matrix Method Module (Rigorous Complex & Double Precision)
=================================================================
"""
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
        
        # [Physics] 엄밀한 계산을 위해 Double Precision (float64/complex128) 강제
        q = q.double().to(self.device)
        thickness = thickness.double().to(self.device)
        roughness = roughness.double().to(self.device)
        
        # SLD 처리 (Real or Complex)
        # 입력 sld가 복소수(금속 등)라면 그대로 유지, 실수라면 float64로 변환
        if torch.is_complex(sld):
            sld = sld.to(dtype=torch.complex128, device=self.device)
        else:
            sld = sld.double().to(self.device)

        # num_layers = Ambient + Films...
        num_layers = thickness.shape[1] 

        # Q 벡터 복소수 확장 (Batch, Q, 1)
        q_aug = q.unsqueeze(-1).to(dtype=torch.complex128)
        
        # -----------------------------------------------------------
        # 1. SLD Complex Processing (Absorption Handling)
        # -----------------------------------------------------------
        # 금속의 경우 SLD 자체가 복소수 (Re: Scattering, Im: Absorption)
        # 입력이 실수인 경우 수치 안정성을 위해 미세한 허수(EPS) 추가
        
        EPS = 1e-16
        
        # Ambient(첫번째 층) 기준 상대 굴절률 계산
        # sld 단위 보정 (10^-6 A^-2)
        sld_scaled = sld * 1e-6
        
        if not torch.is_complex(sld):
            # 비흡수성 물질(유전체) 가정 시 안정성 EPS 추가
            sld_complex = (sld_scaled.unsqueeze(1) - sld_scaled[:, 0:1].unsqueeze(1)) + 1j * EPS
        else:
            # 흡수성 물질(금속) - 이미 복소수이므로 그대로 연산
            # Ambient SLD를 빼주는 것은 스넬의 법칙 기준점 설정
            sld_complex = (sld_scaled.unsqueeze(1) - sld_scaled[:, 0:1].unsqueeze(1))

        # -----------------------------------------------------------
        # 2. Optical Matrix Calculation
        # -----------------------------------------------------------
        
        # Wave vector z-component (k_z)
        # 전반사 영역에서 sqrt 내부가 음수가 되며 자연스럽게 허수 k_z 생성 (Evanescent wave)
        k_n = torch.sqrt((q_aug / 2)**2 - 4 * math.pi * sld_complex)
        
        # Layer shift
        k_curr = k_n[..., :-1] # Layer j
        k_next = k_n[..., 1:]  # Layer j+1

        # Phase Shift (Beta)
        # 두께가 있는 층에서의 위상 변화
        # thickness shape: (B, Layers) -> broadcast with k_curr
        beta = 1j * thickness.unsqueeze(1) * k_curr
        
        # Overflow 방지를 위한 exp 처리 (Double Precision이라 상대적으로 안전하지만 클램핑)
        # beta의 실수부(흡수)가 너무 크면 exp가 발산할 수 있음
        exp_beta = torch.exp(beta)
        exp_m_beta = torch.exp(-beta)

        # Fresnel Reflection Coefficient (with Névot-Croce Roughness)
        # Roughness는 계면(Interface)의 속성
        sig2 = (roughness.unsqueeze(1) ** 2)
        
        # 분모 0 방지
        denom = k_curr + k_next
        denom = torch.where(denom.abs() < EPS, torch.tensor(EPS, dtype=denom.dtype, device=self.device), denom)
        
        # Roughness Damping Factor
        # k가 복소수일 때 -2*k*k*sig2 부분이 양의 실수가 되어 발산할 수 있으므로 주의 필요하나,
        # 물리적으로 올바른 영역 내에서는 감쇠로 작용함.
        roughness_factor = torch.exp(-2 * k_curr * k_next * sig2)
        
        r_n = ((k_curr - k_next) / denom) * roughness_factor

        # Matrix Elements construction
        # M_j = [ [exp(i*beta), r*exp(i*beta)], [r*exp(-i*beta), exp(-i*beta)] ]
        m_00 = exp_beta
        m_01 = r_n * exp_beta
        m_10 = r_n * exp_m_beta
        m_11 = exp_m_beta

        # -----------------------------------------------------------
        # 3. Matrix Multiplication Loop
        # -----------------------------------------------------------
        # 초기값 (Ambient Interface)
        m_accum_00, m_accum_01 = m_00[..., 0], m_01[..., 0]
        m_accum_10, m_accum_11 = m_10[..., 0], m_11[..., 0]

        # Loop over layers
        for i in range(1, num_layers):
            a, b, c, d = m_accum_00, m_accum_01, m_accum_10, m_accum_11
            e, f, g, h = m_00[..., i], m_01[..., i], m_10[..., i], m_11[..., i]
            
            # Complex Matrix Multiplication
            m_accum_00 = a * e + b * g
            m_accum_01 = a * f + b * h
            m_accum_10 = c * e + d * g
            m_accum_11 = c * f + d * h

        # -----------------------------------------------------------
        # 4. Reflectivity Calculation
        # -----------------------------------------------------------
        # R = |M_10 / M_00|^2
        numerator = m_accum_10
        denominator = m_accum_00 + EPS # Div by zero protection
        
        reflectivity = (numerator / denominator).abs() ** 2
        
        # -----------------------------------------------------------
        # 5. Footprint Correction (Geometric)
        # -----------------------------------------------------------
        if sample_length is not None and beam_width is not None:
            # Footprint 계산도 Double로 수행
            sample_length = sample_length.double()
            beam_width = beam_width.double()
            
            sin_alpha = torch.clamp(q * wavelength / (4.0 * math.pi), -1.0, 1.0)
            sigma_b = beam_width / 2.3548
            
            # 빔이 시료를 벗어나는 비율 계산
            arg = (sample_length * sin_alpha) / (2.8284 * sigma_b + 1e-9)
            reflectivity = reflectivity * torch.erf(arg)

        # Output Casting: 신경망과의 호환성을 위해 마지막에 float32로 변환
        return reflectivity.float()

def abeles(*args, **kwargs):
    model = AbelesMatrix(device=args[0].device if args else None)
    return model(*args, **kwargs)