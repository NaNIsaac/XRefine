import numpy as np
import torch
import torch.nn as nn


class DifferentiableFootprint(nn.Module):
    """PINN의 Physics-Loss 계산을 위한 미분 가능한 Footprint 레이어"""

    def __init__(self, wavelength: float = 1.5406):
        super().__init__()
        self.wavelength = wavelength
        self.register_buffer("pi", torch.tensor(np.pi))

    def forward(
        self,
        q: torch.Tensor,
        r_sim: torch.Tensor,
        sample_len: torch.Tensor,
        beam_width: torch.Tensor,
    ) -> torch.Tensor:
        # 1. q에서 sin(alpha) 추출
        sin_alpha = q * self.wavelength / (4.0 * self.pi)

        # 2. Gaussian Beam Sigma 계산 (FWHM -> Sigma)
        # FWHM = 2 * sqrt(2 * ln2) * sigma_b
        denom_sigma = 2.0 * torch.sqrt(2.0 * torch.log(torch.tensor(2.0, device=q.device)))
        sigma_b = beam_width / denom_sigma

        # 3. Footprint Factor 계산 (erf 기반)
        s_half = sample_len / 2.0

        # 실제 연산에 변수 사용 (물리적 의미: 시료 경계 s_half와 빔 투영 크기의 비율)
        arg = (s_half * sin_alpha) / (
            torch.sqrt(torch.tensor(2.0, device=q.device)) * sigma_b + 1e-9
        )
        footprint_factor = torch.erf(arg)

        return r_sim * footprint_factor
