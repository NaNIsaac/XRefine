import torch
import torch.nn as nn
import numpy as np

class DifferentiableFootprint(nn.Module):
    """Footprint 및 기기 효과를 보정하는 미분 가능한 레이어"""
    def __init__(self, beam_width=0.1, wavelength=1.5406):
        super().__init__()
        self.beam_w = beam_width
        self.wavelen = wavelength

    def forward(self, q, r_sim, sample_len):
        # 1. q -> theta (degrees) 변환
        theta = 2 * torch.arcsin(q * self.wavelen / (4 * np.pi)) * (180 / np.pi)
        alpha = theta / 2.0
        
        # 2. Footprint Factor (Gaussian Beam 모사)
        # s: 시료 절반 길이, sigma_b: 빔의 가우시안 표준편차
        s = sample_len / 2.0
        sigma_b = self.beam_w / (2 * np.sqrt(2 * torch.log(torch.tensor(2.0))))
        
        sin_alpha = torch.sin(torch.deg2rad(alpha))
        # erf를 이용한 미분 가능한 풋프린트 수식
        footprint_factor = torch.erf(s * sin_alpha / (np.sqrt(2) * sigma_b))
        
        return r_sim * footprint_factor
