import torch
import torch.nn as nn
from xrefine.core.interface import Simulator
from xrefine.core.corrections import DifferentiableFootprint
from .config import CONFIG

class XRRParameterEncoder(nn.Module):
    def __init__(self, n_layers=1):
        super().__init__()
        # 1D-CNN: 그래프의 미세 진동(Fourier features) 추출
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        # 출력: [thick, rough, sld] + [sample_len, I0_scale]
        self.regressor = nn.Linear(128 * 16, 3 * n_layers + 2)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        feat = self.encoder(x).view(x.size(0), -1)
        out = torch.sigmoid(self.regressor(feat))

        # 물리 범위 강제 구속
        thick_range = CONFIG['thick_range']
        rough_range = CONFIG['rough_range']
        sld_range = CONFIG['sld_range']

        thick = out[:, 0:1] * (thick_range[1] - thick_range[0]) + thick_range[0]
        rough = out[:, 1:2] * (rough_range[1] - rough_range[0]) + rough_range[0]
        sld   = out[:, 2:3] * (sld_range[1] - sld_range[0]) + sld_range[0]

        # 기기 오차 예측 (L: 10~40mm, I0_scale: 0.5~2.0)
        s_len = out[:, 3:4] * 30.0 + 10.0
        i0_scale = out[:, 4:5] * 1.5 + 0.5
        
        return thick, rough, sld, s_len, i0_scale

class XRR_PINN(nn.Module):
    def __init__(self, n_layers=1, device='cpu'):
        super().__init__()
        self.q_grid = CONFIG['q_grid'].to(device)
        self.encoder = XRRParameterEncoder(n_layers=n_layers)
        self.simulator = Simulator(method='abeles', device=device)
        self.footprint = DifferentiableFootprint()

    def forward(self, r_obs):
        # 1. AI 예측
        thick, rough, sld, s_len, i0 = self.encoder(r_obs)
        
        # 2. 물리 엔진 구성
        sld_full = torch.cat([torch.zeros_like(sld), sld, torch.ones_like(sld)*20.1], dim=-1)
        rough_full = torch.cat([rough, rough], dim=-1)
        q_batch = self.q_grid.unsqueeze(0).repeat(r_obs.size(0), 1)
        
        # 3. Differentiable Forward Loop (Physics + Corrections)
        r_sim = self.simulator(q_batch, thick, rough_full, sld_full)
        r_sim = self.footprint(q_batch, r_sim, s_len)
        r_final = r_sim * i0
        
        return (thick, rough, sld), r_final