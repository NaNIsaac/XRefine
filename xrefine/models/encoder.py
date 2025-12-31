import torch
import torch.nn as nn

class XRRParameterEncoder(nn.Module):
    def __init__(self, q_len, out_dim=6): # [d, sig, rho] + [L, dq, I0]
        super().__init__()
        # 1D-CNN 구조 (기존 xrr_model.py 참고)
        self.conv = nn.Sequential(
            nn.Conv1d(2, 64, 7, padding=3), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(16)
        )
        self.regressor = nn.Linear(128 * 16, out_dim)

    def forward(self, x):
        feat = self.conv(x).view(x.size(0), -1)
        # [Anti-Gravity] 0~1 사이로 가두고 물리적 스케일링
        out = torch.sigmoid(self.regressor(feat))
        
        # 물리적 범위 강제 (Config 기반)
        thick = out[:, 0:1] * 1000.0 + 10.0
        rough = out[:, 1:2] * 20.0
        sld   = out[:, 2:3] * 80.0 + 5.0
        # 기기 파라미터 (학습 대상)
        s_len = out[:, 3:4] * 30.0 + 10.0
        dq    = out[:, 4:5] * 0.005 + 0.001
        i0    = out[:, 5:6] * 1.5 + 0.5
        
        return thick, rough, sld, s_len, dq, i0