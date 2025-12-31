import torch
import torch.nn as nn

class XRRParameterEncoder(nn.Module):
    """
    1D-CNN 기반 XRR 파라미터 추출기.
    입력 그래프로부터 물리적 특징(주파수, 감쇠)을 추출하여 
    제한된 물리 범위 내의 파라미터로 매핑합니다.
    """
    def __init__(self, q_len=2000, out_dim=7):
        super().__init__()
        # 특징 추출부: 커널 사이즈를 크게 가져가서 전역적인 진동 패턴을 파악함
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(16)
        )
        
        # 회귀부
        self.regressor = nn.Sequential(
            nn.Linear(128 * 16, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        # x shape: [batch, q_len]
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # 전처리: LogR (-10 ~ 0) 범위를 0 ~ 1로 선형 압축하여 CNN 전달
        x = torch.clamp((x + 10.0) / 10.0, 0.0, 1.0)
        
        feat = self.conv(x).view(x.size(0), -1)
        # Sigmoid를 사용하여 모든 출력을 0~1 사이로 강제함 (Anti-Gravity 기초)
        out = torch.sigmoid(self.regressor(feat))

        # 물리적 스케일링 (PINN Wrapper에서 수행할 수도 있으나 여기서 기본 매핑 제공)
        d_f = out[:, 0:1] * 990.0 + 10.0   # 10 ~ 1000 A
        sig_f = out[:, 1:2] * 25.0           # 0 ~ 25 A
        rho_f = out[:, 2:3] * 95.0 + 5.0     # 5 ~ 100 SLD

        d_ox = out[:, 3:4] * 45.0 + 5.0    # 5 ~ 50 A (SiO2)
        sig_ox = out[:, 4:5] * 14.5 + 0.5    # 0.5 ~ 15 A

        L = out[:, 5:6] * 10.0 + 15.0
        log_bkg = out[:, 6:7] * 2.0 - 8.0    # -8.0 ~ -5.0 (Ibkg)

        return d_f, sig_f, rho_f, d_ox, sig_ox, L, log_bkg