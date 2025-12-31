import numpy as np
import torch
import torch.nn as nn
from xrefine.core.interface import Simulator
from xrefine.core.corrections import DifferentiableFootprint

class FourierFeatureMapping(nn.Module):
    """
    Random Fourier Features mapping:
    q 좌표를 고주파 공간으로 매핑하여 미세 프링지(Fringe) 인식을 강화함.
    """
    def __init__(self, input_dim=1, mapping_size=32, scale=15.0):
        super().__init__()
        self.mapping_size = mapping_size
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x_proj = (x.transpose(1, 2) @ self.B).transpose(1, 2)
        x_proj = x_proj * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)

class XRRPhysicsModel(nn.Module):
    """CNN Encoder: 0~1 사이의 정규화된 파라미터 예측"""
    def __init__(self, q_len=2000, n_channels=64, depth=5, mlp_hidden=512, 
                 use_fourier=True, fourier_scale=15.0, n_layers=1):
        super().__init__()
        self.use_fourier = use_fourier
        # 출력: d_f(N), sig_f(N), rho_f(N), d_ox(1), sig_ox(1), L(1), log_bkg(1) = 3N + 4
        self.final_out_dim = 3 * n_layers + 4 

        if use_fourier:
            self.fourier = FourierFeatureMapping(input_dim=1, mapping_size=32, scale=fourier_scale)
            enc_in_channels = 1 + (32 * 2) 
        else:
            enc_in_channels = 1

        self.register_buffer('q_grid_norm', torch.linspace(0, 1, q_len).view(1, 1, -1))

        layers = []
        curr_dim = enc_in_channels
        for i in range(depth):
            out_dim = n_channels * (2 ** min(i, 2))
            layers.append(nn.Sequential(
                nn.Conv1d(curr_dim, out_dim, kernel_size=11, padding=5, bias=False),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool1d(2)
            ))
            curr_dim = out_dim

        self.encoder = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.regressor = nn.Sequential(
            nn.Linear(curr_dim, mlp_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(mlp_hidden // 2, self.final_out_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        nn.init.xavier_uniform_(self.regressor[-1].weight, gain=0.5)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        # 입력 스케일링: LogR (-10~0) -> (0~1)
        x = torch.clamp((x + 10.0) / 10.0, 0.0, 1.0)
        
        if self.use_fourier:
            q_feat = self.fourier(self.q_grid_norm.expand(x.shape[0], -1, -1))
            x = torch.cat([x, q_feat], dim=1)
            
        feat = self.encoder(x)
        feat = self.global_pool(feat).view(feat.size(0), -1)
        return torch.sigmoid(self.regressor(feat))

class XRR_Fourier_PINN(nn.Module):
    """통합 PINN: Config 기반 동기화 완료"""
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self.q_grid = config.q_grid.to(device)
        
        self.encoder = XRRPhysicsModel(
            q_len=config.q_len, n_layers=config.n_layers,
            fourier_scale=config.fourier_scale, mlp_hidden=config.hidden_dim
        ).to(device)
        
        self.simulator = Simulator(method='abeles', device=device)
        self.footprint = DifferentiableFootprint(
            beam_width=config.instrument.beam_width, 
            wavelength=config.instrument.wavelength
        ).to(device)

    def forward(self, r_obs_log):
        # 1. AI로부터 정규화된 값 획득
        out = self.encoder(r_obs_log)
        
        # 2. 통합 파라미터 복원 (AttributeError 수정 완료)
        n = self.config.n_layers
        cfg = self.config
        
        # [Main Film]
        d_f   = out[:, 0:n] * (cfg.thick_range[1] - cfg.thick_range[0]) + cfg.thick_range[0]
        sig_f = out[:, n:2*n] * (cfg.rough_range[1] - cfg.rough_range[0]) + cfg.rough_range[0]
        rho_f = out[:, 2*n:3*n] * (cfg.sld_range[1] - cfg.sld_range[0]) + cfg.sld_range[0]
        
        # [SiO2] - config.sio2_thick_range 등 명칭 직접 참조
        d_ox   = out[:, 3:4] * (cfg.sio2_thick_range[1] - cfg.sio2_thick_range[0]) + cfg.sio2_thick_range[0]
        sig_ox = out[:, 4:5] * (cfg.sio2_rough_range[1] - cfg.sio2_rough_range[0]) + cfg.sio2_rough_range[0]

        # [Instrumental] - nan 방지를 위해 clamp 추가
        L_norm = torch.clamp(out[:, 5:6], 0.001, 0.999) 
        L = L_norm * (cfg.sample_len_max - cfg.sample_len_min) + cfg.sample_len_min

        log_bkg_norm = torch.clamp(out[:, 6:7], 0.001, 0.999)
        log_bkg = log_bkg_norm * (cfg.augmentations.bg_range[1] - cfg.augmentations.bg_range[0]) + cfg.augmentations.bg_range[0]
        
        # 3. 물리 구조 조립
        thickness = torch.cat([d_f, d_ox], dim=-1)
        # Roughness: [Amb/Film, Film/SiO2, SiO2/Sub] (3개 인터페이스)
        # 상부 거칠기는 sig_f, 하부(SiO2 연관) 거칠기는 sig_ox 사용
        roughness = torch.cat([sig_f, sig_ox, sig_ox], dim=-1) 
        
        amb = torch.zeros(d_f.size(0), 1, device=self.device)
        sio2 = torch.ones_like(d_f) * cfg.sio2_sld
        sub = torch.ones_like(d_f) * cfg.substrate_sld
        sld_full = torch.cat([amb, rho_f, sio2, sub], dim=-1)
        
        # 4. 물리 시뮬레이션
        q_batch = self.q_grid.unsqueeze(0).expand(d_f.size(0), -1)
        dq = torch.full((d_f.size(0), 1), cfg.instrument.res, device=self.device)
        
        r_sim = self.simulator(q_batch, thickness, roughness, sld_full, dq=dq)
        r_corr = self.footprint(q_batch, r_sim, L)
        
        # 5. 최종 결합 및 Clamp
        r_final = torch.clamp(r_corr + torch.pow(10.0, log_bkg), min=1e-10)
        
        return (d_f, sig_f, rho_f, d_ox, sig_ox, L, log_bkg), r_final