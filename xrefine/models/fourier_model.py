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
        use_fourier=True, fourier_scale=15.0, n_layers=1, out_dim=None):
        super().__init__()
        self.use_fourier = use_fourier
        

        if out_dim is not None:
            self.final_out_dim = out_dim
        else:
            # d_f(N), sig_f(N), rho_f(N), d_ox(1), sig_ox(1), L(1), log_bkg(1) = 3N + 4
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
            out_dim_conv = n_channels * (2 ** min(i, 2)) # 변수명 중복 방지
            layers.append(nn.Sequential(
                nn.Conv1d(curr_dim, out_dim_conv, kernel_size=11, padding=5, bias=False),
                nn.BatchNorm1d(out_dim_conv),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool1d(2)
            ))
            curr_dim = out_dim_conv

        self.encoder = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.regressor = nn.Sequential(
            nn.Linear(curr_dim, mlp_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 여기서 self.final_out_dim이 정확히 적용됩니다.
            nn.Linear(mlp_hidden // 2, self.final_out_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        nn.init.xavier_uniform_(self.regressor[-1].weight, gain=0.5)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = torch.clamp((x + 10.0) / 10.0, 0.0, 1.0)

        if self.use_fourier:
            q_feat = self.fourier(self.q_grid_norm.expand(x.shape[0], -1, -1))
            x = torch.cat([x, q_feat], dim=1)

        feat = self.encoder(x)
        feat = self.global_pool(feat).view(feat.size(0), -1)
        return torch.sigmoid(self.regressor(feat))

class XRR_Fourier_PINN(nn.Module):
    def __init__(self, config, simulator: nn.Module = None, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self.q_grid = config.q_grid.to(device)

        # AI 모델 Dimension: 3*N + 5
        self.encoder = XRRPhysicsModel(
            q_len=config.q_len, 
            n_layers=config.n_layers,
            mlp_hidden=config.hidden_dim,
            out_dim=3 * config.n_layers + 5
        ).to(device)

        self.simulator = simulator if simulator else Simulator(method='abeles', device=device)
        self.footprint = DifferentiableFootprint(
            beam_width=config.instrument.beam_width, 
            wavelength=config.instrument.wavelength
        ).to(device)

    def _unnormalize(self, out):
        """AI의 0~1 출력을 물리적 스케일로 변환"""
        cfg = self.config
        n = cfg.n_layers
        
        out = 0.001 + 0.998 * out # 0과 1 사이의 극단값 방지
        # 헬퍼 함수: 선형 매핑
        def rescale(val, range_tuple):
            return val * (range_tuple[1] - range_tuple[0]) + range_tuple[0]

        # 1. Main Film
        d_f   = rescale(out[:, 0:n], cfg.thick_range)
        sig_f = rescale(out[:, n:2*n], cfg.rough_range)
        rho_f = rescale(out[:, 2*n:3*n], cfg.sld_range)
        
        # 2. SiO2 Layer
        idx = 3 * n
        d_ox   = rescale(out[:, idx:idx+1], cfg.sio2_thick_range)
        sig_ox = rescale(out[:, idx+1:idx+2], cfg.sio2_rough_range)
        rho_ox = rescale(out[:, idx+2:idx+3], cfg.sio2_sld_range)
        
        # 3. Instrumental
        L = rescale(out[:, idx+3:idx+4], cfg.sample_len_range)
        log_bkg = rescale(out[:, idx+4:idx+5], cfg.augmentations.bg_range)
        
        return d_f, sig_f, rho_f, d_ox, sig_ox, rho_ox, L, log_bkg

    def validate_physics(self, d_f, sig_f, d_ox, sig_ox):
        """
        1. 거칠기(sigma)는 층 두께(d)의 절반을 넘기 어렵다 (Gaussian Interface 한계).
        2. 박막 두께는 산화막보다 얇아질 수 없는 등의 상식적인 구속 조건 추가 가능.
        """
        # Sigma vs Thickness Constraint (ReLU를 사용해 위반 시에만 페널티 부여)
        penalty_f = torch.mean(torch.relu(sig_f - 0.5 * d_f)) 
        penalty_ox = torch.mean(torch.relu(sig_ox - 0.5 * d_ox))
        
        # Total Penalty
        return penalty_f + penalty_ox

    def forward(self, r_obs_log):

        if torch.isnan(r_obs_log).any():
            r_obs_log = torch.nan_to_num(r_obs_log, nan=-10.0)
        out = self.encoder(r_obs_log)
        d_f, sig_f, rho_f, d_ox, sig_ox, rho_ox, L, log_bkg = self._unnormalize(out)
        
        # 2. 물리 시뮬레이션 영역 (Autocast 비활성화)
        # Physics engine requires at least float32 (complex64) for stability
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # 데이터를 float32로 강제 변환 (안전장치)
            thickness = torch.cat([d_f, d_ox], dim=-1).float()
            roughness = torch.cat([sig_f, sig_ox, sig_ox], dim=-1).float()

            amb = torch.zeros(d_f.size(0), 1, device=self.device)
            sub = torch.full((d_f.size(0), 1), self.config.substrate_sld, device=self.device)
            sld_full = torch.cat([amb, rho_f, rho_ox, sub], dim=-1).float()
            
            q_batch = self.q_grid.unsqueeze(0).expand(d_f.size(0), -1).float()
            dq = torch.tensor([[self.config.instrument.res]], device=self.device, dtype=torch.float32)

            # 시뮬레이터 호출
            r_sim = self.simulator(q_batch, thickness, roughness, sld_full, dq=dq)
            r_sim = torch.nan_to_num(r_sim, nan=1e-10, posinf=1.0, neginf=1e-10)
            r_sim = torch.clamp(r_sim, min=1e-10, max=1.0)
            r_corr = self.footprint(q_batch, r_sim, L.float())

        # 3. 결과 결합 (다시 신경망 영역으로 복귀)
        r_final = torch.clamp(r_corr + torch.pow(10.0, log_bkg), min=1e-10)
        
        phys_penalty = self.validate_physics(d_f, sig_f, d_ox, sig_ox)
        
        params = (d_f, sig_f, rho_f, d_ox, sig_ox, rho_ox, L, log_bkg)
        return params, r_final, phys_penalty
    
    