import numpy as np
import torch
import torch.nn as nn

from xrefine.core.interface import Simulator
from xrefine.core.corrections import DifferentiableFootprint
from xrefine.core.structures import ParamSet
from xrefine.logger import logger

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

        raw_out = self.regressor(feat)
        return torch.sigmoid(raw_out)

class XRR_Fourier_PINN(nn.Module):
    def __init__(self, config, simulator: nn.Module | None = None, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self.q_grid = config.q_grid.to(device)

        # 3*N + 7: d_f, sig_f, rho_f (3N) + d_ox, sig_ox, rho_ox (3) + L, bkg, i0, beam_width (4) = 3N + 7
        self.encoder = XRRPhysicsModel(
            q_len=config.q_len, 
            n_layers=config.n_layers,
            mlp_hidden=config.hidden_dim,
            out_dim=3 * config.n_layers + 7
        ).to(device)

        self.simulator = simulator if simulator else Simulator(method='abeles', device=device)
        
        # [수정] Footprint 레이어는 이제 초기화 시 beam_width를 고정하지 않고 forward에서 받음
        self.footprint = DifferentiableFootprint(
            wavelength=config.instrument.wavelength
        ).to(device)

    def _unnormalize(self, out: torch.Tensor) -> ParamSet:
        """AI의 0~1 출력을 물리적 스케일로 변환"""
        cfg = self.config
        n = cfg.n_layers
        rescale = lambda val, r: val * (r[1] - r[0]) + r[0]

        # 1. Main Film
        d_f = rescale(out[:, 0:n], cfg.thick_range)
        sig_f = rescale(out[:, n:2*n], cfg.rough_range)
        rho_f = rescale(out[:, 2*n:3*n], cfg.sld_range)

        # 2. SiO2 Layer
        idx = 3 * n
        d_ox = rescale(out[:, idx:idx+1], cfg.sio2_thick_range)
        sig_ox = rescale(out[:, idx+1:idx+2], cfg.sio2_rough_range)
        rho_ox = rescale(out[:, idx+2:idx+3], cfg.sio2_sld_range)

        # 3. Instrumental (중요: bkg는 로그 스케일, i0는 선형 스케일)
        L = rescale(out[:, idx+3:idx+4], cfg.sample_len_range)
        log_bkg = rescale(out[:, idx+4:idx+5], cfg.augmentations.bg_range)
        
        i0_range = (1 - cfg.augmentations.intensity_noise_scale, 1 + cfg.augmentations.intensity_noise_scale)
        i0 = rescale(out[:, idx+5:idx+6], i0_range)
        
        beam_width = rescale(out[:, idx+6:idx+7], cfg.beam_width_range)

        return {
            "d_f": d_f, "sig_f": sig_f, "rho_f": rho_f,
            "d_ox": d_ox, "sig_ox": sig_ox, "rho_ox": rho_ox,
            "L": L, "i0": i0, "bkg": log_bkg, "beam_width": beam_width
        }

    def validate_physics(self, d_f, sig_f, d_ox, sig_ox):
        """물리적 제약 조건 페널티"""
        penalty_f = torch.mean(torch.relu(sig_f - 0.5 * d_f)) 
        penalty_ox = torch.mean(torch.relu(sig_ox - 0.5 * d_ox))
        return penalty_f + penalty_ox

    def forward(self, r_obs_log) -> tuple[ParamSet, torch.Tensor, torch.Tensor]:
        if torch.isnan(r_obs_log).any():
            r_obs_log = torch.nan_to_num(r_obs_log, nan=-10.0)
            
        out = self.encoder(r_obs_log)
        params = self._unnormalize(out)
        
        # --- 물리 시뮬레이션 영역 ---
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # 1. 텐서 조립 및 타입 통일
            thickness = torch.cat([params["d_f"], params["d_ox"]], dim=-1).float()
            # Roughness 구조: [Amb/Film1, ..., FilmN/SiO2, SiO2/Sub]
            roughness = torch.cat([params["sig_f"], params["sig_ox"], params["sig_ox"]], dim=-1).float()

            amb = torch.zeros(params["d_f"].size(0), 1, device=self.device)
            sub = torch.full((params["d_f"].size(0), 1), self.config.substrate_sld, device=self.device)
            sld_full = torch.cat([amb, params["rho_f"], params["rho_ox"], sub], dim=-1).float()

            q_batch = self.q_grid.unsqueeze(0).expand(params["d_f"].size(0), -1).float()
            dq = torch.tensor([[self.config.instrument.res]], device=self.device, dtype=torch.float32)

            # 2. 순수 시뮬레이션 (Linear Scale)
            r_sim = self.simulator(q_batch, thickness, roughness, sld_full, dq=dq)
            
            # 3. [수정] Footprint 보정: 이제 beam_width도 모델의 예측치를 사용함
            r_corr = self.footprint(
                q_batch, 
                r_sim, 
                params["L"].float(), 
                params["beam_width"].float()
            )

        # --- 결과 결합 (AI Scaling) ---
        # [수정] 최종 반사율 공식: R_final = I0 * R_sim + Background
        r_final = (params["i0"] * r_corr) + torch.pow(10.0, params["bkg"])
        r_final = torch.clamp(r_final, min=1e-11, max=2.0) # 수치 안정성

        phys_penalty = self.validate_physics(params["d_f"], params["sig_f"], params["d_ox"], params["sig_ox"])

        return params, r_final, phys_penalty