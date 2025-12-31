import torch
import torch.nn as nn
from xrefine.core.interface import Simulator
from xrefine.core.corrections import DifferentiableFootprint

class XRRParameterEncoder(nn.Module):
    def __init__(self, n_layers=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(16)
        )
        # Output: [thick, rough, sld] x N + [d_ox, sig_ox, L, log_bkg]
        self.regressor = nn.Linear(128 * 16, 3 * n_layers + 4)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        # Standard Scaling
        x = torch.clamp((x + 10.0) / 10.0, 0.0, 1.0)
        feat = self.encoder(x).view(x.size(0), -1)
        return torch.sigmoid(self.regressor(feat))

class XRR_PINN(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self.q_grid = config.q_grid.to(device)
        self.encoder = XRRParameterEncoder(n_layers=config.n_layers).to(device)
        self.simulator = Simulator(method='abeles', device=device)
        self.footprint = DifferentiableFootprint(
            beam_width=config.instrument.beam_width, wavelength=config.instrument.wavelength
        ).to(device)

    def forward(self, r_obs_log):
        out = self.encoder(r_obs_log)
        n = self.config.n_layers
        t_rng, r_rng, s_rng = self.config.thick_range, self.config.rough_range, self.config.sld_range

        # 파라미터 분리 및 복원
        d_f   = out[:, 0:n] * (t_rng[1] - t_rng[0]) + t_rng[0]
        sig_f = out[:, n:2*n] * (r_rng[1] - r_rng[0]) + r_rng[0]
        rho_f = out[:, 2*n:3*n] * (s_rng[1] - s_rng[0]) + s_rng[0]
        
        d_ox   = out[:, -4:-3] * 45.0 + 5.0
        sig_ox = out[:, -3:-2] * 14.5 + 0.5
        L      = out[:, -2:-1] * 30.0 + 10.0
        log_bkg = out[:, -1:] * 3.0 - 8.0

        # 물리 구조 조립
        thickness = torch.cat([d_f, d_ox], dim=-1)
        roughness = torch.cat([sig_f, sig_ox, sig_ox], dim=-1)
        sld_full = torch.cat([torch.zeros_like(d_f), rho_f, torch.ones_like(d_f)*18.8, torch.ones_like(d_f)*20.1], dim=-1)

        q_batch = self.q_grid.unsqueeze(0).expand(d_f.size(0), -1)
        r_sim = self.simulator(q_batch, thickness, roughness, sld_full, dq=None)
        r_corr = self.footprint(q_batch, r_sim, L)
        r_final = torch.clamp(r_corr + torch.pow(10.0, log_bkg), min=1e-10)

        return (d_f, sig_f, rho_f), r_final