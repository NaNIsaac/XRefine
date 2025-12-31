import torch
import torch.nn as nn
from xrefine.core.interface import Simulator
from xrefine.core.corrections import DifferentiableFootprint
# CONFIG import removed

class XRRParameterEncoder(nn.Module):
    def __init__(self, n_layers=1):
        super().__init__()
        # 1D-CNN: Extract fine features (Fourier features)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        # Output: [thick, rough, sld] + [sample_len, I0_scale]
        # Based on legacy Logic: 3*n_layers + 2
        self.regressor = nn.Linear(128 * 16, 3 * n_layers + 2)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        feat = self.encoder(x).view(x.size(0), -1)
        out = torch.sigmoid(self.regressor(feat))
        
        # Return Raw Normalized Output
        # De-normalization happens in the Wrapper/PINN
        return out

class XRR_PINN(nn.Module):
    def __init__(self, config, device='cpu'):
        # config is expected to be XRefineConfig object
        super().__init__()
        self.config = config
        self.q_grid = config.q_grid.to(device)
        
        self.encoder = XRRParameterEncoder(n_layers=config.n_layers)
        self.simulator = Simulator(method='abeles', device=device)
        self.footprint = DifferentiableFootprint(
            beam_width=config.instrument.beam_width, 
            wavelength=config.instrument.wavelength
        )

    def forward(self, r_obs):
        # 1. AI Prediction (Normalized -> Physical)
        out = self.encoder(r_obs)
        
        # Physical Range Constraints from Config
        thick_range = self.config.thick_range
        rough_range = self.config.rough_range
        sld_range = self.config.sld_range
        
        # Legacy Splitting Logic (for n_layers=1)
        # thick(1), rough(1), sld(1)
        thick = out[:, 0:1] * (thick_range[1] - thick_range[0]) + thick_range[0]
        rough = out[:, 1:2] * (rough_range[1] - rough_range[0]) + rough_range[0]
        sld   = out[:, 2:3] * (sld_range[1] - sld_range[0]) + sld_range[0]

        # Instrument Params
        s_len = out[:, 3:4] * 30.0 + 10.0
        i0_scale = out[:, 4:5] * 1.5 + 0.5
        
        # 2. Physics Engine Setup
        sld_full = torch.cat([torch.zeros_like(sld), sld, torch.ones_like(sld)*20.1], dim=-1)
        rough_full = torch.cat([rough, rough], dim=-1)
        
        q_batch = self.q_grid.unsqueeze(0).repeat(r_obs.size(0), 1)
        
        # 3. Differentiable Forward Loop
        # Apply Instrumental Resolution (Smearing)
        dq_val = self.config.instrument.res
        dq = torch.full((r_obs.size(0), 1), dq_val, device=r_obs.device)
        
        r_sim = self.simulator(q_batch, thick, rough_full, sld_full, dq=dq)
        r_sim = self.footprint(q_batch, r_sim, s_len)
        r_final = r_sim * i0_scale # Note: i0 vs i0_scale consistent naming
        
        return (thick, rough, sld), r_final