import numpy as np
import torch
import torch.nn as nn
# CONFIG import removed

class FourierFeatureMapping(nn.Module):
    """
    Random Fourier Features mapping:
    Maps input coordinates (q) to high-dimensional frequency space.
    """
    def __init__(self, input_dim=1, mapping_size=64, scale=10.0):
        super().__init__()
        self.mapping_size = mapping_size
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        # x: [batch, 1, len]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # x: [batch, 1, len]
        x_trans = x.transpose(1, 2) # [batch, len, 1]
        x_proj = x_trans @ self.B # [batch, len, map]
        x_proj = x_proj.transpose(1, 2) * 2 * np.pi # [batch, map, len]
        
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)

class XRRPhysicsModel(nn.Module):
    def __init__(self, 
                 q_len: int = 2000, 
                 n_channels: int = 64, 
                 depth: int = 5, 
                 mlp_hidden: int = 256,
                 dropout: float = 0.1, 
                 use_fourier: bool = True, 
                 fourier_scale: float = 10.0,
                 n_layers: int = 1):
        super().__init__()
        self.use_fourier = use_fourier
        inplace: bool = True
        
        # Total parameters = N(Thick) + N+1(Rough) + N(SLD) + 1(s_len) + 1(i0) = 3N + 3
        self.final_out_dim = 3 * n_layers + 3
        
        if use_fourier:
            self.fourier = FourierFeatureMapping(input_dim=1, mapping_size=32, scale=fourier_scale)
            # Input channels: Data(1) + Fourier(32*2) = 65
            enc_in_channels = 1 + (32 * 2) 
        else:
            enc_in_channels = 1

        self.register_buffer('q_grid', torch.linspace(0, 1, q_len).view(1, 1, -1))

        # Input Normalization to handle LogR range (-12 to 0)
        self.input_norm = nn.BatchNorm1d(1)

        layers = []
        curr_dim = enc_in_channels
        for i in range(depth):
            out_dim = n_channels * (2 ** min(i, 3))
            layers.append(nn.Sequential(
                nn.Conv1d(curr_dim, out_dim, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2, inplace=inplace),
                nn.Dropout1d(dropout),
                nn.MaxPool1d(kernel_size=2)
            ))
            curr_dim = out_dim

        self.encoder = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.regressor = nn.Sequential(
            nn.Linear(curr_dim, mlp_hidden),
            nn.LeakyReLU(0.2, inplace=inplace),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.LeakyReLU(0.2, inplace=inplace),
            nn.Linear(mlp_hidden // 2, self.final_out_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        # x: [batch, 1, len]
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        bs = x.shape[0]
        
        # Apply Input Normalization
        x = self.input_norm(x)
        
        if self.use_fourier:
            q_batch = self.q_grid.expand(bs, -1, -1)
            q_fourier = self.fourier(q_batch)
            x = torch.cat([x, q_fourier], dim=1)
            
        feat = self.encoder(x)
        feat = self.global_pool(feat).squeeze(-1)
        out = torch.sigmoid(self.regressor(feat))
        
        return out

from xrefine.core.interface import Simulator
from xrefine.core.corrections import DifferentiableFootprint

class XRR_Fourier_PINN(nn.Module):
    def __init__(self, config, device='cpu'):
        # Accept config object
        super().__init__()
        self.config = config
        self.q_grid = config.q_grid.to(device)
        
        self.encoder = XRRPhysicsModel(
            q_len=config.q_len,
            n_layers=config.n_layers,
            n_channels=64, # could be in config
            use_fourier=True, # could be in config
            fourier_scale=config.fourier_scale,
            mlp_hidden=config.hidden_dim
        )
        
        self.simulator = Simulator(method='abeles', device=device)
        self.footprint = DifferentiableFootprint(
            beam_width=config.instrument.beam_width,
            wavelength=config.instrument.wavelength
        )

    def forward(self, r_obs):
        # 1. AI Prediction (Normalized)
        out = self.encoder(r_obs)
        
        # 2. Denormalization & Constraints
        # Explicit N_layers logic from Config
        n = self.config.n_layers
        thick_range = self.config.thick_range
        rough_range = self.config.rough_range
        sld_range = self.config.sld_range
        
        # Slicing Logic
        # Total = 3N + 3
        # thick(n), rough(n+1), sld(n), s_len(1), i0(1)
        
        idx_t = n
        idx_r = idx_t + (n + 1)
        idx_s = idx_r + n
        idx_inst = idx_s + 2
        
        # Improved Variable Naming
        thickness = out[:, :idx_t] * (thick_range[1] - thick_range[0]) + thick_range[0]
        roughness = out[:, idx_t:idx_r] * (rough_range[1] - rough_range[0]) + rough_range[0]
        sld = out[:, idx_r:idx_s] * (sld_range[1] - sld_range[0]) + sld_range[0]
        
        s_len = out[:, -2:-1] * 30.0 + 10.0
        i0_scale = out[:, -1:] * 1.5 + 0.5
        
        # 3. Physics Engine Setup
        # Construct full SLD/Roughness (if needed)
        # Assuming AI predicts Layer SLD (size N).
        # We need Size N+2 for Simulator [Amb, Layers, Sub]
        
        ambient = torch.zeros(thickness.shape[0], 1, device=thickness.device)
        substrate_sld = torch.full((thickness.shape[0], 1), 20.1, device=thickness.device) 
        
        sld_full = torch.cat([ambient, sld, substrate_sld], dim=-1)
        
        # Roughness:
        # AI returns size N+1.
        # Simulator expects N+1. (Amb/L1 ... Ln/Sub)
        rough_full = roughness
        
        q_batch = self.q_grid.unsqueeze(0).repeat(r_obs.size(0), 1)
        
        # 4. Differentiable Forward Loop
        # Apply Instrumental Resolution (Smearing)
        dq_val = self.config.instrument.res
        dq = torch.full((r_obs.size(0), 1), dq_val, device=r_obs.device)
        
        r_sim = self.simulator(q_batch, thickness, rough_full, sld_full, dq=dq)
        r_sim = self.footprint(q_batch, r_sim, s_len)
        r_final = r_sim * i0_scale
        
        return (thickness, rough_full, sld_full), r_final
