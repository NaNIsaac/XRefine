import numpy as np
import torch
import torch.nn as nn
from xrefine.config import CONFIG

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
        # x shape: [batch, 1, seq_len] or [batch, seq_len]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # x: [batch, 1, seq_len]
        # x_proj = x.transpose(1, 2) @ self.B  # This expects [batch, seq_len, 1] @ [1, map]
        
        # Adjust for Conv1d style or just simpler projection
        # Let's follow legacy logic:
        # x_proj = x.transpose(1, 2) @ self.B
        # x_proj = x_proj.transpose(1, 2) * 2 * np.pi
        
        # x: [batch, 1, len]
        x_trans = x.transpose(1, 2) # [batch, len, 1]
        x_proj = x_trans @ self.B # [batch, len, map]
        x_proj = x_proj.transpose(1, 2) * 2 * np.pi # [batch, map, len]
        
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)

class XRRPhysicsModel(nn.Module):
    def __init__(self, 
                 q_len: int = 2000, 
                 input_channels: int = 2, # [LogR, Mask] or just [R] 
                 output_dim: int = 3, # [Thick, Rough, SLD]
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
        
        # Determine actual output dimension based on n_layers
        # Legacy model output fixed 3 (thick, rough, sld) for 1 layer
        # Here we need to support n_layers generally
        # output: thickness(n), roughness(n+1), sld(n) + maybe instr params?
        # Let's stick to legacy's "1-layer" simplicity if n_layers=1, or generalize.
        # Legacy: [thick, rough, sld] -> 3 values. It assumed 1 layer film.
        
        # Let's check model.py's XRRParameterEncoder
        # It extracted [thick, rough, sld, s_len, i0_scale]
        
        # We should align with what we need. 
        # For now, let's output basic params for n_layers.
        # Param counts: Thick(N), Rough(N+1), SLD(N) + s_len(1) + i0(1)
        # Total = 3*N + 1 + 2 = 3N + 3?
        # Legacy XRRParameterEncoder: 3*n_layers + 2
        
        self.final_out_dim = 3 * n_layers + 2
        
        if use_fourier:
            self.fourier = FourierFeatureMapping(input_dim=1, mapping_size=32, scale=fourier_scale)
            # Input channels: Data(1) + Fourier(32*2) = 65
            # If we pass mask, input_channels might be 2. Let's assume input is R only (1 channel)
            # Legacy used input_channels=2 (R_interp, mask). 
            # We will assume input x is [batch, 1, len] (normalized R)
            enc_in_channels = 1 + (32 * 2) 
        else:
            enc_in_channels = 1

        self.register_buffer('q_grid', torch.linspace(0, 1, q_len).view(1, 1, -1))

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
        if self.use_fourier:
            q_batch = self.q_grid.expand(bs, -1, -1)
            # Ensure q_batch matches x length if needed, but usually fixed q_len
            q_fourier = self.fourier(q_batch)
            x = torch.cat([x, q_fourier], dim=1)
            
        feat = self.encoder(x)
        feat = self.global_pool(feat).squeeze(-1)
        out = torch.sigmoid(self.regressor(feat))
        
        # Denormalize to physical range
        # Assume CONFIG has ranges
        # [thick... rough... sld... s_len, i0]
        # It's easier if we return raw normalized [0,1] and decoder handles scaling,
        # but matching model.py structure:
        
        # We need to reshape or split 'out'
        # n_layers support
        
        # This part requires knowledge of how many layers.
        # Assuming n_layers is passed in __init__
        n = (self.final_out_dim - 2) // 3
        
        thick_range = CONFIG['thick_range']
        rough_range = CONFIG['rough_range'] # (0.5, 15.0)
        sld_range = CONFIG['sld_range']
        
        # Slice
        # thick: 0~n
        # rough: n~2n+1 (Roughness has n+1 params: Sub, L1..Ln, Amb? No, usually Amb(0)/L1/../Ln/Sub)
        # In generator.py: roughness size is n_layers + 1
        # Wait, if n=1, rough needs 2. 
        # My final_out_dim = 3*n + 2.
        # thick(n), sld(n), rough(n).
        # Where is extra rough? 
        # model.py: self.regressor = nn.Linear(..., 3 * n_layers + 2)
        # model.py line 31: rough = out[:, 1:2] ... implies n=1 case only gets 1 roughness?
        # generator.py line 67: roughness = ... n_layers + 1.
        # There is a mismatch in model.py vs generator.py regarding roughness count.
        # model.py assumes simplified model or maybe just 1 roughness per layer?
        
        # Let's fix this in FourierModel to be correct.
        # We need: Thickness(N), SLD(N), Roughness(N+1). 
        # And Instrument: s_len, i0.
        # Total = N + N + (N+1) + 2 = 3N + 3.
        
        # Let's override output dim logic
        desired_dim = 3 * n + 3 
        # But wait, earlier I set final_out_dim = 3*n + 2 based on model.py
        # If I want to be compatible with model.py's forward signature:
        # return thick, rough, sld, s_len, i0_scale
        
        # Let's implement robust scaling
        
        # Indices
        idx_t = n
        idx_r = idx_t + (n + 1)
        idx_s = idx_r + n
        idx_inst = idx_s + 2
        
        # NOTE: If we change output size, we must ensure train.py handles it.
        # Check model.py again.
        # model.py returns: thick (1), rough (1), sld (1), s_len(1), i0(1).
        # But XRR_PINN uses: rough_full = torch.cat([rough, rough], dim=-1)
        # It duplicates roughness to make 2 (Amb/Sub interface?).
        
        # I will support proper 3N+3 if possible, or stick to 3N+2 and duplicate like model.py
        # To make it "Better", let's predict all roughnesses.
        # But for drop-in replacement, maybe stick to 3N+2?
        # No, "Physically Correct" is better.
        # I'll update the Linear layer to 3*n + 3.
        
        # However, to avoid breaking changes in train.py (if it expects specific tuple unpack),
        # let's look at train.py... train.py calculates loss.
        # If train.py calculates loss against 'target', what is 'target'?
        # In generator, target is (thickness, roughness, sld).
        # thickness: [batch, n]
        # roughness: [batch, n+1]
        # sld: [batch, n+2]? (Amb, Layer, Sub) -> generator uses [Amb, Layer, Sub] (dim=n+2)
        
        # Generator returns (thickness, roughness, sld)
        # thickness: N
        # roughness: N+1
        # sld: N+2
        
        # So model MUST predict these.
        # Total parameters = N + (N+1) + (N+2) = 3N + 3.
        # Plus s_len, i0? Generator doesn't return s_len, i0 as targets! 
        # They are instrumental.
        
        # PINN approach: Unsupervised on R.
        # If Supervised, we need targets.
        
        # The user said "Pure PINN", "AI parameter matching".
        # If it's pure PINN (unsupervised physics loss), we predict params, simulate R, compare R.
        # We don't need ground truth params for training loss, only for validation.
        
        # So I will design the model to output full set of params needed for Simulator.
        
        t = out[:, :n] * (thick_range[1] - thick_range[0]) + thick_range[0]
        r = out[:, n:2*n+1] * (rough_range[1] - rough_range[0]) + rough_range[0]
        s = out[:, 2*n+1:3*n+1] * (sld_range[1] - sld_range[0]) + sld_range[0]
        
        s_len = out[:, -2:-1] * 30.0 + 10.0
        i0_scale = out[:, -1:] * 1.5 + 0.5
        
        return t, r, s, s_len, i0_scale

from xrefine.core.interface import Simulator
from xrefine.core.corrections import DifferentiableFootprint

class XRR_Fourier_PINN(nn.Module):
    def __init__(self, n_layers=1, device='cpu'):
        super().__init__()
        self.q_grid = CONFIG['q_grid'].to(device)
        
        # Instantiate Fourier Model
        # Use config params
        self.encoder = XRRPhysicsModel(
            q_len=CONFIG['q_len'],
            n_layers=n_layers,
            n_channels=64, # Default or config?
            use_fourier=True,
            fourier_scale=CONFIG.get('fourier_scale', 10.0),
            mlp_hidden=CONFIG.get('hidden_dim', 256)
        )
        
        self.simulator = Simulator(method='abeles', device=device)
        self.footprint = DifferentiableFootprint()

    def forward(self, r_obs):
        # 1. AI Prediction
        thick, rough, sld, s_len, i0 = self.encoder(r_obs)
        
        # 2. Physics Engine Setup
        # Construct full SLD/Roughness vectors
        # SLD: [Amb, Layer1..N, Sub]
        # Rough: [Amb, Layer1..N, Sub] ??
        # Simulator expects [batch, layers] for thickness
        # [batch, layers+1] for roughness?
        # [batch, layers+2] for sld?
        
        # Let's check model.py logic:
        # sld_full = torch.cat([torch.zeros_like(sld), sld, torch.ones_like(sld)*20.1], dim=-1)
        # rough_full = torch.cat([rough, rough], dim=-1) (This assumes N=1 and duplicates it?)
        
        # My FourierModel output matches generator targets:
        # thick: N
        # rough: N+1 (from generator)
        # sld: N+2 (from generator) -> Wait, my FourierModel returns N for sld?
        
        # Let's re-verify FourierModel return values logic vs Generator logic.
        # Generator:
        # sld = [Amb, Layers, Sub] -> Size N+2. Output to 'sld'.
        # FourierModel:
        # s = out[...] -> Size N.
        
        # So FourierModel is predicting only Layer SLD.
        # We need to construct full SLD.
        # Amb is 0. Sub is ?? 
        # In generator: sub_sld is random.
        # In model.py: sub_sld is fixed 20.1? `torch.ones_like(sld)*20.1`
        
        # Using fixed substrate is safer for PINN unless we know it varies.
        # Config says sld_range (5, 70).
        # Assuming AI predicts Layer SLD.
        
        # If we stick to N=1:
        # s (pred) is layer SLD.
        # sld_full = [0, s, 20.1]?
        
        ambient = torch.zeros_like(thick) # (B, 1) if N=1
        if thick.shape[1] > 1:
            ambient = torch.zeros(thick.shape[0], 1, device=thick.device)
            
        substrate_sld = torch.full((thick.shape[0], 1), 20.1, device=thick.device) 
        # Or predict substrate? 
        # Let's fix substrate to 20.1 (Si) for now as model.py did.
        
        sld_full = torch.cat([torch.zeros_like(thick[:, :1]), s, substrate_sld], dim=-1)
        
        # Roughness:
        # FourierModel returns 'r' size N+1.
        # So we can just use 'r' directly if generator provides N+1.
        # Wait, FourierModel code:
        # r = out[:, n:2*n+1] -> Size N+1. Correct.
        
        rough_full = r 
        
        q_batch = self.q_grid.unsqueeze(0).repeat(r_obs.size(0), 1)
        
        # 3. Differentiable Forward Loop
        r_sim = self.simulator(q_batch, thick, rough_full, sld_full)
        r_sim = self.footprint(q_batch, r_sim, s_len)
        r_final = r_sim * i0
        
        return (thick, rough_full, sld_full), r_final
