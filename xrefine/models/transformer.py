"""
Transformer-VAE Physics-Informed Neural Network
===============================================
구조:
1. Input (Log R) -> Fourier Feature Mapping (High Freq Encoding)
2. Linear Projection + Positional Encoding
3. Transformer Encoder (Self-Attention) -> Global Pooling
4. VAE Latent Space (Probabilistic)
5. Physics Decoder (Abeles Matrix)
"""
import torch
import torch.nn as nn
import math
from dataclasses import fields

from xrefine.core.structures import ParamSet
from xrefine.core.abeles import AbelesMatrix
from xrefine.config import XRefineConfig

# ==============================================================================
# 1. Helper Modules
# ==============================================================================

class FourierFeatureMapping(nn.Module):
    """
    [핵심] 입력 신호(q)를 고차원 주파수 영역으로 매핑.
    Transformer가 미세한 Fringe 패턴을 학습하는 데 필수적임.
    """
    def __init__(self, mapping_size: int = 64, scale: float = 40.0):
        super().__init__()
        self.mapping_size = mapping_size
        # Random Gaussian Matrix (Fixed)
        self.register_buffer('B', torch.randn(1, mapping_size) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq_Len) or (Batch, Seq_Len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1) # (B, S, 1)
            
        # Projection: (B, S, 1) @ (1, M) -> (B, S, M)
        # Transpose 불필요 (Transformer는 [Batch, Seq, Feat]를 원함)
        x_proj = x @ self.B 
        x_proj = x_proj * 2 * torch.pi
        
        # [Sin, Cos] 결합 -> (B, S, 2*M)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PositionalEncoding(nn.Module):
    """
    순서가 있는 시계열 데이터(XRR)에 위치 정보를 주입
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (Batch, Seq_Len, d_model)
        return x + self.pe[:, :x.size(1), :]

# ==============================================================================
# 2. Transformer-VAE Encoder (Brain)
# ==============================================================================

class XRRTransformerVAE(nn.Module):
    def __init__(self, config: XRefineConfig):
        super().__init__()
        
        # --- Hyperparameters ---
        d_model = config.model.hidden_dim
        nhead = config.model.n_heads
        num_layers = config.model.encoder_depth
        input_len = config.q_len
        mapping_size = config.model.mapping_size
        
        # --- 1. Input Embedding ---
        self.fourier = FourierFeatureMapping(mapping_size=mapping_size, scale=config.model.fourier_scale)
        
        # Input Dimension: Original(1) + Fourier(2*M)
        input_dim = 1 + (mapping_size * 2)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_len)
        
        # --- 2. Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=config.model.dim_feedforward, 
            dropout=config.model.dropout, 
            batch_first=True, 
            activation=config.model.activation
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- 3. VAE Head ---
        # ParamSet 구조를 분석하여 출력 차원 자동 계산
        # config.model.n_layers가 있으면 반영 (Multilayer Film)
        n_layers = getattr(config.model, 'n_layers', 1)
        self.output_dim = sum([n_layers if f.metadata.get('mult') == 'n' else 1 for f in fields(ParamSet)])
        
        self.fc_mu = nn.Linear(d_model, self.output_dim)
        self.fc_logvar = nn.Linear(d_model, self.output_dim)

    def forward(self, x):
        # x: (Batch, Seq_Len)
        
        # 1. Feature Engineering
        if x.dim() == 2: x = x.unsqueeze(-1)
        # Intensity Scaling (-9~0 -> 0~1)
        x_norm = (torch.clamp(x, min=-9.0, max=0.0) + 9.0) / 9.0
        
        # Fourier Mapping
        fourier_feat = self.fourier(x_norm) # (B, S, 2M)
        src = torch.cat([x_norm, fourier_feat], dim=-1) # (B, S, 1+2M)
        
        # 2. Embedding
        src = self.input_proj(src) # (B, S, d_model)
        src = self.pos_encoder(src)
        
        # 3. Encoding
        memory = self.transformer(src) # (B, S, d_model)
        
        # 4. Pooling (Global Average)
        latent = memory.mean(dim=1) # (B, d_model)
        
        # 5. Probabilistic Output
        mu = self.fc_mu(latent)
        logvar = self.fc_logvar(latent)
        
        # Reparameterization Trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu # Inference 모드에서는 Mean 사용
            
        return z, mu, logvar

# ==============================================================================
# 3. PINN Wrapper (Physics Integration)
# ==============================================================================

class XRR_Transformer_PINN(nn.Module):
    def __init__(self, config: XRefineConfig):
        super().__init__()
        self.config = config
        
        # (1) AI Encoder (Brain)
        self.encoder = XRRTransformerVAE(config)
        
        # (2) Physics Simulator (Body)
        self.simulator = AbelesMatrix(device=config.device)
        self.register_buffer('q_grid', config.q_grid)

    @property
    def device(self):
        return next(self.parameters()).device

    def _map_to_physics(self, z: torch.Tensor) -> ParamSet:
        """
        Latent Vector Z -> ParamSet (물리 범위 매핑)
        [Zero-Magic Number] ParamSet.names()를 활용한 동적 매핑
        """
        # ParamSet 필드 구성에 맞춰 z 텐서를 쪼갬 (Split)
        n = getattr(self.config.model, 'n_layers', 1)
        widths = [n if f.metadata.get('mult') == 'n' else 1 for f in fields(ParamSet)]
        
        chunks = torch.split(z, widths, dim=1)
        
        p_dict = {}
        for i, (field_info, chunk) in enumerate(zip(fields(ParamSet), chunks)):
            key = field_info.name
            
            # Sigmoid로 0~1 정규화
            val_norm = torch.sigmoid(chunk)
            
            # Config 범위로 스케일링
            p_min, p_max = getattr(self.config.ranges, key)
            val_phys = val_norm * (p_max - p_min) + p_min
            
            p_dict[key] = val_phys
            
        return ParamSet(**p_dict)

    def validate_physics(self, p: ParamSet) -> torch.Tensor:
        """물리적 제약 조건 (Roughness vs Thickness)"""
        penalty_f = torch.mean(torch.relu(p.sig_f - 0.5 * p.d_f))
        penalty_ox = torch.mean(torch.relu(p.sig_ox - 0.5 * p.d_ox))
        return penalty_f + penalty_ox

    def forward(self, r_obs_log: torch.Tensor):
        """
        Training Forward Pass
        Returns: (Params, R_pred, (mu, logvar, penalty))
        """
        # NaN Handling
        if torch.isnan(r_obs_log).any():
            r_obs_log = torch.nan_to_num(r_obs_log, nan=-10.0)

        # 1. Encoder (VAE)
        z, mu, logvar = self.encoder(r_obs_log)
        
        # 2. Physics Mapping
        params = self._map_to_physics(z)
        
        # 3. Physics Simulation Setup
        batch_size = z.size(0)
        q_batch = self.q_grid.unsqueeze(0).expand(batch_size, -1) # (B, Q)
        
        # Offset Correction
        if hasattr(params, 'tth_offset'):
            q_batch = q_batch + params.tth_offset * 0.01

        # Structure Assembly
        thickness, roughness, sld_full = params.assemble_structure()
        
        # 4. Run Simulator
        r_sim = self.simulator(
            q=q_batch,
            thickness=thickness,
            roughness=roughness,
            sld=sld_full,
            sample_length=params.L,
            beam_width=params.beam_width,
            wavelength=self.config.instrument.wavelength
        )
        
        # 5. Background & Scaling
        r_final = (params.i0 * r_sim) + torch.pow(10.0, params.bkg)
        r_final = torch.clamp(r_final, min=1e-11, max=2.0)
        
        # 6. Loss Terms Collection
        phys_penalty = self.validate_physics(params)
        
        # Trainer에서 KL Divergence와 Penalty를 모두 처리하도록 튜플로 반환
        return params, r_final, (mu, logvar, phys_penalty)

    def forward_with_params(self, params: ParamSet):
        """Refinement / Inference용 Forward"""
        device = self.device
        params.to(device)
        
        thickness, roughness, sld_full = params.assemble_structure()
        q_batch = self.q_grid.unsqueeze(0).expand(thickness.size(0), -1).to(device)
        
        if hasattr(params, 'tth_offset'):
            q_batch = q_batch + params.tth_offset * 0.01
            
        r_sim = self.simulator(
            q=q_batch, thickness=thickness, roughness=roughness, sld=sld_full,
            sample_length=params.L, beam_width=params.beam_width,
            wavelength=self.config.instrument.wavelength
        )
        
        r_final = (params.i0 * r_sim) + torch.pow(10.0, params.bkg)
        r_final = torch.clamp(r_final, min=1e-11, max=2.0)
        
        penalty = self.validate_physics(params)
        return params, r_final, penalty