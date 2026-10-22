"""
Latent Transformer-VAE PINN (Uncertainty Aware)
===============================================
구조:
1. Input (Log R) -> Fourier Feature Mapping
2. Linear Projection + Positional Encoding
3. Transformer Encoder (Self-Attention) -> Global Pooling
4. VAE Latent Space (Probabilistic)
5. Physics Decoder (Abeles Matrix)
"""

import torch
import torch.nn as nn
from torch import Tensor

from xrefine.config import XRefineConfig
from xrefine.constants import LOG_MASK_VALUE  # [FIX] 상수 임포트
from xrefine.core.structures import ParamSet
from xrefine.models import register_model
from xrefine.models.common import BasePINNModel, FourierFeatureMapping, PositionalEncoding


class TransformerVAEEncoder(nn.Module):
    """Transformer-VAE Encoder (Brain)"""

    def __init__(self, config: XRefineConfig, output_dim: int):
        super().__init__()

        d_model = config.model.hidden_dim
        n_heads = config.model.n_heads
        num_layers = config.model.encoder_depth
        input_len = config.q_len
        mapping_size = config.model.mapping_size

        # 1. Input Embedding
        self.fourier = FourierFeatureMapping(
            mapping_size=mapping_size, scale=config.model.fourier_scale, output_format="transformer"
        )

        # Input: Original(1) + Fourier(2M)
        input_dim = 1 + (mapping_size * 2)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_len)

        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            batch_first=True,
            activation=config.model.activation,
            norm_first=True,  # [Recommend] Pre-LN for better convergence
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. VAE Head
        self.fc_mu = nn.Linear(d_model, output_dim)
        self.fc_logvar = nn.Linear(d_model, output_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: (B, Seq_Len) - Log reflectivity
        Returns:
            z, mu, logvar
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, S, 1)

        # [FIX] Intensity Normalization (Sync with CNN/ViT logic)
        # LOG_MASK_VALUE(-15.0) ~ 0.0 범위를 0 ~ 1 (또는 -1 ~ 1)로 매핑
        # Transformer는 보통 0~1 입력을 선호하는 경향이 있음 (Positional Encoding과 스케일 매칭)
        range_val = -LOG_MASK_VALUE  # 15.0
        x_norm = (torch.clamp(x, min=LOG_MASK_VALUE, max=0.0) + range_val) / range_val

        # Fourier + Projection
        fourier_feat = self.fourier(x_norm)  # (B, S, 2M)
        src = torch.cat([x_norm, fourier_feat], dim=-1)  # (B, S, 1+2M)
        src = self.input_proj(src)  # (B, S, d_model)
        src = self.pos_encoder(src)

        # Transformer
        memory = self.transformer(src)  # (B, S, d_model)
        latent = memory.mean(dim=1)  # Global Average Pooling

        # VAE Output
        mu = self.fc_mu(latent)
        logvar = self.fc_logvar(latent)

        # Reparameterization
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        return z, mu, logvar


@register_model("latent_trans")  # Registry Key
class LatentTransPINN(BasePINNModel):  # Class Name Updated
    """
    Transformer-VAE 기반 PINN 모델.
    확률적 Latent Space로 불확실성 모델링.
    """

    def __init__(self, config: XRefineConfig):
        super().__init__(config)
        self.encoder_net = TransformerVAEEncoder(config, self.output_dim)

        # VAE 관련 정보 저장용
        self._mu = None
        self._logvar = None

    def encode(self, x: Tensor) -> Tensor:
        """Transformer-VAE Encoder로 파라미터 추출"""
        z, mu, logvar = self.encoder_net(x)
        self._mu = mu
        self._logvar = logvar
        return z

    def forward(
        self, r_obs_log: Tensor, fixed_values: ParamSet | None = None
    ) -> tuple[ParamSet, Tensor, tuple]:
        """
        Training Forward Pass.
        Returns: (params, r_pred, (mu, logvar, penalty))
        """
        if torch.isnan(r_obs_log).any():
            r_obs_log = torch.nan_to_num(r_obs_log, nan=LOG_MASK_VALUE)

        # Encode
        raw_output = self.encode(r_obs_log)

        # Decode & Merge (Fixed Params)
        params = self.unnormalize(torch.sigmoid(raw_output), fixed_values)

        # Simulate
        r_final = self.simulate(params)

        # Loss components
        penalty = self.validate_physics(params)

        return params, r_final, (self._mu, self._logvar, penalty)
