"""
Legacy CNN Model (Ported from ReflectoExp)
==========================================
Reference: runs/exp07_fourier_hires/xrr_model.py
Architecture: Standard CNN with Fourier Features
"""
import torch
import torch.nn as nn
from torch import Tensor

from xrefine.config import XRefineConfig
from xrefine.models.common import BasePINNModel, FourierFeatureMapping
from xrefine.models import register_model


class LegacyCNNEncoder(nn.Module):
    """
    Ported XRRPhysicsModel Encoder logic.
    Structure: [Conv1d(7) -> BN -> Leaky -> Drop -> MaxPool(2)] x Depth
    """
    def __init__(self, config: XRefineConfig, output_dim: int):
        super().__init__()
        
        # Hyperparameter Mapping
        q_len = config.q_len
        n_channels = config.model.hidden_dim        # default: 64 (legacy)
        depth = config.model.encoder_depth          # default: 6 (legacy)
        mlp_hidden = config.model.dim_feedforward   # default: 512 (legacy)
        dropout = config.model.dropout
        
        # Fourier Settings
        # Legacy used hardcoded 32 mapping size often, but we use config
        mapping_size = config.model.mapping_size
        use_fourier = mapping_size > 0
        fourier_scale = config.model.fourier_scale
        
        self.use_fourier = use_fourier
        
        # 1. Fourier Feature Mapping
        if use_fourier:
            self.fourier = FourierFeatureMapping(
                mapping_size=mapping_size,
                scale=fourier_scale,
                output_format='cnn' # Returns (B, 2M, L)
            )
            # Legacy: Input(1) + Fourier(2M)
            in_channels = 1 + (mapping_size * 2)
        else:
            in_channels = 1

        # 2. CNN Encoder
        layers = []
        curr_dim = in_channels
        
        for i in range(depth):
            # Legacy logic: channel doubles every layer until some limit?
            # Provided code: out_dim = n_channels * (2 ** min(i, 3))
            out_dim = n_channels * (2 ** min(i, 3))
            
            layers.append(nn.Sequential(
                nn.Conv1d(curr_dim, out_dim, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size=2)
            ))
            curr_dim = out_dim

        self.encoder = nn.Sequential(*layers)
        
        # 3. Global Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 4. Regressor (MLP)
        self.regressor = nn.Sequential(
            nn.Linear(curr_dim, mlp_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(mlp_hidden // 2, output_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, Q_Len)
        Returns:
            (B, Output_Dim)
        """
        # (B, L) -> (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Intensity Normalization (Same as ViT/Common)
        # LogR is roughly -10 to 0. Normalize to -1 to 1 or 0 to 1?
        # Legacy code didn't normalize inside model, it assumed input was prepared.
        # Common practice in this project:
        x_norm = (torch.clamp(x, min=-9.0, max=0.0) + 9.0) / 9.0
        
        if self.use_fourier:
            # Fourier returns (B, 2M, L)
            fourier_feat = self.fourier(x_norm)
            # Concatenate along channel dim: (B, 1+2M, L)
            x_in = torch.cat([x_norm, fourier_feat], dim=1)
        else:
            x_in = x_norm
            
        feat = self.encoder(x_in) # (B, C_last, L_pooled)
        feat = self.global_pool(feat).squeeze(-1) # (B, C_last)
        
        return self.regressor(feat)


@register_model("cnn")
class XRR_Fourier_PINN(BasePINNModel):
    """
    Legacy CNN Implementation (Reference: Exp07)
    """
    def __init__(self, config: XRefineConfig):
        super().__init__(config)
        self.encoder_net = LegacyCNNEncoder(config, self.output_dim)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder_net(x)
