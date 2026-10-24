import torch
import torch.nn as nn
from torch import Tensor

from xrefine.config import XRefineConfig
from xrefine.constants import LOG_MASK_VALUE
from xrefine.models import register_model
from xrefine.models.common import BasePINNModel, FourierFeatureMapping


class FourierConvEncoder(nn.Module):
    """
    Ported XRRPhysicsModel Encoder logic.
    Structure: [Conv1d(7) -> BN -> Leaky -> Drop -> MaxPool(2)] x Depth
    """

    def __init__(self, config: XRefineConfig, output_dim: int):
        super().__init__()

        # Hyperparameter Mapping
        q_len = config.q_len
        n_channels = config.model.hidden_dim
        depth = config.model.encoder_depth
        mlp_hidden = config.model.dim_feedforward
        dropout = config.model.dropout

        # Fourier Settings
        mapping_size = config.model.mapping_size
        use_fourier = mapping_size > 0
        fourier_scale = config.model.fourier_scale

        self.use_fourier = use_fourier

        if use_fourier:
            self.fourier = FourierFeatureMapping(
                mapping_size=mapping_size,
                scale=fourier_scale,
                output_format="cnn",  # Returns (B, 2M, L)
            )
            # Legacy: Input(1) + Fourier(2M)
            in_channels = 1 + (mapping_size * 2)
        else:
            in_channels = 1

        layers = []
        curr_dim = in_channels

        for i in range(depth):
            # Channel scaling strategy
            out_dim = n_channels * (2 ** min(i, 3))

            layers.append(
                nn.Sequential(
                    nn.Conv1d(curr_dim, out_dim, kernel_size=7, padding=3, bias=False),
                    nn.BatchNorm1d(out_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout),
                    nn.MaxPool1d(kernel_size=2),
                )
            )
            curr_dim = out_dim

        self.backbone = nn.Sequential(*layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # +3 input dims for the concatenated global stats
        self.regressor = nn.Sequential(
            nn.Linear(curr_dim + 3, mlp_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(mlp_hidden // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, Q_Len) or (B, 1, Q_Len)
        Returns:
            (B, Output_Dim)
        """
        # (B, L) -> (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # global statistics of the input: max, low-q mean, std
        stat_max = x.max(dim=-1)[0]  # (B, 1)

        stat_low_q = x[..., :50].mean(dim=-1)  # (B, 1)

        stat_std = x.std(dim=-1)  # (B, 1)

        global_stats = torch.cat([stat_max, stat_low_q, stat_std], dim=1)

        # Intensity Normalization (-1 ~ 1)
        x_norm = (torch.clamp(x, min=LOG_MASK_VALUE, max=0.0) - LOG_MASK_VALUE / 2) / (
            -LOG_MASK_VALUE / 2
        )

        if self.use_fourier:
            fourier_feat = self.fourier(x_norm)
            x_in = torch.cat([x_norm, fourier_feat], dim=1)
        else:
            x_in = x_norm

        # CNN Backbone
        feat = self.backbone(x_in)  # (B, C, L_pooled)
        feat = self.global_pool(feat).flatten(1)  # (B, C) -> (B, Hidden_Dim)

        # [FIX 3] Feature Fusion (Conv Feature + Global Hint)
        # (B, Hidden) + (B, 3) -> (B, Hidden + 3)
        feat_enriched = torch.cat([feat, global_stats], dim=1)

        return self.regressor(feat_enriched)


@register_model("fourier_conv")
class FourierConvPINN(BasePINNModel):
    """
    Legacy CNN Implementation (Reference: Exp07)
    """

    def __init__(self, config: XRefineConfig):
        super().__init__(config)
        self.encoder_net = FourierConvEncoder(config, self.output_dim)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder_net(x)
