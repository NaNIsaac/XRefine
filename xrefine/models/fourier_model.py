"""
CNN-based Fourier PINN Model (Dilated CNN Encoder)
===================================================
Dilated Convolutional Network로 넓은 수용장을 확보하여 Fringe 주기 파악.
"""
import torch
import torch.nn as nn
from torch import Tensor

from xrefine.config import XRefineConfig
from xrefine.models.common import BasePINNModel, FourierFeatureMapping


class DilatedCNNEncoder(nn.Module):
    """
    Dilated CNN Encoder: 넓은 수용장으로 Fringe 주기를 파악.
    Fourier Feature + Multi-scale Dilated Convolution.
    """
    def __init__(self, config: XRefineConfig, output_dim: int):
        super().__init__()
        
        mapping_size = config.model.mapping_size
        scale = config.model.fourier_scale
        base_channels = config.model.hidden_dim
        kernel_size = config.model.kernel_size
        
        self.fourier = FourierFeatureMapping(
            mapping_size=mapping_size, 
            scale=scale, 
            output_format='cnn'
        )
        
        # Input: Intensity(1) + Fourier(2M)
        in_channels = 1 + (mapping_size * 2)
        
        layers = []
        curr_dim = in_channels
        
        for i in range(config.model.encoder_depth):
            out_dim = base_channels * (2 ** min(i // 2, 2))
            dilation = 2 ** i
            padding = ((kernel_size - 1) * dilation) // 2
            
            layers.append(nn.Sequential(
                nn.Conv1d(
                    curr_dim, out_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=False
                ),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool1d(2)
            ))
            curr_dim = out_dim

        self.encoder = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.regressor = nn.Sequential(
            nn.Linear(curr_dim, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(base_channels, base_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(base_channels // 2, output_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, Q_Len) - Log10 reflectivity
        Returns:
            (B, output_dim) - Raw output (before sigmoid)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, Q)
        
        # Intensity Normalization (-9 ~ 0 -> 0 ~ 1)
        x_norm = (torch.clamp(x, min=-9.0, max=0.0) + 9.0) / 9.0
        
        # Fourier Features
        fourier_feat = self.fourier(x_norm)  # (B, 2M, Q)
        x_in = torch.cat([x_norm, fourier_feat], dim=1)  # (B, 1+2M, Q)
        
        feat = self.encoder(x_in)
        feat = self.global_pool(feat).view(feat.size(0), -1)
        
        return self.regressor(feat)


class XRR_Fourier_PINN(BasePINNModel):
    """
    Dilated CNN 기반 PINN 모델.
    Fourier Feature Mapping으로 고주파 Fringe 패턴 포착.
    """
    def __init__(self, config: XRefineConfig, device='cpu'):
        super().__init__(config)
        self.encoder_net = DilatedCNNEncoder(config, self.output_dim)

    def encode(self, x: Tensor) -> Tensor:
        """CNN Encoder로 파라미터 추출"""
        return self.encoder_net(x)