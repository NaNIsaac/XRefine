"""
Vision Transformer (ViT) Physics-Informed Neural Network
=========================================================
XRR 데이터를 패치 기반으로 처리하는 ViT 아키텍처.

구조:
1. Input (Log R) -> [Optional] Fourier Feature -> Patch Embedding
2. [CLS] Token + Positional Encoding
3. Transformer Encoder (Self-Attention)
4. [CLS] Token -> Prediction Head
5. Physics Decoder (Abeles Matrix)
"""
import torch
import torch.nn as nn
from torch import Tensor

from xrefine.config import XRefineConfig
from xrefine.constants import LOG_MASK_VALUE
from xrefine.models.common import (
    BasePINNModel, 
    FourierFeatureMapping,
    PatchEmbed1D
)


class ViTEncoder(nn.Module):
    """Vision Transformer Encoder for 1D XRR signals"""
    
    def __init__(self, config: XRefineConfig, output_dim: int):
        super().__init__()
        
        embed_dim = config.model.hidden_dim
        depth = config.model.encoder_depth
        num_heads = config.model.n_heads
        patch_size = getattr(config.model, 'patch_size', 16)
        mapping_size = getattr(config.model, 'mapping_size', 0)
        fourier_scale = getattr(config.model, 'fourier_scale', 40.0)
        input_len = config.q_len
        
        # 1. Optional Fourier Feature
        self.use_fourier = mapping_size > 0
        if self.use_fourier:
            self.fourier = FourierFeatureMapping(
                mapping_size=mapping_size,
                scale=fourier_scale,
                output_format='cnn'  # (B, 2M, L) format for Conv1d patch embed
            )
            in_channels = 2 * mapping_size
        else:
            self.fourier = None
            in_channels = 1
        
        # 2. Patch Embedding
        self.patch_embed = PatchEmbed1D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        num_patches = input_len // patch_size
        
        # 3. Learnable Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=config.model.dropout)
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            activation=config.model.activation,
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        
        # 5. Prediction Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(embed_dim * 2, output_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, Q_Len) - Log10 reflectivity
        Returns:
            (B, output_dim) - Raw output
        """
        # (B, L) -> (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Normalization (LOG_MASK_VALUE ~ 0 -> -1 ~ 1)
        x = (torch.clamp(x, min=LOG_MASK_VALUE, max=0.0) - LOG_MASK_VALUE/2) / (-LOG_MASK_VALUE/2)
        
        # Optional Fourier
        if self.use_fourier:
            x = self.fourier(x)  # (B, 2M, L)
        
        # Patch Embedding
        x = self.patch_embed(x)  # (B, N_Patches, Embed_Dim)
        batch_size = x.shape[0]
        
        # Add CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        
        # Transformer
        x = self.blocks(x)
        x = self.norm(x)
        
        # Only CLS token for prediction
        cls_out = x[:, 0]
        return self.head(cls_out)


class XRR_ViT_PINN(BasePINNModel):
    """
    Vision Transformer 기반 PINN 모델.
    패치 기반 처리로 지역/전역 패턴 동시 포착.
    """
    def __init__(self, config: XRefineConfig):
        super().__init__(config)
        self.encoder_net = ViTEncoder(config, self.output_dim)

    def encode(self, x: Tensor) -> Tensor:
        """ViT Encoder로 파라미터 추출"""
        return self.encoder_net(x)
