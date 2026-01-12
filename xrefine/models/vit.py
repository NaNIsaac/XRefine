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

Fourier Feature 옵션:
- mapping_size > 0: Fourier Feature 활성화 (고주파 패턴 포착 향상)
- mapping_size = 0: Fourier 비활성화 (원본 ViT 동작)
"""
import math
import torch
import torch.nn as nn
from dataclasses import fields

from xrefine.core.structures import ParamSet
from xrefine.core.abeles import AbelesMatrix
from xrefine.config import XRefineConfig


class FourierFeatureExtractor(nn.Module):
    """
    Random Fourier Feature Mapping (NeRF-style)
    고주파 패턴(Fringe 간섭)을 신경망이 더 잘 학습하도록 위치 인코딩.
    
    Input: (B, 1, L) - normalized log reflectivity
    Output: (B, 2*mapping_size, L) - sin/cos encoded features
    """
    def __init__(self, mapping_size: int = 64, scale: float = 40.0):
        super().__init__()
        self.mapping_size = mapping_size
        # Random Fourier basis (frozen, not learnable)
        B_matrix = torch.randn(1, mapping_size) * scale
        self.register_buffer('B', B_matrix)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 1, L)
        B_expanded = self.B.unsqueeze(-1)  # (1, mapping_size, 1)
        projected = 2 * math.pi * x * B_expanded  # (B, mapping_size, L)
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=1)  # (B, 2*mapping_size, L)


class PatchEmbed1D(nn.Module):
    """
    1D 신호를 Patch 단위로 잘라 임베딩 (ViT의 핵심)
    - Conv1d를 사용하여 'Sliding Window' 효과와 'Feature Projection'을 동시에 수행
    """
    def __init__(self, in_chans: int = 1, embed_dim: int = 128, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size  # Non-overlapping patches
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, in_chans, Seq_Len)
        x = self.proj(x)  # -> (Batch, Embed_Dim, Num_Patches)
        x = x.transpose(1, 2)  # -> (Batch, Num_Patches, Embed_Dim)
        x = self.norm(x)
        return x


class XRR_ViT_PINN(nn.Module):
    """
    Vision Transformer for XRR PINN.
    
    ViT treats the 1D XRR curve as a sequence of patches, similar to how
    image ViT treats an image as a sequence of patch embeddings.
    
    Fourier Feature:
    - mapping_size > 0: Enable Fourier (better high-freq pattern capture)
    - mapping_size = 0: Disable Fourier (original ViT behavior)
    """
    def __init__(self, config: XRefineConfig):
        super().__init__()
        self.config = config
        
        # --- Hyperparameters ---
        embed_dim = config.model.hidden_dim
        depth = config.model.encoder_depth
        num_heads = config.model.n_heads
        patch_size = getattr(config.model, 'patch_size', 16)
        mapping_size = getattr(config.model, 'mapping_size', 0)
        fourier_scale = getattr(config.model, 'fourier_scale', 40.0)
        input_len = config.q_len
        
        # --- 0. Optional Fourier Feature ---
        self.use_fourier = mapping_size > 0
        if self.use_fourier:
            self.fourier = FourierFeatureExtractor(
                mapping_size=mapping_size, 
                scale=fourier_scale
            )
            in_chans = 2 * mapping_size  # sin + cos
        else:
            self.fourier = None
            in_chans = 1
        
        # --- 1. Patch Embedding ---
        self.patch_embed = PatchEmbed1D(
            in_chans=in_chans, 
            embed_dim=embed_dim, 
            patch_size=patch_size
        )
        
        num_patches = input_len // patch_size
        
        # --- 2. Learnable Tokens ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=config.model.dropout)

        # --- 3. Transformer Encoder (Brain) ---
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

        # --- 4. Prediction Head (Output) ---
        n_layers = getattr(config.model, 'n_layers', 1)
        self.output_dim = sum([
            n_layers if f.metadata.get('mult') == 'n' else 1 
            for f in fields(ParamSet)
        ])
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(embed_dim * 2, self.output_dim)
        )

        # --- 5. PINN Physics Engine ---
        self.simulator = AbelesMatrix(device=config.device)
        self.register_buffer('q_grid', config.q_grid)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def device(self):
        return next(self.parameters()).device

    def _unnormalize(self, raw_out: torch.Tensor) -> ParamSet:
        """신경망 출력(Sigmoid) -> 물리 파라미터 범위로 변환"""
        out_norm = torch.sigmoid(raw_out)
        
        n = getattr(self.config.model, 'n_layers', 1)
        widths = [n if f.metadata.get('mult') == 'n' else 1 for f in fields(ParamSet)]
        chunks = torch.split(out_norm, widths, dim=1)
        
        params_dict = {}
        for (f, val) in zip(fields(ParamSet), chunks):
            p_min, p_max = getattr(self.config.ranges, f.name)
            params_dict[f.name] = val * (p_max - p_min) + p_min
            
        return ParamSet(**params_dict)

    def validate_physics(self, p: ParamSet) -> torch.Tensor:
        """물리적 제약 조건 페널티"""
        penalty_f = torch.mean(torch.relu(p.sig_f - 0.5 * p.d_f))
        penalty_ox = torch.mean(torch.relu(p.sig_ox - 0.5 * p.d_ox))
        return penalty_f + penalty_ox

    def forward(self, r_obs_log: torch.Tensor):
        """
        Training Forward Pass.
        
        Args:
            r_obs_log: (Batch, Q_Len) - Log10 Scale reflectivity
            
        Returns:
            (Params, R_pred, Penalty) - Compatible with Trainer
        """
        if torch.isnan(r_obs_log).any():
            r_obs_log = torch.nan_to_num(r_obs_log, nan=-15.0)

        # (B, L) -> (B, 1, L)
        if r_obs_log.dim() == 2:
            x = r_obs_log.unsqueeze(1)
        else:
            x = r_obs_log

        # 1. Input Normalization (-15 ~ 0 -> roughly -1 ~ 1)
        x = (torch.clamp(x, min=-15.0, max=0.0) + 7.5) / 7.5

        # 2. Optional Fourier Feature Mapping
        if self.use_fourier:
            x = self.fourier(x)  # (B, 1, L) -> (B, 2*mapping_size, L)

        # 3. Patch Embedding
        x = self.patch_embed(x)  # (B, N_Patches, Embed_Dim)
        B = x.shape[0]

        # 3. Add CLS Token & Positional Encoding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, E)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        # 4. Transformer Blocks
        x = self.blocks(x)
        x = self.norm(x)

        # 5. Prediction (Only use CLS token)
        cls_out = x[:, 0]
        raw_out = self.head(cls_out)
        
        # 6. Physics Reconstruction (PINN Logic)
        params = self._unnormalize(raw_out)
        
        thickness, roughness, sld_full = params.assemble_structure()
        
        q_batch = self.q_grid.unsqueeze(0).expand(B, -1)
        if hasattr(params, 'tth_offset'):
            q_batch = q_batch + params.tth_offset * 0.01

        r_sim = self.simulator(
            q=q_batch,
            thickness=thickness,
            roughness=roughness,
            sld=sld_full,
            sample_length=params.L,
            beam_width=params.beam_width,
            wavelength=self.config.instrument.wavelength
        )
        
        r_final = (params.i0 * r_sim) + torch.pow(10.0, params.bkg)
        r_final = torch.clamp(r_final, min=1e-15, max=2.0)
        
        phys_penalty = self.validate_physics(params)

        return params, r_final, phys_penalty

    def forward_with_params(self, params: ParamSet):
        """Refinement/Inference용 Forward"""
        device = self.device
        params.to(device)
        
        thickness, roughness, sld_full = params.assemble_structure()
        q_batch = self.q_grid.unsqueeze(0).expand(thickness.size(0), -1).to(device)
        
        if hasattr(params, 'tth_offset'):
            q_batch = q_batch + params.tth_offset * 0.01
            
        r_sim = self.simulator(
            q=q_batch, 
            thickness=thickness, 
            roughness=roughness, 
            sld=sld_full,
            sample_length=params.L, 
            beam_width=params.beam_width,
            wavelength=self.config.instrument.wavelength
        )
        
        r_final = (params.i0 * r_sim) + torch.pow(10.0, params.bkg)
        r_final = torch.clamp(r_final, min=1e-15, max=2.0)
        
        penalty = self.validate_physics(params)
        return params, r_final, penalty
