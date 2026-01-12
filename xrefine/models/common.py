"""
Common Components for PINN Models
==================================
중복 제거를 위한 공통 모듈:
- FourierFeatureMapping: Random Fourier Feature Encoding
- PositionalEncoding: Sinusoidal Position Embedding
- BasePINNModel: PINN 모델의 공통 인터페이스
"""
import math
from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from xrefine.core.abeles import AbelesMatrix
from xrefine.core.structures import ParamSet
from xrefine.config import XRefineConfig


# ==============================================================================
# 1. Feature Encoding Modules
# ==============================================================================

class FourierFeatureMapping(nn.Module):
    """
    Random Fourier Feature Mapping (NeRF-style).
    
    고주파 XRR Fringe 패턴을 신경망이 더 잘 학습하도록 위치 인코딩.
    scale이 높을수록 미세한 간섭 무늬 변화에 예민해짐.
    
    Usage:
        # CNN용 (output: B, 2M, L)
        fourier = FourierFeatureMapping(mapping_size=64, output_format='cnn')
        
        # Transformer용 (output: B, L, 2M)
        fourier = FourierFeatureMapping(mapping_size=64, output_format='transformer')
    """
    def __init__(
        self, 
        mapping_size: int = 64, 
        scale: float = 40.0,
        output_format: str = 'cnn'  # 'cnn' or 'transformer'
    ):
        super().__init__()
        self.mapping_size = mapping_size
        self.output_format = output_format
        
        # Random Gaussian Matrix (frozen, not learnable)
        random_matrix = torch.randn(1, mapping_size) * scale
        self.register_buffer('B', random_matrix)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor
               - CNN format: (B, 1, L) or (B, L)
               - Transformer format: (B, L) or (B, L, 1)
        
        Returns:
            - CNN format: (B, 2*mapping_size, L)
            - Transformer format: (B, L, 2*mapping_size)
        """
        if self.output_format == 'cnn':
            # CNN expects (B, C, L) format
            if x.dim() == 2:
                x = x.unsqueeze(1)  # (B, L) -> (B, 1, L)
            # (B, 1, L) -> (B, L, 1) @ (1, M) -> (B, L, M)
            x_proj = x.transpose(1, 2) @ self.B
            x_proj = x_proj * 2 * math.pi
            out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            return out.transpose(1, 2)  # (B, 2M, L)
        else:
            # Transformer expects (B, L, C) format
            if x.dim() == 2:
                x = x.unsqueeze(-1)  # (B, L) -> (B, L, 1)
            # (B, L, 1) @ (1, M) -> (B, L, M)
            x_proj = x @ self.B
            x_proj = x_proj * 2 * math.pi
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # (B, L, 2M)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for Transformers.
    순서가 있는 시계열 데이터(XRR)에 위치 정보를 주입.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """x: (Batch, Seq_Len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


# ==============================================================================
# 2. Base PINN Model (Abstract)
# ==============================================================================

class BasePINNModel(nn.Module, ABC):
    """
    PINN 모델의 공통 인터페이스.
    
    모든 PINN 모델이 공유하는 기능:
    - 물리 파라미터 변환 (_unnormalize)
    - 물리 제약 검증 (validate_physics)  
    - 외부 파라미터 시뮬레이션 (forward_with_params)
    - Device 속성
    """
    def __init__(self, config: XRefineConfig):
        super().__init__()
        self.config = config
        self.simulator = AbelesMatrix(device=config.device)
        self.register_buffer('q_grid', config.q_grid)
        
        # [REFAC] 출력 차원 계산 (Learnable 파라미터만!)
        self.n_layers = getattr(config.model, 'n_layers', 1)
        self.learnable_fields = [f for f in fields(ParamSet) if f.metadata.get('learnable', True)]
        self.fixed_fields = [f for f in fields(ParamSet) if not f.metadata.get('learnable', True)]
        
        self.output_dim = sum([
            self.n_layers if f.metadata.get('mult') == 'n' else 1 
            for f in self.learnable_fields
        ])

    @property
    def device(self) -> torch.device:
        """모델이 위치한 디바이스 반환"""
        return next(self.parameters()).device

    def unnormalize(self, raw_output: Tensor, fixed_values: ParamSet | None = None) -> ParamSet:
        """
        신경망 출력(0~1)을 물리 파라미터 범위로 변환하고 Fixed 파라미터와 병합.
        
        Args:
            raw_output: (B, output_dim) - Sigmoid 된 학습 파라미터
            fixed_values: (Optional) 고정 파라미터 값을 담은 ParamSet (Inference 시 사용)
            
        Returns:
            Full ParamSet (Learnable + Fixed)
        """
        batch_size = raw_output.size(0)
        
        # 1. Learnable Parameter Decoding
        widths = [
            self.n_layers if f.metadata.get('mult') == 'n' else 1 
            for f in self.learnable_fields
        ]
        chunks = torch.split(raw_output, widths, dim=1)
        
        params_dict = {}
        for fld, val in zip(self.learnable_fields, chunks):
            p_min, p_max = getattr(self.config.ranges, fld.name)
            params_dict[fld.name] = val * (p_max - p_min) + p_min
            
        # 2. Fixed Parameter Merging
        for fld in self.fixed_fields:
            if fixed_values is not None:
                # 외부에서 주입된 값 사용 (Training: GT, Inference: User Input)
                val = getattr(fixed_values, fld.name)
                # 배치 크기 맞추기 (Broadcasting)
                if val.size(0) != batch_size:
                    val = val.expand(batch_size, -1)
            else:
                # 기본값 사용 (Range의 중간값)
                p_min, p_max = getattr(self.config.ranges, fld.name)
                default_val = (p_min + p_max) / 2.0
                mult = fld.metadata.get('mult', 1)
                width = self.n_layers if mult == 'n' else 1
                val = torch.full((batch_size, width), default_val, device=self.device)
                
            params_dict[fld.name] = val

        return ParamSet(**params_dict)

    def validate_physics(self, params: ParamSet) -> Tensor:
        """
        물리적 제약 조건 검증.
        Roughness가 Thickness의 50%를 초과하면 페널티 부과.
        """
        penalty_film = torch.mean(torch.relu(params.sig_f - 0.5 * params.d_f))
        penalty_oxide = torch.mean(torch.relu(params.sig_ox - 0.5 * params.d_ox))
        return penalty_film + penalty_oxide

    def simulate(self, params: ParamSet) -> Tensor:
        """
        ParamSet으로부터 반사율 시뮬레이션.
        
        [IMPORTANT] 순수 물리 시뮬레이션만 수행!
        i0, bkg 적용은 data generation (augmentations.py)에서만 수행됨.
        모델 예측 시에도 동일하게 순수 반사율만 출력해야 타겟과 비교 가능.
        """
        thickness, roughness, sld_full = params.assemble_structure()
        batch_size = thickness.size(0)
        
        # Q 그리드 준비
        q_batch = self.q_grid.unsqueeze(0).expand(batch_size, -1).to(self.device)
        
        # 2theta offset 보정 (있으면 적용)
        if hasattr(params, 'tth_offset') and params.tth_offset is not None:
            q_batch = q_batch + params.tth_offset * 0.01
        
        # 물리 시뮬레이션 (순수 반사율)
        r_sim = self.simulator(
            q=q_batch,
            thickness=thickness,
            roughness=roughness,
            sld=sld_full,
            sample_length=params.L,
            beam_width=params.beam_width,
            wavelength=self.config.instrument.wavelength
        )
        
        # [FIX] i0, bkg applied here to match data generation
        r_final = (params.i0 * r_sim) + torch.pow(10.0, params.bkg)
        
        return torch.clamp(r_final, min=1e-15, max=2.0)

    def forward_with_params(self, params: ParamSet) -> Tuple[ParamSet, Tensor, Tensor]:
        """
        외부에서 주입된 파라미터로 시뮬레이션 (Refinement 단계용).
        """
        params = params.to(self.device)
        r_final = self.simulate(params)
        penalty = self.validate_physics(params)
        return params, r_final, penalty

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """
        서브클래스에서 구현: 입력 -> 정규화된 파라미터 출력 (0~1 범위)
        """
        pass

    def forward(self, r_obs_log: Tensor, fixed_values: ParamSet | None = None) -> Tuple[ParamSet, Tensor, any]:
        """
        Training Forward Pass.
        
        Args:
            r_obs_log: (B, Q_Len) - Log10 Scale reflectivity
            fixed_values: (B, ...) - Fixed parameters (GT for training)
            
        Returns:
            (params, r_pred, extra) - extra는 모델별로 다름 (penalty 또는 VAE 정보)
        """
        # NaN 처리
        if torch.isnan(r_obs_log).any():
            r_obs_log = torch.nan_to_num(r_obs_log, nan=-10.0)
        
        # 인코더로 파라미터 추출
        raw_output = self.encode(r_obs_log)
        
        # 파라미터 디코딩 및 병합
        params = self.unnormalize(torch.sigmoid(raw_output), fixed_values)
        
        # 시뮬레이션
        r_final = self.simulate(params)
        
        # 물리 페널티
        penalty = self.validate_physics(params)
        
        return params, r_final, penalty


# ==============================================================================
# 3. Patch Embedding (ViT용)
# ==============================================================================

class PatchEmbed1D(nn.Module):
    """
    1D 신호를 Patch 단위로 임베딩 (ViT의 핵심).
    Conv1d로 Sliding Window + Feature Projection 동시 수행.
    """
    def __init__(
        self, 
        in_channels: int = 1, 
        embed_dim: int = 128, 
        patch_size: int = 16
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size  # Non-overlapping patches
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, Seq_Len)
        Returns:
            (B, Num_Patches, Embed_Dim)
        """
        x = self.proj(x)  # -> (B, Embed_Dim, Num_Patches)
        x = x.transpose(1, 2)  # -> (B, Num_Patches, Embed_Dim)
        return self.norm(x)
