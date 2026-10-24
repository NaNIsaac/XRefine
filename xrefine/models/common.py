"""Common Components for PINN Models
==================================
Shared modules to avoid duplication:
- FourierFeatureMapping: Random Fourier Feature Encoding
- PositionalEncoding: Sinusoidal Position Embedding
- BasePINNModel: common interface of the PINN models
"""

import math
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from xrefine.config import XRefineConfig
from xrefine.core.abeles import AbelesMatrix
from xrefine.core.smearing import instrument_reflectivity
from xrefine.core.structures import ParamSet  # Aliased DynamicParamSet
from xrefine.data.normalization import apply_normalization_tensor


class FourierFeatureMapping(nn.Module):
    """
    Random Fourier Feature Mapping (NeRF-style).
    """

    def __init__(self, mapping_size: int = 64, scale: float = 40.0, output_format: str = "cnn"):
        super().__init__()
        self.mapping_size = mapping_size
        self.output_format = output_format
        random_matrix = torch.randn(1, mapping_size) * scale
        self.register_buffer("B", random_matrix)

    def forward(self, x: Tensor) -> Tensor:
        if self.output_format == "cnn":
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x_proj = x.transpose(1, 2) @ self.B
            x_proj = x_proj * 2 * math.pi
            out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            return out.transpose(1, 2)
        else:
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            x_proj = x @ self.B
            x_proj = x_proj * 2 * math.pi
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for Transformers.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1), :]


class BasePINNModel(nn.Module, ABC):
    """Common interface of the PINN models."""

    def __init__(self, config: XRefineConfig):
        super().__init__()
        self.config = config
        self.simulator = AbelesMatrix(device=config.device)
        self.register_buffer("q_grid", config.q_grid)

        # [REFAC] Calculate output dimension based on Dynamic Layer Config
        self.param_map = self._build_param_map(config)
        self.output_dim = len(self.param_map)

    def _build_param_map(self, config: XRefineConfig) -> list[tuple[str, list[float]]]:
        """
        Builds a flattened list of (param_key, range) for the neural network output head.
        Returns: [('Film.thickness', [100, 500]), ('Film.roughness', [1, 5]), ...]
        """
        mapping = []

        for layer in config.sample.layers:
            mapping.append((f"{layer.name}.thickness", layer.thickness))
            mapping.append((f"{layer.name}.roughness", layer.roughness))
            mapping.append((f"{layer.name}.sld", layer.sld))

            if getattr(layer, "sld_imag", None) is not None:
                mapping.append((f"{layer.name}.sld_imag", layer.sld_imag))

        mapping.append(("Substrate.roughness", config.sample.substrate.roughness))
        mapping.append(("Substrate.sld", config.sample.substrate.sld))

        if getattr(config.sample.substrate, "sld_imag", None) is not None:
            mapping.append(("Substrate.sld_imag", config.sample.substrate.sld_imag))

        for key in ParamSet.INSTRUMENT_KEYS:
            if hasattr(config.sample, key):
                rng = getattr(config.sample, key)
                if rng is not None:
                    mapping.append((key, rng))

        return mapping

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def unnormalize(self, raw_output: Tensor, fixed_values: ParamSet | None = None) -> ParamSet:
        """
        Convert NN output (0~1) to physical ranges.
        Constructs a DynamicParamSet.

        [Feature] Constraint Learning:
        If 'fixed_values' is provided, parameters present in it will OVERWRITE
        the network predictions. This cuts the gradient for those parameters,
        treating them as constants (Ground Truth).
        """
        batch_size = raw_output.size(0)

        # Split output into chunks of 1 (since all scalars)
        vals = torch.split(raw_output, 1, dim=1)

        params_dict = {}

        for (key, (p_min, p_max)), val in zip(self.param_map, vals, strict=False):
            # val: (B, 1) Normalized 0~1
            physical_val = val * (p_max - p_min) + p_min
            params_dict[key] = physical_val

        # fixed values override the predictions
        if fixed_values is not None:
            for key, target_val in fixed_values._params.items():
                if key in params_dict:
                    if target_val.size(0) == 1 and batch_size > 1:
                        target_val = target_val.expand(batch_size, -1)

                    # overwrite the prediction; this parameter is detached from the network's
                    # computational graph
                    params_dict[key] = target_val

        # Order of layer names for assembly
        layer_names = [l.name for l in self.config.sample.layers]

        return ParamSet(params_dict, layer_names, device=self.device)

    def validate_physics(self, params: ParamSet) -> Tensor:
        """
        Dynamic physics constraints.
        Iterate over all layers and check roughness < 0.5 * thickness.
        """
        penalty = torch.tensor(0.0, device=self.device)

        for name in params.layer_names:
            d = params[f"{name}.thickness"]
            sigma = params[f"{name}.roughness"]

            # ReLU(sigma - 0.5*d)
            curr_penalty = torch.mean(torch.relu(sigma - 0.5 * d))
            penalty = penalty + curr_penalty

        return penalty

    def simulate(self, params: ParamSet) -> Tensor:
        """
        ParamSet -> Reflectivity
        """
        # Assemble big tensors
        thickness, roughness, sld_full = params.assemble_structure()
        batch_size = thickness.size(0)

        # Q Grid
        q_batch = self.q_grid.unsqueeze(0).expand(batch_size, -1).to(self.device)

        r_sim = instrument_reflectivity(
            q_batch,
            thickness,
            roughness,
            sld_full,
            params.L,
            params.beam_width,
            params.dq if "dq" in params._params else None,
            params.tth_offset if "tth_offset" in params._params else None,
            self.simulator,
            self.config.instrument.wavelength,
        )
        # Normalise the physics curve the way the generator normalises the data, so the
        # i0 label of 1.0 (and the background shifted by the same scale) describe it.
        r_sim, _ = apply_normalization_tensor(r_sim, self.q_grid, self.config)

        # I0 / Bkg
        i0 = params.i0
        bkg = params.bkg

        r_final = (i0 * r_sim) + torch.pow(10.0, bkg)

        return torch.clamp(r_final, min=1e-15, max=2.0)

    def forward_with_params(self, params: ParamSet) -> tuple[ParamSet, Tensor, Tensor]:
        params = params.to(self.device)
        r_final = self.simulate(params)
        penalty = self.validate_physics(params)
        return params, r_final, penalty

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        pass

    def forward(
        self, r_obs_log: Tensor, fixed_values: ParamSet | None = None
    ) -> tuple[ParamSet, Tensor, any]:
        if torch.isnan(r_obs_log).any():
            r_obs_log = torch.nan_to_num(r_obs_log, nan=-10.0)

        raw_output = self.encode(r_obs_log)

        # Pass fixed_values if needed (currently implementation ignores it but keeps signature)
        params = self.unnormalize(torch.sigmoid(raw_output), fixed_values)

        r_final = self.simulate(params)
        penalty = self.validate_physics(params)

        return params, r_final, penalty


class PatchEmbed1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 128,
        patch_size: int = 16,
        stride: int | None = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size

        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=self.stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = x.transpose(1, 2)
        return self.norm(x)
