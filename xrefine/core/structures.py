from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Iterator, Self, Any
import torch
from torch import Tensor
import torch.nn as nn
from xrefine.constants import RE_ANGSTROM
from xrefine.config import SampleConfig  # We need to know the structure to build params

@dataclass
class Instrument:
    """Legacy Instrument class, kept for compatibility with older tests/logic if needed."""
    wavelength: float = 1.54
    I0: float = 1.0
    Ibkg: float = 1e-10
    res: float = 0.0001
    sample_len: float = 10.0 # mm
    beam_width: float = 0.2  # mm

class DynamicParamSet:
    """
    Dynamic container for learnable physical parameters.
    Replaces the hardcoded 'ParamSet' with a dictionary-based approach
    that adapts to any layer structure defined in SampleConfig.
    
    Structure:
    - layers: list of layer names (ordered)
    - params: dict[str, Tensor] -> 'Film.thickness', 'Film.roughness', etc.
    - instrument: dict[str, Tensor] -> 'i0', 'bkg', etc.
    """
    
    # Standard keys for instrument parameters
    INSTRUMENT_KEYS = {'L', 'i0', 'bkg', 'beam_width', 'dq', 'tth_offset'}

    def __init__(self, 
                 params: Dict[str, Tensor], 
                 layer_names: List[str],
                 device: torch.device = None):
        self._params = params
        self.layer_names = layer_names # Order matters for assembly!
        self.device = device or (next(iter(params.values())).device if params else torch.device('cpu'))

    @classmethod
    def from_config(cls, config: SampleConfig, batch_size: int = 1, device: torch.device = None) -> Self:
        """
        Initialize parameters (randomly or centrally) based on ranges in SampleConfig.
        This creates the 'Parameter State' for a batch.
        """
        device = device or torch.device('cpu')
        params = {}
        layer_names = [l.name for l in config.layers]

        def sample_val(rng):
            # Uniform sampling: min + rand * (max - min)
            val = rng[0] + torch.rand(batch_size, 1, device=device) * (rng[1] - rng[0])
            return val

        # 1. Layer Parameters
        for layer in config.layers:
            params[f"{layer.name}.thickness"] = sample_val(layer.thickness)
            params[f"{layer.name}.roughness"] = sample_val(layer.roughness)
            params[f"{layer.name}.sld"] = sample_val(layer.sld)
        
        # 2. Substrate
        # Substrate has roughness (interface with bottom layer) and SLD. Thickness is infinite (0).
        params["Substrate.roughness"] = sample_val(config.substrate.roughness)
        params["Substrate.sld"] = sample_val(config.substrate.sld)
        
        # 3. Ambient
        # Ambient is usually fixed air (sld=0), but we can make it learnable if needed.
        # For now, treat it as fixed constant 0 in assembly, unless specified.
        # We generally DON'T store fixed constants in learnable params dict.

        # 4. Instrument/Global Parameters
        # config.sample has these ranges now
        for key in cls.INSTRUMENT_KEYS:
            if hasattr(config, key):
                rng = getattr(config, key)
                params[key] = sample_val(rng)

        return cls(params, layer_names, device)

    def __getattr__(self, name: str) -> Tensor:
        """Allow dot access: p.i0, p['Film.thickness']"""
        if name in self._params:
            return self._params[name]
        # Fallback for old code: d_f -> Film.thickness? 
        # For now, strict access only.
        raise AttributeError(f"'DynamicParamSet' has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Tensor:
        return self._params[key]

    def __setitem__(self, key: str, value: Tensor):
        self._params[key] = value

    @classmethod
    def cat(cls, ps_list: list[DynamicParamSet]) -> DynamicParamSet:
        """Concatenate a list of ParamSets along the batch dimension."""
        if not ps_list:
            raise ValueError("Empty ParamSet list")
        
        layer_names = ps_list[0].layer_names
        device = ps_list[0].device
        
        merged_params = {}
        keys = ps_list[0]._params.keys()
        
        for key in keys:
            tensors = [ps[key] for ps in ps_list]
            merged_params[key] = torch.cat(tensors, dim=0)
            
        return cls(merged_params, layer_names, device)

    def to(self, device: torch.device) -> Self:
        """Move all parameter tensors to the specified device."""
        self.device = device
        self._params = {k: v.to(device) for k, v in self._params.items()}
        return self

    def detach(self) -> Self:
        return self.__class__(
            {k: v.detach() for k, v in self._params.items()},
            self.layer_names,
            self.device
        )
    
    def clone(self) -> Self:
        return self.__class__(
            {k: v.clone() for k, v in self._params.items()},
            self.layer_names,
            self.device
        )
        
    def requires_grad_(self, mode: bool = True) -> Self:
        for v in self._params.values():
            v.requires_grad_(mode)
        return self

    def assemble_structure(self) -> tuple[Tensor, Tensor, Tensor]:
        """
        Construct the monolithic tensors required by the Abeles engine.
        
        Returns:
            thickness: (B, N_layers + 1) [Ambient, L1, ..., LN]
            roughness: (B, N_layers + 1) [Amb/L1, L1/L2, ..., LN/Sub]
            sld:       (B, N_layers + 2) [Ambient, L1, ..., LN, Sub]
        """
        # Batch size check
        first_key = next(iter(self._params))
        batch_size = self._params[first_key].shape[0]

        # 1. Thickness
        # Ambient (0) + Layers
        # Note: Abeles expects Ambient thickness to be 0 (or dummy).
        t_amb = torch.zeros((batch_size, 1), device=self.device)
        t_layers = [self._params[f"{name}.thickness"] for name in self.layer_names]
        
        thickness = torch.cat([t_amb] + t_layers, dim=1) # (B, 1+N)

        # 2. Roughness
        # Interfaces: Amb/L1, L1/L2, ..., LN/Sub
        # We assume layer.roughness is the roughness of the interface ON TOP of the layer.
        # Wait, usually roughness is associated with an interface.
        # L1.roughness -> Amb/L1
        # L2.roughness -> L1/L2
        # Sub.roughness -> LN/Sub
        
        r_layers = [self._params[f"{name}.roughness"] for name in self.layer_names]
        r_sub = self._params["Substrate.roughness"]
        
        roughness = torch.cat(r_layers + [r_sub], dim=1) # (B, N+1)

        # 3. SLD Assembly (Complex Support)
        # Ambient(Air)는 보통 실수 0.0 (흡수 없음)
        sld_amb = torch.zeros((batch_size, 1), device=self.device, dtype=torch.complex128)
        
        sld_list = []
        
        # Layers + Substrate 순회
        all_layers = self.layer_names + ["Substrate"]
        
        for name in all_layers:
            # Real Part
            rho_real = self._params[f"{name}.sld"]
            
            # Imaginary Part (Absorption) 확인
            imag_key = f"{name}.sld_imag"
            if imag_key in self._params:
                rho_imag = self._params[imag_key]
                # SLD = Real + 1j * Imag
                sld_val = rho_real.double() + 1j * rho_imag.double()
            else:
                # 없으면 실수만 사용 (Complex128 형변환)
                sld_val = rho_real.double() + 0j
            
            sld_list.append(sld_val)

        # Ambient + Layers/Substrate 결합
        # 결과 shape: (B, N+2), dtype=complex128
        sld = torch.cat([sld_amb] + sld_list, dim=1)

        return thickness, roughness, sld
    
    # Backward compatibility shim for 'ParamSet'
    # We alias DynamicParamSet to ParamSet so imports don't break immediately
    # but the content is different.
    
ParamSet = DynamicParamSet