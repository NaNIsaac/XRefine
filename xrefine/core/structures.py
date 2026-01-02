from dataclasses import dataclass, field
from typing import Any, List, TypedDict
from io import StringIO

import torch
from torch import Tensor
import numpy as np

RE_ANGSTROM = 2.81794032e-5

@dataclass
class Instrument:
    wavelength: float = 1.54
    I0: float = 1.0
    Ibkg: float = 1e-10
    res: float = 0.0001
    sample_len: float = 10.0 # mm
    beam_width: float = 0.2  # mm

@dataclass
class Layer:
    thickness: float  # [Å]
    roughness: float  # [Å]
    sld: complex  # [10^-6 Å^-2]
    f: complex = field(default=complex(1, 0), repr=False)

    @property
    def density(self) -> float:
        """
        Unit: [atoms / Å^3]
        Formula:
            SLD = r_e * rho * f
            rho = SLD / (r_e * f)
        """
        if self.f.real == 0:
            return 0.0

        # SLD is usually given in 10^-6 A^-2
        # We use real part for density calculation
        real_sld_value = self.sld.real * 1e-6 
        
        return real_sld_value / (RE_ANGSTROM * self.f.real)

# Common Layers
AIR = Layer(thickness=0, roughness=0, sld=complex(0, 0))
# Example substrate (Si) - values can be adjusted
SI_SUBSTRATE = Layer(thickness=0, roughness=2.0, sld=complex(20.1, 0.46))

@dataclass
class Sample:
    layers: List[Layer] = field(default_factory=list)
    Ambient: Layer = field(default_factory=lambda: Layer(thickness=0, roughness=0, sld=complex(0, 0)))
    Substrate: Layer = field(default_factory=lambda: Layer(thickness=0, roughness=2.0, sld=complex(20.1, 0.46)))

    def to_tensor(self, device: torch.device, batch_size: int = 1) -> tuple[Tensor, Tensor, Tensor]:
        """
        Convert sample parameters to tensors for simulation.
        
        Args:
            device: Torch device.
            batch_size: Batch size to repeat parameters.
            
        Returns:
            (thickness, roughness, sld)
            thickness: (batch, n_layers)
            roughness: (batch, n_layers + 1)
            sld: (batch, n_layers + 2)
        """
        # Thickness: [L1, ..., LN]
        tk_vals = [l.thickness for l in self.layers]
        thickness = torch.tensor([tk_vals], device=device, dtype=torch.float64)
        
        # Roughness: [Amb/L1, L1/L2, ..., LN/Sub]
        # Mapping: L1.roughness -> Amb/L1 (r[0])
        # ... LN.roughness -> LN-1/LN (r[N-1])
        # Sub.roughness -> LN/Sub (r[N])
        r_vals = [l.roughness for l in self.layers] + [self.Substrate.roughness]
        roughness = torch.tensor([r_vals], device=device, dtype=torch.float64)

        # SLD: [Ambient, L1, ..., LN, Sub]
        sld_vals = [self.Ambient.sld] + [l.sld for l in self.layers] + [self.Substrate.sld]
        # Convert complex to whatever simulator expects? 
        # Usually simulator handles complex if dtyped.
        sld = torch.tensor([sld_vals], device=device, dtype=torch.complex128)
        
        if batch_size > 1:
            thickness = thickness.repeat(batch_size, 1)
            roughness = roughness.repeat(batch_size, 1)
            sld = sld.repeat(batch_size, 1)
            
        return thickness, roughness, sld

    @classmethod
    def from_tensor(cls, thickness: Tensor, roughness: Tensor, sld: Tensor) -> 'Sample':
        """
        Create a Sample from single tensors (1D). 
        Used for reconstructing objects from prediction or optimization results.
        params must be 1D or scalar tensors.
        """
        # Ensure CPU and python primitives
        if thickness.dim() > 1:
             raise ValueError("from_tensor expects 1D tensors (single sample).")
             
        t = thickness.detach().cpu().numpy()
        r = roughness.detach().cpu().numpy()
        s = sld.detach().cpu().numpy()
        
        n_layers = len(t)
        
        # sld: [Amb, L1...LN, Sub] -> size N+2
        # r: [L1_r, ..., LN_r, Sub_r] -> size N+1 (assuming L1_r is top interface)
        
        # Ambient
        amb_sld = complex(s[0]) if np.iscomplexobj(s) else complex(s[0], 0)
        ambient = Layer(thickness=0, roughness=0, sld=amb_sld)
        
        layers = []
        for i in range(n_layers):
            l_thick = float(t[i])
            l_rough = float(r[i]) # Interface above
            l_sld_val = s[i+1]
            l_sld = complex(l_sld_val) if np.iscomplexobj(s) else complex(l_sld_val, 0)
            
            layers.append(Layer(thickness=l_thick, roughness=l_rough, sld=l_sld))
            
        # Substrate
        sub_sld_val = s[-1]
        sub_sld = complex(sub_sld_val) if np.iscomplexobj(s) else complex(sub_sld_val, 0)
        sub_rough = float(r[-1])
        substrate = Layer(thickness=0, roughness=sub_rough, sld=sub_sld)
        
        return cls(layers=layers, Ambient=ambient, Substrate=substrate)

@dataclass
class Experiment:
    q: np.ndarray
    refl: np.ndarray
    instrument: Instrument
    sample: Sample

@dataclass
class FitParam:
    value: float
    min: float
    max: float
    active: bool = True

@dataclass
class FitTarget:
    obj: Any
    attr_name: str
    spec: FitParam


class ParamSet[T: (Tensor, float)](TypedDict):
    """
    XRR 분석에 필요한 모든 가변 변수 세트
    """
    # 1. Main Film
    d_f: T    # (B, N)
    sig_f: T  # (B, N)
    rho_f: T  # (B, N)

    # 2. SiO2 Layer
    d_ox: T   # (B, 1)
    sig_ox: T # (B, 1)
    rho_ox: T # (B, 1)

    # 3. Instrumental Factors
    L: T      # (B, 1) - Sample Length
    i0: T     # (B, 1) - Intensity Scale
    bkg: T    # (B, 1) - Background Log level
    beam_width: T  # (B, 1) - Beam Width


def format_params(params: ParamSet, title: str = "Parameters") -> str:
    """
    match-case를 사용하여 ParamSet을 이쁜 문자열로 변환합니다.
    """
    output = StringIO()
    output.write(f"[{title}]\n")
    
    for k, v in params.items():
        match v:
            case torch.Tensor() if v.numel() == 1:
                output.write(f"  {k:12}: {v.item():.4f}\n")
            case torch.Tensor():
                output.write(f"  {k:12}: {v.detach().cpu().numpy().flatten()[:5]:.4f} (batch)\n")
            case float():
                output.write(f"  {k:12}: {v:.4f}\n")
            case _:
                output.write(f"  {k:12}: {str(v)}\n")
                
    return output.getvalue()