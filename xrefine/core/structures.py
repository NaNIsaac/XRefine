from dataclasses import dataclass, field, fields
from typing import Any, List
from collections.abc import Iterator

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

@dataclass
class ParamSet:
    d_f: Tensor = field(metadata={'mult': 'n', 'type': 'physics'})
    sig_f: Tensor  = field(metadata={'mult': 'n', 'type': 'physics'})
    rho_f: Tensor  = field(metadata={'mult': 'n', 'type': 'physics'})
    d_ox: Tensor   = field(metadata={'mult': 1, 'type': 'physics'})
    sig_ox: Tensor = field(metadata={'mult': 1, 'type': 'physics'})
    rho_ox: Tensor = field(metadata={'mult': 1, 'type': 'physics'})
    L: Tensor = field(metadata={'mult': 1, 'type': 'instrument'})
    i0: Tensor = field(metadata={'mult': 1, 'type': 'instrument'})
    bkg: Tensor = field(metadata={'mult': 1, 'type': 'instrument'}) 
    beam_width: Tensor = field(metadata={'mult': 1, 'type': 'instrument'})
    dq: Tensor = field(metadata={'mult': 1, 'type': 'instrument'})  # Resolution [Å⁻¹]
    tth_offset: Tensor = field(metadata={'mult': 1, 'type': 'instrument'})  # 2θ offset [°]
    rho_sub: Tensor = field(metadata={'mult': 1, 'type': 'physics'})   # Substrate SLD
    sig_sub: Tensor = field(metadata={'mult': 1, 'type': 'physics'})   # Substrate roughness [Å]

    @classmethod
    def names(cls) -> List[str]:
        """
        인스턴스 생성 없이 필드 이름 목록을 반환합니다.
        외부에서 keys = ['d_f', ...] 하드코딩 대신 ParamSet.names() 사용 가능.
        """
        return [f.name for f in fields(cls)]

    @classmethod
    def physics_names(cls) -> List[str]:
        """물리 파라미터 이름만 반환"""
        return [f.name for f in fields(cls) if f.metadata.get('type') == 'physics']

    @classmethod
    def instrument_names(cls) -> List[str]:
        """기기 파라미터 이름만 반환"""
        return [f.name for f in fields(cls) if f.metadata.get('type') == 'instrument']

    def keys(self) -> Iterator[str]:
        """필드 이름들에 대한 iterator 반환"""
        return (f.name for f in fields(self))

    def values(self) -> Iterator[Tensor]:
        """텐서 값들에 대한 iterator 반환"""
        return (getattr(self, f.name) for f in fields(self))

    def items(self) -> Iterator[tuple[str, Tensor]]:
        """(이름, 텐서) 쌍에 대한 iterator 반환"""
        return ((f.name, getattr(self, f.name)) for f in fields(self))

    def __getitem__(self, key: str) -> Tensor:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        # fields()를 직접 조회하여 효율적으로 체크
        return any(f.name == key for f in fields(self))

    def requires_grad_(self, requires_grad: bool = True) -> "ParamSet":
        """In-place로 모든 텐서의 gradient 추적 설정"""
        for v in self.values():
            if isinstance(v, torch.Tensor):
                v.requires_grad_(requires_grad)
        return self

    def get_optimized_copy(self) -> "ParamSet":
        """
        최적화(Refinement)를 위해 detach 및 clone된 
        새로운 ParamSet 객체를 생성하고 gradient를 활성화합니다.
        """
        return ParamSet(**{
            k: v.detach().clone().requires_grad_(True) 
            for k, v in self.items()
        })

    def assemble_structure(self) -> tuple[Tensor, Tensor, Tensor]:
        """
        물리 시뮬레이션을 위한 텐서 조립
        [FIX] Ambient 층을 명시적으로 추가하여 차원 불일치 해결
        """
        device = self.d_f.device
        batch_size = self.d_f.shape[0]
        
        # 1. Thickness: [Ambient(0), Film, Oxide] -> Size 3
        # Abeles 알고리즘은 0번 레이어를 Ambient로 가정하므로 0.0 추가 필수
        ambient_thick = torch.zeros((batch_size, 1), device=device)
        thickness = torch.cat([ambient_thick, self.d_f, self.d_ox], dim=-1).float()
        
        # 2. Roughness: [Amb/Film, Film/Ox, Ox/Sub] -> Size 3 (Interfaces)
        roughness = torch.cat([self.sig_f, self.sig_ox, self.sig_sub], dim=-1).float()
        
        # 3. SLD: [Ambient, Film, Oxide, Substrate] -> Size 4
        ambient_sld = torch.zeros((batch_size, 1), device=device)
        sld = torch.cat([ambient_sld, self.rho_f, self.rho_ox, self.rho_sub], dim=-1).float()
        
        return thickness, roughness, sld

    @classmethod
    def cat(cls, ps_list: list["ParamSet"]) -> "ParamSet":
        """여러 조각의 ParamSet을 배치 차원(dim=0)으로 합침"""
        merged = {}
        for f in fields(cls):
            tensors = [getattr(ps, f.name) for ps in ps_list]
            merged[f.name] = torch.cat(tensors, dim=0)
        return cls(**merged)

    def clone(self) -> "ParamSet":
        """모든 텐서 필드를 복제하여 새로운 ParamSet 객체 생성"""
        return ParamSet(**{
            f.name: getattr(self, f.name).clone() 
            for f in fields(self)
        })

    def detach(self) -> "ParamSet":
        """그래프에서 분리된(gradient 추적 중단) 복제본 생성"""
        return ParamSet(**{
            f.name: getattr(self, f.name).detach() 
            for f in fields(self)
        })

    def to(self, device):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, torch.Tensor):
                setattr(self, f.name, v.to(device))
        return self

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        items = []
        limit = 4 # 이 값보다 배치가 크면 요약 출력

        for f in fields(self):
            v = getattr(self, f.name)
            match v:
                case torch.Tensor():
                    arr = v.detach().cpu().numpy()
                    match arr.ndim:
                        case 0:
                            val = f"{arr.item():.4f}"
                            if f.name == "bkg":
                                val += f" (10^{arr.item():.2f})"
                        case 1:
                            val = f"[{', '.join(f'{x:.4f}' for x in arr)}]"
                        case 2 if arr.shape[0] == 1:
                            val = f"[{', '.join(f'{x:.4f}' for x in arr[0])}]"
                        case 2 if arr.shape[0] > limit:
                            # [Adaptive Repr] 앞 2개, 뒤 1개만 보여줌
                            head = [f"S{i}: [{', '.join(f'{x:.2f}' for x in row)}]" for i, row in enumerate(arr[:2])]
                            tail = [f"S{arr.shape[0]-1}: [{', '.join(f'{x:.2f}' for x in arr[-1])}]"]
                            val = "\n      " + "\n      ".join(head + ["      ... (omitted)"] + tail)
                        case 2:
                            samples = [f"S{i}: [{', '.join(f'{x:.2f}' for x in row)}]" for i, row in enumerate(arr)]
                            val = "\n      " + "\n      ".join(samples)
                        case _:
                            val = f"Tensor{list(arr.shape)}"
                case _:
                    val = str(v)
            items.append(f"{f.name}={val}")

        return f"{class_name}(\n    " + ",\n    ".join(items) + "\n)"
