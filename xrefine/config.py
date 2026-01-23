import torch
import yaml
from pathlib import Path
from typing import Literal, Self, Optional, List, Annotated
from pydantic import BaseModel, Field, ConfigDict, field_validator, field_serializer, PrivateAttr

from xrefine.physics_utils import tth2q

# Helper Type for Ranges [min, max]
Range = Annotated[List[float], Field(min_length=2, max_length=2)]

# ==============================================================================
# 1. Physical Structure Configuration (Dynamic Layers)
# ==============================================================================

class MaterialConfig(BaseModel):
    """Configuration for a single material property (SLD, Roughness)."""
    sld: Range
    sld_imag: Range 
    roughness: Range

class LayerConfig(BaseModel):
    """Configuration for a single layer in the stack."""
    name: str
    thickness: Range
    roughness: Range
    sld: Range
    sld_imag: Range 
    repeats: int = Field(default=1, ge=1, description="Number of times this layer repeats")

class SubstrateConfig(BaseModel):
    name: str = "Substrate"
    roughness: Range
    sld: Range
    sld_imag: Range 

class AmbientConfig(BaseModel):
    name: str = "Ambient"
    sld: float = 0.0
    sld_imag: Range = Field(default=[0.0, 0.0]) 

class SampleConfig(BaseModel):
    """
    Complete sample definition replacing the old flat PhysicalRangeConfig.
    Now includes instrument variables that vary per sample.
    """
    ambient: AmbientConfig = Field(default_factory=AmbientConfig)
    substrate: SubstrateConfig
    layers: List[LayerConfig]
    
    # [Synced with YAML] Global/Instrumental parameters (Ranges)
    L: Range = Field(default=[5.0, 30.0], description="Sample Length [mm]")
    beam_width: Range = Field(default=[0.01, 0.5], description="Beam Width [mm]")
    bkg: Range = Field(default=[-10.0, -5.0], description="Background Log10")
    i0: Range = Field(default=[0.8, 1.2], description="Intensity Scale")
    dq: Range = Field(default=[0.0005, 0.02], description="Resolution [A^-1]")
    tth_offset: Range = Field(default=[-0.05, 0.05], description="TwoTheta Offset [deg]")

# ==============================================================================
# 2. Existing Configs (Refined)
# ==============================================================================

class InstrumentConfig(BaseModel):
    """
    Fixed Hardware Constants only.
    Variable parameters (beam_width, res, I0) moved to SampleConfig.
    """
    wavelength: float = 1.5406

class AugmentationConfig(BaseModel):
    intensity_noise_scale: float = 0.02
    bg_range: Range = [-10.0, -6.0]
    res_sigma_range: Range = [0.0001, 0.005]
    q_shift_sigma: float = 0.001
    beam_profile: Literal['gaussian', 'square'] = 'gaussian'
    measure_step_range: Range = [0.001, 0.1]

class ModelArchConfig(BaseModel):
    model_type: Literal['fourier_conv', 'fourier_vit', 'latent_trans'] = 'fourier_conv'
    # [Synced with YAML] Optimized Architecture
    hidden_dim: int = 64
    encoder_depth: int = 6
    
    # CNN specific
    kernel_size: int = 15
    dilation: int = 2
    fourier_scale: float = 40.0
    mapping_size: int = 64
    
    # Transformer/ViT
    n_heads: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = 'leaky_relu'
    
    # ViT specific
    patch_size: int = 16

class WeightConfig(BaseModel):
    global_weights: dict[str, float] = Field(
        default={
            "L": 0.1, "i0": 10.0, "bkg": 1.0, "beam_width": 0.1,
            "dq": 1.0, "tth_offset": 1.0
        }
    )
    layer_weights: dict[str, float] = Field(default_factory=dict)
    specific_layer_weights: dict[str, dict[str, float]] = Field(default_factory=dict)

class TrainingConfig(BaseModel):
    batch_size: int = 512
    lr: float = 0.002
    weight_decay: float = 1e-4
    epochs: int = 300
    steps_per_epoch: int = 20
    warmup_steps: int = 200
    early_stop_patience: int = 50
    
    sup_w_start: float = 10000.0
    sup_w_end: float = 100.0
    phys_penalty_w: float = 1.0
    
    # [Synced with YAML] Updated Loss Weights
    w_linear: float = 20.0
    w_gradient: float = 50.0
    w_log_base: float = 50.0
    
    weights: WeightConfig = Field(default_factory=WeightConfig)


class DataConfig(BaseModel):
    """
    Configuration for data preprocessing.
    Normalization strategy selection:
    - "plateau": Low-angle top-mean = 1.0 (robust, recommended)
    - "max": Global max = 1.0 (simple)
    - "anchor": Specific angle (0.2 deg) = 1.0 (physical anchor)
    - "none": No normalization (raw data - requires wide I0 range)
    """
    normalization_mode: Literal["plateau", "max", "anchor", "none"] = "plateau"
    
    # Anchor mode: target 2theta angle
    anchor_tth: float = 0.2
    
    # Plateau mode: Q upper limit for searching
    plateau_q_limit: float = 0.08


# ==============================================================================
# 3. Root Configuration
# ==============================================================================

class XRefineConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='ignore')

    experiment_name: str = 'XRefine_Fourier_Metal_v1'
    save_root: Path = Path("D:\\03_Resources\\Data\\XRR_AI\\xrefine")
    log_dir: Path = Path("logs")
    
    device: torch.device = Field(
        default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    q_min_tth: float = 0.05
    q_max_tth: float = 15.0
    q_len: int = 3000

    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
    sample: SampleConfig = Field(description="Dynamic Layer Configuration")
    augmentations: AugmentationConfig = Field(default_factory=AugmentationConfig)
    train: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelArchConfig = Field(default_factory=ModelArchConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    _q_grid_override: Optional[torch.Tensor] = PrivateAttr(default=None)

    @field_validator('device', mode='before')
    def parse_device(cls, v):
        if isinstance(v, str):
            return torch.device(v)
        return v

    @field_serializer('device')
    def serialize_device(self, device: torch.device, _info):
        return str(device)

    @property
    def q_grid(self) -> torch.Tensor:
        if self._q_grid_override is not None:
            return self._q_grid_override
        q_start = tth2q(self.q_min_tth, self.instrument.wavelength)
        q_end = tth2q(self.q_max_tth, self.instrument.wavelength)
        return torch.linspace(q_start, q_end, self.q_len)

    @q_grid.setter
    def q_grid(self, value: torch.Tensor):
        self._q_grid_override = value

    def save_yaml(self, path: str | Path):
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode='json')
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> Self:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)

# Dummy Default for strict testing
DEFAULT_SAMPLE = SampleConfig(
    substrate=SubstrateConfig(
        roughness=[0.0, 8.0], 
        sld=[18.0, 22.0], 
        sld_imag=[0.0, 1.0]
    ),
    layers=[
        LayerConfig(
            name="Top_Layer", 
            thickness=[0.0, 50.0], 
            roughness=[0.0, 10.0], 
            sld=[0.0, 50.0],
            sld_imag=[0.0, 1.0]
        ),
        LayerConfig(
            name="Target_Film", 
            thickness=[10.0, 1200.0], 
            roughness=[0.0, 30.0], 
            sld=[0.0, 150.0],
            sld_imag=[0.0, 15.0]
        )
    ]
)

# Load CONFIG from config.yaml if exists, otherwise use default
_config_path = Path(__file__).parent.parent / "config.yaml"
if _config_path.exists():
    CONFIG = XRefineConfig.load_yaml(_config_path)
else:
    CONFIG = XRefineConfig(sample=DEFAULT_SAMPLE)