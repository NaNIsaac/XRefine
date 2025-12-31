from typing import Literal, Tuple
from pydantic import BaseModel, Field
import torch
import numpy as np

# Utility for Q conversion
def tth2q(tth: float, wavelength: float = 1.5406) -> float:
    rad = np.deg2rad(0.5 * tth)
    return float((4 * np.pi / wavelength) * np.sin(rad))

class InstrumentConfig(BaseModel):
    wavelength: float = 1.5406
    beam_width: float = 0.1      # mm
    sample_len: float = 20.0     # mm
    res: float = 0.002           # dq resolution
    I0: float = 1e7              # Max Intensity
    Ibkg: float = 1e-7           # Background level

class AugmentationConfig(BaseModel):
    intensity_noise_scale: float = 0.2
    bg_range: Tuple[float, float] = (-8.0, -5.0)
    res_sigma_range: Tuple[float, float] = (0.0001, 0.005)
    q_shift_sigma: float = 0.002
    beam_profile: Literal['gaussian', 'square'] = 'gaussian'

class XRefineConfig(BaseModel):
    # Experiment
    experiment_name: str = 'default_experiment'
    save_root: str = r'experiments'

    # Physics / Data
    q_min_tth: float = 0.05
    q_max_tth: float = 15.0
    q_len: int = 2000

    # Ranges
    thick_range: Tuple[float, float] = (10.0, 1000.0)
    rough_range: Tuple[float, float] = (0.5, 15.0)
    sld_range: Tuple[float, float] = (5.0, 70.0)

    # Sub-configs
    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
    augmentations: AugmentationConfig = Field(default_factory=AugmentationConfig)

    # Training / Model
    batch_size: int = 32
    lr: float = 2e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs: int = 100 # For main loop reference

    # Model Architecture
    model_type: Literal['pinn', 'fourier'] = 'fourier'
    n_layers: int = 1
    fourier_scale: float = 10.0
    hidden_dim: int = 256

    # Runtime Computed Properties (ignored by Pydantic serialization usually, or use property)
    @property
    def q_grid(self) -> torch.Tensor:
        q_min = tth2q(self.q_min_tth, self.instrument.wavelength)
        q_max = tth2q(self.q_max_tth, self.instrument.wavelength)
        return torch.linspace(q_min, q_max, self.q_len)

    @property
    def q_min(self) -> float:
        return tth2q(self.q_min_tth, self.instrument.wavelength)

    @property
    def q_max(self) -> float:
        return tth2q(self.q_max_tth, self.instrument.wavelength)

# Global Default Instance (for backward compatibility / smooth refactor)
CONFIG = XRefineConfig(experiment_name="test_pydantic")
