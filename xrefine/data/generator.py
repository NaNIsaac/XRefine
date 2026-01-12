import torch
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader
from typing import Optional, Tuple

from xrefine.core.abeles import AbelesMatrix
from xrefine.config import XRefineConfig
from xrefine.core.structures import ParamSet
from xrefine.data.augmentations import XRRAugmentor

def generate_synthetic_batch(
    batch_size: int, 
    q_grid: Tensor, 
    config: XRefineConfig, 
    simulator: Optional[AbelesMatrix] = None, 
    augmenter: Optional[XRRAugmentor] = None
) -> Tuple[Tensor, ParamSet]:
    """
    Generate a synthetic batch of XRR data using dynamic layer configuration.
    Returns: (Reflectivity [Linear], ParamSet)
    """
    device = config.device
    q_grid = q_grid.to(device)
    
    if simulator is None:
        simulator = AbelesMatrix(device=device)
    
    # 1. Parameter Sampling (Dynamic)
    # This replaces the manual field iteration with the class method
    params_gt = ParamSet.from_config(config.sample, batch_size=batch_size, device=device)

    # 2. Structure Assembly
    thickness, roughness, sld_full = params_gt.assemble_structure()

    q_batch = q_grid.view(1, -1).expand(batch_size, -1)
    
    # Offset Augmentation
    # DynamicParamSet supports .tth_offset if it's in the params dict
    if 'tth_offset' in params_gt._params:
        tth_offset = params_gt.tth_offset
        tth_offset_rad = tth_offset * (3.14159 / 180.0) / 2.0
        q_offset = (4 * 3.14159 / config.instrument.wavelength) * torch.sin(tth_offset_rad)
        q_batch = q_batch + q_offset

    # 3. Physics Simulation (Pure)
    refl_pure = simulator(
        q=q_batch,
        thickness=thickness,
        roughness=roughness,
        sld=sld_full,
        sample_length=params_gt.L,
        beam_width=params_gt.beam_width,
        wavelength=config.instrument.wavelength
    )

    # 4. Augmentation & Post-processing
    if augmenter is not None:
        # Augmenter expects Linear Scale input
        r_final = augmenter(q_batch, refl_pure, params_gt)
    else:
        # Manual Augmentation fallback
        r_final = refl_pure * params_gt.i0 + torch.pow(10.0, params_gt.bkg)
        # Add Noise
        noise = torch.randn_like(r_final) * config.augmentations.intensity_noise_scale * r_final
        r_final = r_final + noise

    # Return Linear Scale
    r_final = torch.clamp(r_final, min=1e-12, max=2.0)

    return r_final, params_gt

class XRRStreamingDataset(IterableDataset):
    def __init__(self, config: XRefineConfig, q_grid: torch.Tensor, steps_per_epoch: int):
        super().__init__()
        self.config = config
        self.q_grid = q_grid
        self.steps_per_epoch = steps_per_epoch
        try:
            self.augmenter = XRRAugmentor(config).to(config.device)
        except:
            self.augmenter = None

    def __iter__(self):
        # Worker Loop
        for _ in range(self.steps_per_epoch):
            r_noisy, params_gt = generate_synthetic_batch(
                batch_size=self.config.train.batch_size,
                q_grid=self.q_grid,
                config=self.config,
                augmenter=self.augmenter
            )
            yield r_noisy, params_gt

def get_xrr_dataloader(config: XRefineConfig, q_grid: Tensor) -> DataLoader:
    dataset = XRRStreamingDataset(
        config, q_grid, steps_per_epoch=config.train.steps_per_epoch
    )
    return DataLoader(dataset, batch_size=None, num_workers=0)
