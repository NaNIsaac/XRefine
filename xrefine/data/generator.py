import torch
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader
from typing import Optional, Tuple
from dataclasses import fields

from xrefine.core.abeles import AbelesMatrix
from xrefine.config import XRefineConfig
from xrefine.core.structures import ParamSet
from xrefine.data.augmentations import XRRAugmentor
from xrefine.logger import logger

def generate_synthetic_batch(
    batch_size: int, 
    q_grid: Tensor, 
    config: XRefineConfig, 
    simulator: Optional[AbelesMatrix] = None, 
    augmenter: Optional[XRRAugmentor] = None
) -> Tuple[Tensor, ParamSet]:
    """
    [Fixed] Linear Scale Reflectivity를 반환합니다.
    (Trainer에서 Log 변환을 수행하므로 여기서는 순수 값을 내보냅니다.)
    """
    device = config.device
    q_grid = q_grid.to(device)
    
    if simulator is None:
        simulator = AbelesMatrix(device=device)
    
    n = getattr(config.model, 'n_layers', 1)
    sampled_values = {}

    # 1. Parameter Sampling
    for fld in fields(ParamSet):
        mult = fld.metadata.get('mult', 1)
        width = n if mult == 'n' else 1
        p_range = getattr(config.ranges, fld.name)
        low, high = p_range
        
        sampled_values[fld.name] = torch.rand(
            (batch_size, width), device=device, dtype=torch.float32
        ) * (high - low) + low

    params_gt = ParamSet(**sampled_values)

    # 2. Structure Assembly
    thickness, roughness, sld_full = params_gt.assemble_structure()

    q_batch = q_grid.view(1, -1).expand(batch_size, -1)
    
    # Offset Augmentation
    if hasattr(params_gt, 'tth_offset'):
        tth_offset_rad = params_gt.tth_offset * (3.14159 / 180.0) / 2.0
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
        # Augmenter는 Linear Scale 입력을 기대함
        r_final = augmenter(q_batch, refl_pure, params_gt)
    else:
        # Manual Augmentation
        r_final = refl_pure * params_gt.i0 + torch.pow(10.0, params_gt.bkg)
        # Add Noise (Linear Scale에서 추가해야 물리적으로 타당함)
        noise = torch.randn_like(r_final) * config.augmentations.intensity_noise_scale * r_final
        r_final = r_final + noise

    # [중요] Log를 취하지 않고 Linear 상태로 반환! 
    # (Trainer에서 Clamp 후 Log를 취할 예정)
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