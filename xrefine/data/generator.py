import torch
from torch import Tensor
import numpy as np

from xrefine.core.interface import Simulator

# Type hint for Config object (forward reference or import if possible, lets use 'Any' or specific type if imported)
# To avoid cycle imports, we can use Duck Typing or just 'object'
from typing import Optional, Any

def generate_synthetic_batch(
    batch_size: int,
    q_grid: Tensor,
    config: Any, # XRefineConfig
    simulator: Optional[Simulator] = None,
    augmenter: Optional[Any] = None # XRRAugmentations
) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
    """
    Generate a synthetic batch of XRR data.
    
    Args:
        config: XRefineConfig object.
        augmenter: Pre-instantiated XRRAugmentations object.
    """
    device = q_grid.device
    
    if simulator is None:
        simulator = Simulator(method='abeles', device=device)
        
    # Extract params from Config Object
    # Config is now Pydantic Model
    n_layers = config.n_layers
    
    # 1. Random Parameters
    t_min, t_max = config.thick_range
    thickness = torch.rand(batch_size, n_layers, device=device) * (t_max - t_min) + t_min
    
    r_min, r_max = config.rough_range
    # Roughness: [Amb/L1, ..., LN/Sub] -> n_layers + 1
    roughness = torch.rand(batch_size, n_layers + 1, device=device) * (r_max - r_min) + r_min
    
    sld_min, sld_max = config.sld_range
    
    layers_sld = torch.rand(batch_size, n_layers, device=device) * (sld_max - sld_min) + sld_min
    sub_sld = torch.rand(batch_size, 1, device=device) * (sld_max - sld_min) + sld_min
    ambient_sld = torch.zeros(batch_size, 1, device=device)
    
    sld = torch.cat([ambient_sld, layers_sld, sub_sld], dim=1)
    
    # 2. Simulation with Instrument
    # Instrument config is in config.instrument (Pydantic model)
    # We need to map it to 'dq'
    dq_val = config.instrument.res
    dq = torch.full((batch_size, 1), dq_val, device=device)
    
    if q_grid.ndim == 1:
        q_grid = q_grid.view(1, -1)
    if q_grid.size(0) == 1:
        q_grid = q_grid.expand(batch_size, -1)
        
    r_model = simulator(q_grid, thickness, roughness, sld, dq=dq)
    
    # 3. Post-processing (Augmentations)
    if augmenter is not None:
        # Convert to numpy for augmentation
        r_np = r_model.detach().cpu().numpy()
        q_np = q_grid.detach().cpu().numpy()
        
        r_aug_list = []
        for i in range(batch_size):
            curr_q = q_np[i] if q_np.ndim > 1 else q_np
            curr_r = r_np[i]
            
            _, r_aug = augmenter(curr_q, curr_r)
            r_aug_list.append(r_aug)
            
        r_noisy = torch.tensor(np.array(r_aug_list), dtype=torch.float32, device=device)
        r_noisy = torch.clamp(r_noisy, min=1e-10)
    else:
        # Fallback if no augmenter provided (should not happen in new pipeline)
        r_noisy = torch.clamp(r_model, min=1e-10)
    
    return r_noisy, (thickness, roughness, sld)

