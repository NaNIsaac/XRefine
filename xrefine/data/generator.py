
import torch
from torch import Tensor
import numpy as np

from xrefine.core.interface import Simulator
from xrefine.core.structures import Instrument

def generate_synthetic_batch(
    batch_size: int,
    q_grid: Tensor,
    config: dict,
    simulator: Simulator = None
) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
    """
    Generate a synthetic batch of XRR data.
    
    Args:
        config: Dictionary. Can contain 'instrument' key with Instrument object.
    """
    device = q_grid.device
    
    if simulator is None:
        simulator = Simulator(method='abeles', device=device)
    
    # Extract Instrument from config or create default
    if 'instrument' in config:
        instr_data = config['instrument']
        if isinstance(instr_data, Instrument):
            instr = instr_data
        else:
             # Assume dict, map keys from our new config structure
             # Our config has 'res' instead of 'dq', need to map if Instrument expects 'res'
             # Instrument dataclass has 'res'. CONFIG has 'res'.
             instr = Instrument(**instr_data)
    else:
        # Fallback to dict keys if instrument not provided
        # Use strict access as requested, assume keys exist
        instr = Instrument(
            wavelength=config['wavelength'],
            I0=config['I0'],
            Ibkg=config['Ibkg'],
            res=config['res'],
            sample_len=config['sample_len'],
            beam_width=config['beam_width']
        )

    n_layers = config.get('n_layer', 1) # Default to 1 if not specified, or use strict? 
    # Config dict doesn't have n_layers currently? It wasn't in original Config class either.
    # Original code: n_layers = config['n_layers'] -> This would fail if not in Config.
    # Let's check where n_layers comes from. It seems it wasn't in Config class!
    # It must be passed or expected to be added. 
    # For now, let's use .get with default or assume 1. User wants strict access.
    # If I use strict access for n_layers and it's missing, it crashes.
    # I'll default to 1 but with a comment or just add n_layers to CONFIG if needed.
    # In model.py: n_layers=1 default.
    n_layers = 1 # Force 1 for now or check config
    if 'n_layers' in config:
        n_layers = config['n_layers']
    
    # 1. Random Parameters
    t_min, t_max = config['thick_range']
    thickness = torch.rand(batch_size, n_layers, device=device) * (t_max - t_min) + t_min
    
    r_min, r_max = config['rough_range']
    # Roughness: [Amb/L1, ..., LN/Sub] -> n_layers + 1
    roughness = torch.rand(batch_size, n_layers + 1, device=device) * (r_max - r_min) + r_min
    
    sld_min, sld_max = config['sld_range']
    
    layers_sld = torch.rand(batch_size, n_layers, device=device) * (sld_max - sld_min) + sld_min
    sub_sld = torch.rand(batch_size, 1, device=device) * (sld_max - sld_min) + sld_min
    ambient_sld = torch.zeros(batch_size, 1, device=device)
    
    sld = torch.cat([ambient_sld, layers_sld, sub_sld], dim=1)
    
    # 2. Simulation with Instrument
    dq = torch.full((batch_size, 1), instr.res, device=device)
    
    if q_grid.ndim == 1:
        q_grid = q_grid.view(1, -1)
    if q_grid.size(0) == 1:
        q_grid = q_grid.expand(batch_size, -1)
        
    r_model = simulator(q_grid, thickness, roughness, sld, dq=dq)
    
    # 3. Post-processing (Augmentations)
    # Use XRRAugmentations if available, or simple logic if not? 
    # The user wants to use the new augmentations.
    # Instantiate XRRAugmentations on the fly or pass it? 
    # For performance, it might be better to pass an instance, but here we'll instantiate for simplicity or assume config has params.
    
    from xrefine.data.augmentations import XRRAugmentations
    
    # Map config to Augmentation params (assuming config has 'augment' dict or similar)
    # If not present, use defaults or extract from config
    
    aug_params = config.get('augmentations', {})
    
    # Calculate delta_q from q_grid
    if q_grid.ndim > 1:
        q_1d = q_grid[0]
    else:
        q_1d = q_grid
        
    delta_q = (q_1d[1] - q_1d[0]).item() if len(q_1d) > 1 else 0.0005
    
    augmenter = XRRAugmentations(
        delta_q=delta_q,
        # Pass other params from config if they exist
        intensity_noise_scale=aug_params.get('intensity_noise_scale', 0.2),
        bg_range=aug_params.get('bg_range', (-8, -5)),
        res_sigma_range=aug_params.get('res_sigma_range', (0.0001, 0.005)),
        q_shift_sigma=aug_params.get('q_shift_sigma', 0.002),
        do_footprint=config.get('apply_footprint', True)
    )

    # Convert to numpy for augmentation (augmentations.py uses numpy/scipy)
    # This might be a bottleneck if moving back and forth GPU->CPU.
    # The legacy augmentation code is numpy based.
    # We generated r_model on device (GPU). 
    r_np = r_model.detach().cpu().numpy()
    q_np = q_grid.detach().cpu().numpy()
    
    # Process batch
    r_aug_list = []
    
    # Augmentation is per-sample usually, but here q_grid might be shared.
    # XRRAugmentations call takes (q, R).
    # We iterate over batch.
    
    for i in range(batch_size):
        curr_q = q_np[i] if q_np.ndim > 1 else q_np
        curr_r = r_np[i]
        
        _, r_aug = augmenter(curr_q, curr_r)
        r_aug_list.append(r_aug)
        
    r_noisy = torch.tensor(np.array(r_aug_list), dtype=torch.float32, device=device)
    
    # Clamp safety
    r_noisy = torch.clamp(r_noisy, min=1e-10)
    
    return r_noisy, (thickness, roughness, sld)

