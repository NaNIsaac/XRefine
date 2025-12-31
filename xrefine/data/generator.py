
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
        instr = config['instrument']
    else:
        # Fallback to dict keys if instrument not provided
        instr = Instrument(
            wavelength=config['wavelength'],
            I0=config['intensity_counts'],
            Ibkg=config['background_signal'],
            res=config['dq'],
            sample_len=config['sample_len'],
            beam_width=config['beam_width']
        )

    n_layers = config['n_layers']
    
    # 1. Random Parameters
    t_min = config['thickness_min']
    t_max = config['thickness_max']
    thickness = torch.rand(batch_size, n_layers, device=device) * (t_max - t_min) + t_min
    
    r_min = config['roughness_min']
    r_max = config['roughness_max']
    # Roughness: [Amb/L1, ..., LN/Sub] -> n_layers + 1
    roughness = torch.rand(batch_size, n_layers + 1, device=device) * (r_max - r_min) + r_min
    
    sld_min = config['sld_min']
    sld_max = config['sld_max']
    
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
    
    # 3. Post-processing
    # Footprint
    theta = 2 * torch.asin(q_grid * instr.wavelength / (4 * np.pi))
    footprint_factor = torch.clamp( (instr.sample_len * torch.sin(theta)) / instr.beam_width, max=1.0)
    
    if config['apply_footprint']:
         r_model = r_model * footprint_factor
         
    # Poisson Noise
    expected_counts = instr.I0 * (r_model + instr.Ibkg)
    noisy_counts = torch.poisson(expected_counts)
    
    r_noisy = noisy_counts / instr.I0
    r_noisy = torch.clamp(r_noisy, min=1e-10)
    
    return r_noisy, (thickness, roughness, sld)

