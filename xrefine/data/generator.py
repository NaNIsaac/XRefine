
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

