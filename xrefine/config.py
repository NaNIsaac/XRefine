import torch

from physics_utils import tth2q

# Configuration Dictionary
CONFIG = {
    'q_min': tth2q(0.05),
    'q_max': tth2q(15.0),
    'q_len': 2000,

    # Experiment Settings
    'experiment_name': 'test',
    'save_root': 'D:\03_Resources\Data\XRR_AI\PINN',

    # Ranges
    'thick_range': (10.0, 1000.0),
    'rough_range': (0.5, 15.0),
    'sld_range': (5.0, 70.0),

    'instrument': {
        'wavelength': 1.5406,
        'beam_width': 0.1,      # mm
        'sample_len': 20.0,     # mm
        'res': 0.002,           # dq resolution
        'I0': 1e7,              # Max Intensity
        'Ibkg': 1e-7            # Background level
    },

    'batch_size': 32,
    'lr': 2e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'apply_footprint': True,
    
    # Model Settings
    'model_type': 'fourier', # 'pinn' or 'fourier'
    'n_layers': 1,
    'fourier_scale': 10.0,
    'hidden_dim': 256,
    
    # Augmentation Settings
    'augmentations': {
        'intensity_noise_scale': 0.2,
        'bg_range': (-8, -5),
        'res_sigma_range': (0.0001, 0.005),
        'q_shift_sigma': 0.002
    }
}

# Pre-compute Q_GRID
CONFIG['q_grid'] = torch.linspace(CONFIG['q_min'], CONFIG['q_max'], CONFIG['q_len'])
