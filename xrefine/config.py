import torch

class Config:
    Q_MIN = 0.01
    Q_MAX = 0.5
    Q_LEN = 256
    Q_GRID = torch.linspace(Q_MIN, Q_MAX, Q_LEN)

    THICK_RANGE = (10.0, 1000.0)
    ROUGH_RANGE = (0.5, 15.0)
    SLD_RANGE = (5.0, 70.0)
    
    INSTRUMENT = {
        'wavelength': 1.5406,
        'beam_width': 0.1,      # mm
        'sample_len': 20.0,     # mm
        'dq': 0.002,            # Resolution
        'I0': 1e7,              # Max Intensity
        'Ibkg': 1e-7            # Background level
    }

    BATCH_SIZE = 32
    LR = 2e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
