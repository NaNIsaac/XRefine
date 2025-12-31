import torch
import torch.nn as nn
import numpy as np

class DifferentiableFootprint(nn.Module):
    def __init__(self, beam_width=0.1, wavelength=1.5406):
        super().__init__()
        self.beam_w = beam_width
        self.wavelen = wavelength

    def forward(self, q, r_sim, sample_len):
        theta = torch.asin(torch.clamp(q * self.wavelen / (4 * np.pi), -0.999, 0.999))
        footprint_factor = (sample_len * torch.sin(theta)) / self.beam_w
        footprint_factor = torch.clamp(footprint_factor, max=1.0, min=1e-6)
        
        return r_sim * footprint_factor