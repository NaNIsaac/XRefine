import numpy as np
import torch
import torch.nn as nn


class DifferentiableFootprint(nn.Module):
    """Differentiable footprint layer for the PINN physics loss."""

    def __init__(self, wavelength: float = 1.5406):
        super().__init__()
        self.wavelength = wavelength
        self.register_buffer("pi", torch.tensor(np.pi))

    def forward(
        self,
        q: torch.Tensor,
        r_sim: torch.Tensor,
        sample_len: torch.Tensor,
        beam_width: torch.Tensor,
    ) -> torch.Tensor:
        sin_alpha = q * self.wavelength / (4.0 * self.pi)

        # FWHM = 2 * sqrt(2 * ln2) * sigma_b
        denom_sigma = 2.0 * torch.sqrt(2.0 * torch.log(torch.tensor(2.0, device=q.device)))
        sigma_b = beam_width / denom_sigma

        s_half = sample_len / 2.0

        # ratio of the sample half-length to the projected beam size
        arg = (s_half * sin_alpha) / (
            torch.sqrt(torch.tensor(2.0, device=q.device)) * sigma_b + 1e-9
        )
        footprint_factor = torch.erf(arg)

        return r_sim * footprint_factor
