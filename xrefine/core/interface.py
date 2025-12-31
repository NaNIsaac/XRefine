from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .abeles import abeles
from .kinematical import kinematical_approximation
from .smearing import abeles_constant_smearing
from .structures import Sample, Instrument


class Simulator(nn.Module):
    """
    Differentiable X-ray Reflectivity Simulator.
    
    Supports both dynamic calculation (standard XRR) and optimization (PINN/fitting).
    """
    def __init__(self, method: str = 'abeles', device: str | torch.device = 'cpu'):
        super().__init__()
        self.method = method
        self.device = torch.device(device)

    def forward(
        self,
        q: Tensor,
        thickness: Tensor,
        roughness: Tensor,
        sld: Tensor,
        dq: Tensor | None = None,
        **kwargs
    ) -> Tensor:
        """
        Calculate reflectivity.
        
        Args:
            q: Scattering vector (inverse Angstroms).
            thickness: Layer thicknesses.
            roughness: Interface roughnesses.
            sld: Scattering length densities.
            dq: Resolution (delta q) for smearing. If None, no smearing is applied.
            
        Returns:
            Reflectivity R(q).
        """
        # Ensure inputs are on correct device
        q = q.to(self.device)
        thickness = thickness.to(self.device)
        roughness = roughness.to(self.device)
        sld = sld.to(self.device)
        
        if self.method == 'abeles':
            if dq is not None:
                if not torch.is_tensor(dq):
                    dq = torch.tensor([[dq]], device=self.device, dtype=q.dtype)
                else:
                    dq = dq.to(self.device)
                return abeles_constant_smearing(q, thickness, roughness, sld, dq=dq, **kwargs)
            else:
                return abeles(q, thickness, roughness, sld)
        elif self.method == 'kinematical':
            return kinematical_approximation(q, thickness, roughness, sld, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def simulate_sample(self, q: Tensor, sample: Sample, instrument: Instrument | None = None) -> Tensor:
        """
        Simulate reflectivity for a Sample object.
        """
        thickness, roughness, sld = sample.to_tensor(device=self.device, batch_size=q.shape[0] if q.dim() > 1 else 1)

        dq = None
        if instrument is not None and instrument.res > 0:
            dq = torch.tensor([[instrument.res]], device=self.device).expand(thickness.shape[0], 1)

        return self(q, thickness, roughness, sld, dq=dq)

    def simulate(self, *args, **kwargs):
        """Alias for forward."""
        return self(*args, **kwargs)
