from __future__ import annotations

import math

import torch
from torch import Tensor


def abeles(
    q: Tensor,
    thickness: Tensor,
    roughness: Tensor,
    sld: Tensor,
) -> Tensor:
    """
    Abeles matrix method for calculating reflectivity.
    
    Args:
        q: Scattering vector magnitude (inverse Angstroms). Shape: (batch_size, num_q)
        thickness: Layer thicknesses (Angstroms). Shape: (batch_size, num_layers)
        roughness: Interface roughnesses (Angstroms). Shape: (batch_size, num_layers + 1)
        sld: Scattering length densities (10^-6 A^-2). Shape: (batch_size, num_layers + 2)
             Order: [Ambient, Layer 1, ..., Layer N, Substrate]
             
    Returns:
        Reflectivity. Shape: (batch_size, num_q)
    """
    c_dtype = torch.complex128 if q.dtype is torch.float64 else torch.complex64

    batch_size, num_layers = thickness.shape

    # SLD handling: expected input is [Ambient, L1, ..., LN, Substrate]
    # If input is [L1, ..., LN, Substrate], prepend 0 (Ambient).
    if sld.shape[-1] == num_layers + 1:
        sld = torch.cat([torch.zeros(batch_size, 1, device=sld.device, dtype=sld.dtype), sld], -1)
    
    if sld.shape[-1] != num_layers + 2:
        raise ValueError(
            f"Number of SLD values ({sld.shape[-1]}) does not equal to num_layers + 2 (ambient + layers + substrate)."
        )

    # Add dummy thickness for ambient (0)
    # thickness shape: (batch, layers) -> (batch, layers + 1)
    thickness_aug = torch.cat([torch.zeros(batch_size, 1, device=thickness.device, dtype=thickness.dtype), thickness], -1)
    
    # Expand dims for broadcasting: (batch, 1, ...)
    sld_aug = sld[:, None]  # (batch, 1, layers+2)
    thickness_aug = thickness_aug[:, None] # (batch, 1, layers+1)
    roughness_aug = roughness[:, None] ** 2 # (batch, 1, layers+1)

    # Convert SLD to appropriate complex unit
    # (batch, 1, layers+2) - (batch, 1, 1) broadcast
    sld_complex = (sld_aug - sld_aug[..., :1]) * 1e-6 + 1e-36j

    # k_z0 calculation
    k_z0 = (q / 2).to(c_dtype) # (batch, q)

    # Ensure k_z0 is (batch, q, 1)
    if k_z0.dim() == 1:
        k_z0 = k_z0.unsqueeze(0)
    if k_z0.dim() == 2:
        k_z0 = k_z0.unsqueeze(-1)

    # k_n calculation
    # k_z0: (batch, q, 1), sld_complex: (batch, 1, layers+2)
    # k_n: (batch, q, layers+2)
    k_n = torch.sqrt(k_z0**2 - 4 * math.pi * sld_complex)

    k_n_curr = k_n[..., :-1] # (batch, q, layers+1) (Ambient to Layer N)
    k_n_next = k_n[..., 1:]  # (batch, q, layers+1) (Layer 1 to Substrate)

    # Beta calculation (phase shift)
    # thickness_aug: (batch, 1, layers+1)
    # k_n_curr: (batch, q, layers+1)
    beta = 1j * thickness_aug * k_n_curr

    exp_beta = torch.exp(beta)
    exp_m_beta = torch.exp(-beta)

    # Fresnel coefficient with roughness
    # roughness_aug: (batch, 1, layers+1)
    factor = (k_n_curr - k_n_next) / (k_n_curr + k_n_next)
    rn = factor * torch.exp(-2 * k_n_curr * k_n_next * roughness_aug)

    row1 = torch.stack([exp_beta, rn * exp_beta], -1)
    row2 = torch.stack([rn * exp_m_beta, exp_m_beta], -1)
    c_matrices = torch.stack([row1, row2], -2) # (batch, q, layers+1, 2, 2)

    # To avoid list overhead:
    m_accum = c_matrices[..., 0, :, :]
    for i in range(1, c_matrices.shape[-3]):
        m_accum = torch.matmul(m_accum, c_matrices[..., i, :, :])

    # Reflectivity calculation
    # r = M[1, 0] / M[0, 0]
    r = (m_accum[..., 1, 0] / m_accum[..., 0, 0]).abs() ** 2
    
    r = torch.clamp(r, max=1.0)

    return r
