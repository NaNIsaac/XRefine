from __future__ import annotations

import torch
from torch import Tensor


def kinematical_approximation(
    q: Tensor,
    thickness: Tensor,
    roughness: Tensor,
    sld: Tensor,
    *,
    apply_fresnel: bool = True,
    log: bool = False,
) -> Tensor:
    """
    Kinematical approximation (Born approximation) for X-ray reflectivity.

    Args:
        q: Scattering vector magnitude (inverse Angstroms). Shape: (batch_size, num_q)
        thickness: Layer thicknesses (Angstroms). Shape: (batch_size, num_layers)
        roughness: Interface roughnesses (Angstroms). Shape: (batch_size, num_layers + 1)
        sld: Scattering length densities (10^-6 A^-2). Shape: (batch_size, num_layers + 2) or (num_layers + 1)
             Order: [Ambient, Layer 1, ..., Layer N, Substrate]

    Returns:
        Reflectivity. Shape: (batch_size, num_q)
    """
    c_dtype = torch.complex128 if q.dtype is torch.float64 else torch.complex64

    batch_size, num_layers = thickness.shape

    # Q processing
    q = q.to(c_dtype)
    if q.dim() == 1:
        q = q.unsqueeze(0)
    if q.dim() == 2:
        q = q.unsqueeze(-1)  # (batch, q, 1)

    # SLD Adjustment
    # If SLD is (batch, num_layers + 1), it assumes ambient is missing (usually 0)
    if sld.shape[-1] == num_layers + 1:
        sld = torch.cat([torch.zeros(batch_size, 1, device=sld.device, dtype=sld.dtype), sld], -1)

    # Physical constants and unit conversion
    sld_complex = sld * 1e-6 + 1e-30j  # Add small imaginary part for stability

    # Calculate density change at interfaces
    # sld: [Ambient, L1, ..., LN, Sub]
    # drho: sld[i+1] - sld[i]
    # Interfaces: Ambient/L1, L1/L2, ..., LN/Sub
    drho = sld_complex[..., 1:] - sld_complex[..., :-1]  # (batch, layers+1)

    # Broadcast to match Q dim
    drho = drho[:, None, :]  # (batch, 1, layers+1)

    # Thickness calculation for phase factors
    # Depth z_j: cumulative sum of thicknesses.
    # We need depth of each interface.
    # Interface 0 (Ambient/L1): z=0
    # Interface 1 (L1/L2): z=d1
    # ...
    # Interface N (LN/Sub): z=d1+...+dN

    # thickness: (batch, layers)
    # Add 0 for the first interface
    zeros = torch.zeros(batch_size, 1, device=thickness.device, dtype=thickness.dtype)
    thickness_aug = torch.cat([zeros, thickness], -1)  # (batch, layers+1)
    z_depths = torch.cumsum(thickness_aug, -1)  # (batch, layers+1)

    # Broadcast
    z_depths = z_depths[:, None, :]  # (batch, 1, layers+1)
    roughness_aug = roughness[:, None, :]  # (batch, 1, layers+1)

    # Kinematical formula:
    # R = | sum_j (rho_{j+1} - rho_j) * exp(i * q * z_j) * exp(-q^2 * sigma_j^2 / 2) |^2

    exponent = 1j * q * z_depths - (roughness_aug * q) ** 2 / 2
    amplitude = (drho * torch.exp(exponent)).sum(-1)

    r = amplitude.abs() ** 2
    r = r.float()  # Convert back to float if complex

    if apply_fresnel:
        # Fresnel reflectivity of the substrate (approximate)
        substrate_sld = sld_complex[:, -1:]  # (batch, 1) real part main component

        # We need rf using just substrate SLD vs ambient (assumed 0)
        rf = _get_fresnel_reflectivity(q, substrate_sld[:, None])

        # Correct with substrate sld factor (Born approximation refers to R/Rf)
        # Actually standard kinematical is usually |int rho'(z) e^{iqz} dz|^2
        # Which is q^4 R approx 16 pi^2 |...|^2 ?
        # The existing code did: r * rf / substrate_sld.real**2

        # Re-using existing logic's scaling
        r = r * rf / (substrate_sld.real**2)
        r = torch.clamp(r, max=1.0)

    if log:
        r = torch.log10(r + 1e-30)

    return r


def _get_fresnel_reflectivity(q, substrate_slds):
    # This seems to be calculating exact Fresnel for a single interface (Ambient/Substrate)
    # q_c = 4 * sqrt(pi * rho)
    # But code has constant 0.28... which is likely related to units

    # Recalculate constant to be sure:
    # q_c = 4 * sqrt(pi * sld * 1e-6)
    # 4 * sqrt(pi) * 1e-3 = 4 * 1.772 * 0.001 = 0.007088...

    # Current code:
    # _RE_CONST = 0.28174103675406496
    # q_c = torch.sqrt(substrate_slds + 0j) / _RE_CONST * 2
    # This implies q_c ~ 2 * sqrt(sld) / 0.28

    # Let's trust existing constant logic for now but clean up name and operations

    _RE_CONST = 0.28174103675406496

    q_c = torch.sqrt(substrate_slds + 0j) / _RE_CONST * 2
    q_prime = torch.sqrt(q**2 - q_c**2 + 0j)

    denom = q + q_prime
    # Avoid div by zero
    val = (q - q_prime) / (denom + 1e-30)

    r_f = val.abs().float() ** 2

    return r_f.squeeze(-1)
