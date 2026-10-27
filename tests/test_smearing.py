"""The 2θ zero offset is a shift of 2θ in degrees, not a fixed shift of q (#6); the smeared
path uses the configured wavelength (#18)."""

import numpy as np
import torch

from xrefine.core.abeles import AbelesMatrix
from xrefine.core.smearing import apply_tth_offset, instrument_reflectivity
from xrefine.physics_utils import tth2q

WAVELENGTH = 1.5406
MO_K_ALPHA = 0.7107


def film_curve(wavelength, dq):
    """A 250 Å film on a substrate, with a 10 mm sample in a 0.1 mm beam so the footprint
    term matters below about 1° 2θ."""
    q = torch.linspace(0.005, 0.25, 300).unsqueeze(0)
    return instrument_reflectivity(
        q,
        torch.tensor([[0.0, 250.0]]),
        torch.tensor([[3.0, 3.0]]),
        torch.complex(torch.tensor([[0.0, 40.0, 20.0]]), torch.tensor([[0.0, 1.0, 0.5]])),
        torch.tensor([[10.0]]),
        torch.tensor([[0.1]]),
        dq,
        None,
        AbelesMatrix(device="cpu"),
        wavelength,
    )[0]


def test_offset_shifts_two_theta_by_the_offset_in_degrees():
    tth = np.linspace(0.1, 10.0, 200)
    q = torch.tensor(tth2q(tth, WAVELENGTH)).unsqueeze(0)
    offset = torch.tensor([[0.05], [-0.02]], dtype=q.dtype)
    shifted = apply_tth_offset(q, offset, WAVELENGTH)
    for row, d in zip(shifted, offset[:, 0].tolist(), strict=True):
        np.testing.assert_allclose(row.numpy(), tth2q(tth + d, WAVELENGTH), rtol=1e-6, atol=1e-9)


def test_instrument_reflectivity_applies_the_offset_in_two_theta():
    # The simulator returns its q, so the curve exposes the q the offset produced.
    def simulator(q, **kwargs):
        return q

    q = torch.linspace(0.02, 0.5, 100).unsqueeze(0)
    seen = instrument_reflectivity(
        q, None, None, None, None, None, None, torch.tensor([[0.05]]), simulator, WAVELENGTH
    )
    dq = (seen - q)[0]
    # dq/d(2θ) = (2π/λ) cos θ per radian: 0.071 1/Å per degree at small angles, 7x the old 0.01
    per_degree = 2 * np.pi / WAVELENGTH * np.deg2rad(1.0)
    assert torch.allclose(dq, torch.full_like(dq, 0.05 * per_degree), rtol=1e-2)


def test_smeared_curve_uses_the_configured_wavelength():
    # A resolution far below the q step leaves the curve unchanged, so with the wavelength passed
    # through, the smeared and unsmeared paths must agree, footprint included.
    plain = film_curve(MO_K_ALPHA, None)
    smeared = film_curve(MO_K_ALPHA, torch.tensor([[2e-5]]))
    np.testing.assert_allclose(smeared.numpy(), plain.numpy(), rtol=1e-3, atol=1e-9)


def test_smeared_footprint_changes_with_the_wavelength():
    dq = torch.tensor([[2e-5]])
    low_q = slice(0, 20)
    cu, mo = film_curve(WAVELENGTH, dq)[low_q], film_curve(MO_K_ALPHA, dq)[low_q]
    # At fixed q, a shorter wavelength means a smaller angle, a larger footprint and less signal.
    assert torch.all(mo < cu)
