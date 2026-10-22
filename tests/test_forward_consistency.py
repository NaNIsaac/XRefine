"""The model's reconstruction from the true labels must reproduce the generated curve.

If the generator and BasePINNModel.simulate disagree on the instrument (footprint, 2θ offset,
resolution, normalisation), no network can fit its own training data.
"""

import torch

from xrefine.config import XRefineConfig
from xrefine.core.abeles import AbelesMatrix
from xrefine.data.generator import generate_synthetic_batch
from xrefine.models import get_model


def test_reconstruction_of_true_labels_matches_generated_curve():
    cfg = XRefineConfig.load_yaml("config.yaml")
    d = cfg.model_dump()
    d["device"] = "cpu"
    d["q_len"] = 600
    cfg = XRefineConfig(**d)
    torch.manual_seed(0)
    r, p = generate_synthetic_batch(8, cfg.q_grid, cfg, AbelesMatrix(device="cpu"), augmenter=None)
    p._params["bkg"] = torch.full_like(
        p._params["bkg"], -15.0
    )  # no augmenter: the data has no background
    model = get_model(cfg)
    with torch.no_grad():
        recon = model.simulate(p)
    q = cfg.q_grid
    sel = (q > 0.01) & (q < 0.15)
    dlog = (torch.log10(recon.clamp(min=1e-15)) - torch.log10(r.clamp(min=1e-15)))[:, sel].abs()
    assert dlog.mean() < 0.01, float(dlog.mean())


def test_smearing_of_a_curve_does_not_depend_on_its_batch():
    from xrefine.core.smearing import abeles_constant_smearing

    q = torch.linspace(0.01, 0.3, 400).unsqueeze(0)
    thickness = torch.tensor([[0.0, 250.0]])
    roughness = torch.tensor([[3.0, 3.0]])
    sld = torch.complex(torch.tensor([[0.0, 40.0, 20.0]]), torch.tensor([[0.0, 1.0, 0.5]]))
    one = abeles_constant_smearing(
        q, thickness, roughness, sld, None, None, dq=torch.tensor([[0.01]])
    )
    two = abeles_constant_smearing(
        q.expand(2, -1),
        thickness.expand(2, -1),
        roughness.expand(2, -1),
        sld.expand(2, -1),
        None,
        None,
        dq=torch.tensor([[0.01], [0.0005]]),
    )
    assert torch.allclose(one[0].log10(), two[0].log10(), atol=1e-3)
