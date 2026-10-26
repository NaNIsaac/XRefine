"""
Data Generator (SSOT Pattern Applied)
=====================================
"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from xrefine.config import XRefineConfig
from xrefine.core.abeles import AbelesMatrix
from xrefine.core.smearing import instrument_reflectivity
from xrefine.core.structures import ParamSet
from xrefine.data.augmentations import XRRAugmentor


def generate_synthetic_batch(
    batch_size: int,
    q_grid: torch.Tensor,
    config: XRefineConfig,
    simulator: AbelesMatrix,
    augmenter: XRRAugmentor | None = None,
) -> tuple[torch.Tensor, ParamSet]:

    device = config.device
    q_grid = q_grid.to(device)

    thickness_list = []
    roughness_list = []
    sld_real_list = []
    sld_imag_list = []

    layer_names = []

    for layer in config.sample.layers:
        layer_names.append(layer.name)

        # GPU Random Generation
        t = (
            torch.rand(batch_size, 1, device=device) * (layer.thickness[1] - layer.thickness[0])
            + layer.thickness[0]
        )
        r = (
            torch.rand(batch_size, 1, device=device) * (layer.roughness[1] - layer.roughness[0])
            + layer.roughness[0]
        )
        s_r = (
            torch.rand(batch_size, 1, device=device) * (layer.sld[1] - layer.sld[0]) + layer.sld[0]
        )

        if layer.sld_imag:
            s_i = (
                torch.rand(batch_size, 1, device=device) * (layer.sld_imag[1] - layer.sld_imag[0])
                + layer.sld_imag[0]
            )
        else:
            s_i = torch.zeros(batch_size, 1, device=device)

        thickness_list.append(t)
        roughness_list.append(r)
        sld_real_list.append(s_r)
        sld_imag_list.append(s_i)

    # Substrate
    sub = config.sample.substrate
    sub_r = (
        torch.rand(batch_size, 1, device=device) * (sub.roughness[1] - sub.roughness[0])
        + sub.roughness[0]
    )
    sub_sld_r = torch.rand(batch_size, 1, device=device) * (sub.sld[1] - sub.sld[0]) + sub.sld[0]
    sub_sld_i = (
        torch.rand(batch_size, 1, device=device) * (sub.sld_imag[1] - sub.sld_imag[0])
        + sub.sld_imag[0]
    )

    # Ambient Parameters (For Physics)
    amb_sld_r = torch.zeros(batch_size, 1, device=device)
    amb_sld_i = torch.zeros(batch_size, 1, device=device)
    amb_thickness = torch.zeros(batch_size, 1, device=device)

    d_stack = torch.stack([amb_thickness] + thickness_list, dim=1).squeeze(-1)
    sigma_stack = torch.stack(roughness_list + [sub_r], dim=1).squeeze(-1)

    rho_stack = torch.stack([amb_sld_r] + sld_real_list + [sub_sld_r], dim=1).squeeze(-1)
    rho_imag_stack = torch.stack([amb_sld_i] + sld_imag_list + [sub_sld_i], dim=1).squeeze(-1)
    sld_stack = torch.complex(rho_stack, rho_imag_stack)

    if q_grid.dim() == 1:
        q_batch = q_grid.unsqueeze(0).expand(batch_size, -1)
    else:
        q_batch = q_grid

    flat_params = {}

    # (1) Layers
    for i, name in enumerate(layer_names):
        flat_params[f"{name}.thickness"] = thickness_list[i]
        flat_params[f"{name}.roughness"] = roughness_list[i]
        flat_params[f"{name}.sld"] = sld_real_list[i]
        flat_params[f"{name}.sld_imag"] = sld_imag_list[i]

    # (2) Substrate
    flat_params["Substrate.roughness"] = sub_r
    flat_params["Substrate.sld"] = sub_sld_r
    flat_params["Substrate.sld_imag"] = sub_sld_i

    # (3) Instrument Parameters Generation (Dynamic & Strict)
    # sorted so the key order is deterministic
    target_keys = sorted(ParamSet.INSTRUMENT_KEYS)

    for key in target_keys:
        rng = None
        if hasattr(config.sample, key):
            rng = getattr(config.sample, key)

        if rng is not None:
            # Random Generation (Uniform)
            val = torch.rand(batch_size, 1, device=device) * (rng[1] - rng[0]) + rng[0]
            flat_params[key] = val
        else:
            raise ValueError(
                f"🚨 Critical Configuration Error: Parameter '{key}' is required by ParamSet but missing in config.sample!\n"
                f"   Please check your 'config.yaml' under 'sample:' section."
            )

    # The curve goes through the same instrument forward model the models reconstruct with
    # (2θ offset, footprint, resolution), using the instrument labels just drawn: a label
    # the data never shows cannot be learned, and the reconstruction would apply it anyway.
    reflectivity = instrument_reflectivity(
        q_batch,
        d_stack,
        sigma_stack,
        sld_stack,
        flat_params["L"],
        flat_params["beam_width"],
        flat_params["dq"],
        flat_params["tth_offset"],
        simulator,
        config.instrument.wavelength,
    )

    params = ParamSet(params=flat_params, layer_names=layer_names, device=device)

    if augmenter is not None:
        reflectivity = augmenter(q_batch, reflectivity, params)

    # same normalization as the real-data loader
    from xrefine.data.normalization import apply_normalization_tensor

    reflectivity, scale = apply_normalization_tensor(reflectivity, q_grid, config)

    # [DESIGN FIX] Normalization destroys original absolute scale processing.
    # We must adjust target labels ('i0', 'bkg') to match the normalized data context.
    if config.data.normalization_mode.lower() != "none":
        # [USER REQUEST] I0 Target Redefinition
        # Since data is normalized (e.g., Anchor 0.2 deg = 1.0), the effective I0 of the curve IS 1.0.
        # We overwrite the target label I0 to 1.0, effectively removing I0 prediction variability
        # while training on robustly augmented (then normalized) inputs.
        if "i0" in params._params:
            # params._params['i0'] = params._params['i0'] / scale
            # Force target to 1.0 (The model should predict normalized physics)
            params._params["i0"].fill_(1.0)

        if "bkg" in params._params:
            params._params["bkg"] = params._params["bkg"] - torch.log10(scale + 1e-12)

    return reflectivity, params


class XRRStreamingDataset(IterableDataset):
    def __init__(self, config: XRefineConfig, q_grid: torch.Tensor, steps_per_epoch: int):
        super().__init__()
        self.config = config
        self.q_grid = q_grid
        self.steps_per_epoch = steps_per_epoch
        self.augmenter = XRRAugmentor(config).to(config.device)

    def __iter__(self):
        # Worker Loop
        for _ in range(self.steps_per_epoch):
            r_noisy, params_gt = generate_synthetic_batch(
                batch_size=self.config.train.batch_size,
                q_grid=self.q_grid,
                config=self.config,
                simulator=AbelesMatrix(device=self.config.device),
                augmenter=self.augmenter,
            )
            yield r_noisy, params_gt


def get_xrr_dataloader(config: XRefineConfig, q_grid: Tensor) -> DataLoader:
    dataset = XRRStreamingDataset(config, q_grid, steps_per_epoch=config.train.steps_per_epoch)
    return DataLoader(dataset, batch_size=None, num_workers=0)
