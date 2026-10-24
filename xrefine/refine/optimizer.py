"""
Optimizer Step (With Temp Bounds Support)
=========================================
"""

import numpy as np
import torch
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from xrefine.config import XRefineConfig
from xrefine.constants import LOG_VALID_THRESHOLD
from xrefine.core.structures import ParamSet
from xrefine.logger import logger
from xrefine.refine.config import StageConfig
from xrefine.refine.losses import LossFn


def get_bounds_from_config(config: XRefineConfig, key: str) -> tuple[float, float]:
    sample = config.sample
    parts = key.split(".")
    try:
        if len(parts) == 1 and hasattr(sample, key):
            val = getattr(sample, key)
            if isinstance(val, list) and len(val) == 2:
                return tuple(val)
        if parts[0] == "Substrate" and hasattr(sample.substrate, parts[1]):
            return tuple(getattr(sample.substrate, parts[1]))
        for layer in sample.layers:
            if layer.name == parts[0] and hasattr(layer, parts[1]):
                return tuple(getattr(layer, parts[1]))
    except:
        pass
    return (0.0, 1.0)


def optimize_step(
    model,
    p_init: ParamSet,
    log_r_obs: torch.Tensor,
    xrr_config: XRefineConfig,
    stage_config: StageConfig,
    verbose: bool = False,
) -> ParamSet:
    target_device = xrr_config.device
    q_grid = xrr_config.q_grid.to(target_device)
    log_r_obs_gpu = log_r_obs.to(target_device)

    # Masking
    q_min = stage_config.q_min or 0.0
    q_max = stage_config.q_max or xrr_config.q_grid.max().item()
    mask = (log_r_obs_gpu > LOG_VALID_THRESHOLD) & (q_grid > q_min) & (q_grid < q_max)

    # Param Setup
    optimize_keys = stage_config.optimize_keys
    fixed_params = {}
    opt_param_info = []
    opt_guess_norm = []

    for k, v_tensor in p_init._params.items():
        val_flat = v_tensor.detach().cpu().numpy().flatten()
        current_val = float(val_flat[0])

        if k in stage_config.fixed_params:
            fixed_val = stage_config.fixed_params[k]
            fixed_params[k] = torch.tensor([[fixed_val]], dtype=torch.float32, device=target_device)
            continue

        if k in optimize_keys:
            # [FIX] Check for temporary bounds override first
            if k in stage_config.temp_bounds:
                p_min, p_max = stage_config.temp_bounds[k]
                # If current value is out of temp bounds, clamp it strictly
                if current_val > p_max or current_val < p_min:
                    current_val = np.clip(current_val, p_min, p_max)
            else:
                # Fallback to global config
                p_min, p_max = get_bounds_from_config(xrr_config, k)

            # Clip current value to bounds
            current_val = np.clip(current_val, p_min, p_max)

            opt_param_info.append({"key": k, "size": 1, "min": p_min, "max": p_max})
            norm_val = (current_val - p_min) / (p_max - p_min + 1e-9)
            opt_guess_norm.append(norm_val)
        else:
            fixed_params[k] = v_tensor.clone().detach()

    total_dim = len(opt_guess_norm)
    if total_dim == 0:
        return p_init

    loss_fn: LossFn = stage_config.loss_fn

    # Objective Function (Same as before)
    def objective(trial_vector):
        try:
            p_dict = {k: v for k, v in fixed_params.items()}
            cursor = 0
            for info in opt_param_info:
                phys_val = trial_vector[cursor] * (info["max"] - info["min"]) + info["min"]
                p_dict[info["key"]] = torch.tensor(
                    [[phys_val]], dtype=torch.float32, device=target_device
                )
                cursor += 1

            p_set = ParamSet(p_dict, p_init.layer_names, device=target_device)

            with torch.no_grad():
                if hasattr(model, "forward_with_params"):
                    _, r_sim, _ = model.forward_with_params(
                        p_set
                    )  # already includes i0 and background
                else:
                    t, r, s = p_set.assemble_structure()
                    r_sim = model.simulator(xrr_config.q_grid.to(target_device), t, r, s)
                    i0 = p_dict.get("i0", torch.tensor([[1.0]], device=target_device)).item()
                    bkg = p_dict.get("bkg", torch.tensor([[-10.0]], device=target_device)).item()
                    r_sim = r_sim * i0 + (10**bkg)

                loss = loss_fn(r_sim, log_r_obs_gpu, mask)

                # Physical Constraints (Global rules still apply)
                phys_penalty = 0.0
                for layer_name in p_init.layer_names:
                    t_key, r_key = f"{layer_name}.thickness", f"{layer_name}.roughness"
                    if t_key in p_dict and r_key in p_dict:
                        t_val, r_val = p_dict[t_key].item(), p_dict[r_key].item()
                        if r_val > t_val * 0.5 and t_val > 5.0:
                            phys_penalty += 10.0 * (r_val - t_val * 0.5)

                return float((loss + phys_penalty).item())
        except:
            return 1e9

    sampler = qmc.LatinHypercube(d=total_dim)
    init_pop = sampler.random(n=stage_config.popsize)
    init_pop[0] = np.array(opt_guess_norm)

    result = differential_evolution(
        objective,
        bounds=[(0, 1)] * total_dim,
        strategy="best1bin",
        maxiter=stage_config.maxiter,
        popsize=stage_config.popsize,
        mutation=(0.5, 1.0),
        recombination=0.7,
        init=init_pop,
        disp=False,
        workers=1,
    )

    final_dict = {k: v for k, v in fixed_params.items()}
    cursor = 0
    for info in opt_param_info:
        phys_val = result.x[cursor] * (info["max"] - info["min"]) + info["min"]
        final_dict[info["key"]] = torch.tensor(
            [[phys_val]], dtype=torch.float32, device=target_device
        )
        cursor += 1

    return ParamSet(final_dict, p_init.layer_names, device=target_device)
