"""
Refinement Optimizer (Robust Gradient Descent)
==============================================
Update: Implements 'Adaptive Gradient Clipping'.
- Early Stages (Scale, SLD): Loose clipping (Allow big moves).
- Later Stages (Thickness, Shape): Strict clipping (Prevent explosion).
"""

import torch
import torch.optim as optim

from xrefine.constants import LOG_VALID_THRESHOLD
from xrefine.logger import logger


def optimize_step(model, p_init, log_r_obs, xrr_config, stage_config, verbose=False):
    # 1. Setup Data & Config
    device = xrr_config.device
    log_r_obs_gpu = log_r_obs.to(device)
    q_grid = xrr_config.q_grid.to(device)

    q_max_limit = stage_config.q_max if stage_config.q_max else 100.0
    mask = (log_r_obs_gpu > LOG_VALID_THRESHOLD) & (q_grid <= q_max_limit)
    if stage_config.q_min:
        mask &= q_grid >= stage_config.q_min

    # 2. Parameter Normalization
    param_info_list = []
    opt_tensor_list = []

    reconst_dict = {k: v.detach().clone() for k, v in p_init._params.items()}

    # Range Lookup Helper
    def get_range(key):
        # 1. Global Parameters
        if hasattr(xrr_config.sample, key):
            return getattr(xrr_config.sample, key)

        # 2. Substrate Parameters
        if key.startswith("Substrate."):
            sub_key = key.split(".")[-1]
            if hasattr(xrr_config.sample.substrate, sub_key):
                return getattr(xrr_config.sample.substrate, sub_key)

        # 3. Layer Parameters
        if "." in key:
            layer_name, param_name = key.split(".")
            for layer in xrr_config.sample.layers:
                if layer.name == layer_name and hasattr(layer, param_name):
                    return getattr(layer, param_name)

        return None

    for key in stage_config.optimize_keys:
        if key not in p_init._params:
            continue

        # Get Bounds
        rng = get_range(key)
        if rng is not None:
            min_v, max_v = rng
            # Ensure safe bounds
            if min_v >= max_v:
                max_v = min_v + 1e-6
        else:
            # Fallback (Safety)
            val = p_init._params[key].item()
            min_v, max_v = val * 0.5, val * 1.5
            if min_v == max_v:
                min_v, max_v = val - 0.1, val + 0.1

        current_val = p_init._params[key].detach()

        # Normalize to [0, 1]
        # norm = (val - min) / (max - min)
        norm_val = (current_val - min_v) / (max_v - min_v + 1e-9)
        norm_val = torch.clamp(norm_val, 0.0, 1.0)

        # Create Learnable Parameter
        p_tensor = norm_val.clone().requires_grad_(True)

        opt_tensor_list.append(p_tensor)
        param_info_list.append({"key": key, "min": min_v, "max": max_v, "tensor": p_tensor})

    if not opt_tensor_list:
        if verbose:
            logger.warning(f"Stage {stage_config.name}: No parameters to optimize.")
        return p_init

    params_to_opt = opt_tensor_list

    # 3. Select Optimizer
    use_lbfgs = "Final" in stage_config.name or "LBFGS" in stage_config.name
    is_sensitive_stage = any(x in stage_config.name for x in ["Thickness", "Shape", "Polish"])

    if use_lbfgs:
        optimizer = optim.LBFGS(
            params_to_opt, lr=1.0, max_iter=20, history_size=10, line_search_fn="strong_wolfe"
        )
        num_steps = max(1, stage_config.maxiter // 4)
    else:
        # Normalized Space에서는 LR 0.01~0.05 정도가 적절함
        base_lr = 0.01 if is_sensitive_stage else 0.05
        optimizer = optim.Adam(params_to_opt, lr=base_lr)
        num_steps = stage_config.maxiter

    loss_fn = stage_config.loss_fn
    bounds_map = stage_config.temp_bounds or {}

    # 4. Optimization Loop
    best_loss = float("inf")
    # p_best keeps a *copy* of VALUES, not tensors
    p_best_vals = {k: v.detach().clone() for k, v in reconst_dict.items()}

    def get_real_values():
        current_vals = {}
        for info in param_info_list:
            # Denormalize: val = norm * (max - min) + min
            # Clamp norm to [0, 1] for safety before mapping
            norm_v = torch.clamp(info["tensor"], 0.0, 1.0)
            real_v = norm_v * (info["max"] - info["min"]) + info["min"]
            current_vals[info["key"]] = real_v
        return current_vals

    def closure():
        optimizer.zero_grad()

        # Update dynamic params
        real_vals = get_real_values()
        for k, v in real_vals.items():
            reconst_dict[k] = v

        # Reconstruct ParamSet (Virtual)
        # We cheat by using p_init structure but replacing internal dict
        p_current = p_init.__class__(reconst_dict, p_init.layer_names, device)

        t, r, s = p_current.assemble_structure()
        r_sim = model.simulator(q_grid, t, r, s)

        i0 = p_current.i0 if hasattr(p_current, "i0") else 1.0
        bkg = p_current.bkg if hasattr(p_current, "bkg") else -10.0
        r_sim = r_sim * i0 + (10**bkg)

        loss = loss_fn(r_sim, log_r_obs_gpu, mask)

        # [Roughness Penalty]
        # Prevent roughness from killing fringes completely
        # 1. Thickness > 0 Penalty (Soft Constraint in Normalized Space)
        # 2. Roughness too high penalty
        pen = 0.0
        # If roughness > 20A, penalize
        # Note: We are optimization normalized params, but penalty on real values
        if "Target_Film.roughness" in reconst_dict:
            r_val = reconst_dict["Target_Film.roughness"]
            if r_val > 15.0:
                pen += (r_val - 15.0) * 0.1

        total_loss = loss + pen

        if total_loss.requires_grad:
            total_loss.backward()

        return total_loss

    for step in range(num_steps):
        try:
            if use_lbfgs:
                optimizer.step(closure)
            else:
                closure()
                optimizer.step()

            # Bounds Manifold Projection (Clamp to [0,1] in Norm Space)
            # This implicitly enforces Physical Bounds defined in Config.ranges
            with torch.no_grad():
                for p in params_to_opt:
                    p.clamp_(0.0, 1.0)

            # Keep Best
            with torch.no_grad():
                current_loss = closure().item()  # Re-eval with clamped values
                if current_loss < best_loss:
                    best_loss = current_loss
                    # Save snapshot of denormalized values
                    real_snapshot = get_real_values()
                    for k, v in real_snapshot.items():
                        p_best_vals[k] = v.detach().clone()

        except RuntimeError as e:
            logger.warning(f"Optimization interrupted at step {step}: {e}")
            break

    # Return constructed ParamSet from best values
    return p_init.__class__(p_best_vals, p_init.layer_names, device)
