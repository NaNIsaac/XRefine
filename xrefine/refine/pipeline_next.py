"""
Refinement Pipeline (Robust Version)
====================================
This pipeline integrates 'Smart-Lock' logic and uses 'optimizer_robust'
to prevent physical violations and parameter explosion during refinement.
"""

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from xrefine.checkpoint import load_checkpoint
from xrefine.config import XRefineConfig
from xrefine.constants import LOG_MASK_VALUE, LOG_VALID_THRESHOLD
from xrefine.core.abeles import AbelesMatrix
from xrefine.core.smearing import abeles_constant_smearing
from xrefine.logger import logger
from xrefine.models import get_model
from xrefine.physics_utils import tth2q

# the config injected at run time takes precedence over this default
from xrefine.refine.config import DEFAULT_CONFIG
from xrefine.refine.optimizer import optimize_step as optimize_global

# robust optimizer with safeguards, used instead of the original optimize_step
from xrefine.refine.optimizer_robust import optimize_step as optimize_robust


def calculate_fom(model, params, log_r_obs, config, cutoff):
    device = config.device
    q_grid = config.q_grid.to(device)
    with torch.no_grad():
        t, r, s = params.assemble_structure()
        r_sim = model.simulator(q_grid, t, r, s)
        i0 = params.i0 if hasattr(params, "i0") else 1.0
        bkg = params.bkg if hasattr(params, "bkg") else -10.0
        r_sim = r_sim * i0 + (10**bkg)
        log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))

        mask = (log_r_obs > LOG_VALID_THRESHOLD) & (q_grid <= cutoff)
        diff = (log_r_sim - log_r_obs)[mask]
        mse = torch.mean(diff**2).item()
    return {"LogMSE": mse}


def report_stage_horizontal(stage_name, p_init, p_final, fom_init, fom_final):
    """Print a horizontal table using a pandas DataFrame."""
    imse, fmse = fom_init["LogMSE"], fom_final["LogMSE"]
    imp = (imse - fmse) / imse * 100 if imse != 0 else 0

    lines = []
    lines.append("=" * 90)
    lines.append(f"Result: {stage_name} | LogMSE: {imse:.5f} -> {fmse:.5f} ({imp:+.1f}%)")
    lines.append("-" * 90)

    all_keys = list(p_final._params.keys())

    groups = {
        "Target": [k for k in all_keys if "Target_Film" in k],
        "Oxide": [k for k in all_keys if "Top_Layer" in k],
        "System": [k for k in all_keys if "Target" not in k and "Top" not in k],
    }

    name_map = {
        "thickness": "d",
        "roughness": "sig",
        "sld": "rho",
        "sld_imag": "iRho",
        "beam_width": "Beam",
        "tth_offset": "Offset",
        "i0": "I0",
        "bkg": "Bkg",
        "L": "Len",
        "dq": "Res",
    }

    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)
    pd.set_option("display.colheader_justify", "right")

    for label, keys in groups.items():
        if not keys:
            continue

        # Sort
        order = {"thickness": 1, "roughness": 2, "sld": 3, "sld_imag": 4}
        keys.sort(key=lambda x: order.get(x.split(".")[-1], 99))

        data = {}
        for k in keys:
            short = k.split(".")[-1]
            col = name_map.get(short, short)
            if "Substrate" in k:
                col = f"Sub_{col}"

            v_i = p_init._params[k].item()
            v_f = p_final._params[k].item()

            diff = v_f - v_i
            if abs(diff) > 1e-4:
                val_str = f"{v_f:.3f} ({diff:+.3f})"
            else:
                val_str = f"{v_f:.3f}"

            data[col] = [val_str]

        df = pd.DataFrame(data, index=[label])
        lines.append(df.to_string())
        lines.append("")

    lines.append("=" * 90)
    full_log_str = "\n" + "\n".join(lines)
    logger.info(full_log_str)


def load_real_data(file_path: str, config: XRefineConfig) -> torch.Tensor:
    """
    Load and Normalize Data using configurable normalization strategy.

    Normalization mode is controlled by config.data.normalization_mode:
    - "plateau": Low-angle top-mean = 1.0 (robust, recommended)
    - "max": Global max = 1.0 (simple)
    - "anchor": Specific angle = 1.0 (physical anchor)
    - "none": No normalization (raw data)
    """
    from xrefine.data.normalization import apply_normalization

    try:
        df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python", dtype=str)
        for col in df.columns:
            df[col] = (
                df[col]
                .str.replace("(", "", regex=False)
                .str.replace(")", "", regex=False)
                .astype(float)
            )

        x_raw_tth, y_raw = df.values[:, 0], df.values[:, 1]
        x_raw_q = tth2q(x_raw_tth, config.instrument.wavelength)

        # configurable normalization via config.data.normalization_mode
        y_norm, scale = apply_normalization(y_raw, config, q_array=x_raw_q, tth_array=x_raw_tth)
        logger.info(f"  [Data] Normalization Scale Factor: {scale:.4e}")

        # log transform (values > 1.0 are allowed for noise handling)
        log_y_raw = np.log10(np.maximum(y_norm, 1e-11))

        q_target = config.q_grid.cpu().numpy()
        log_y_interp = np.interp(q_target, x_raw_q, log_y_raw)

        # Out of range masking
        log_y_interp[(q_target < x_raw_q.min()) | (q_target > x_raw_q.max())] = LOG_MASK_VALUE

        return torch.from_numpy(log_y_interp).float().view(1, -1)

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def pre_fit_i0(model, p_init, log_r_obs, xrr_config, q_limit: float = 0.06):
    """
    Pre-fit I0 to match the low-angle plateau region.

    This ensures the simulation scale matches the normalized data BEFORE
    any other refinement. I0 is fitted using Linear MSE in the plateau region,
    then fixed for subsequent stages.

    Args:
        model: The PINN model with simulator
        p_init: Initial ParamSet from AI
        log_r_obs: Normalized observed data (log10 scale)
        xrr_config: XRefineConfig
        q_limit: Upper Q limit for plateau fitting (default 0.06 Å⁻¹)

    Returns:
        Updated ParamSet with fitted I0
    """
    from torch import optim

    device = xrr_config.device
    q_grid = xrr_config.q_grid.to(device)
    log_r_obs_gpu = log_r_obs.to(device)

    # Mask for low-Q plateau region only
    mask = (log_r_obs_gpu > LOG_VALID_THRESHOLD) & (q_grid < q_limit)

    if mask.sum() < 5:
        logger.warning(f"Pre-fit I0: Not enough points in Q < {q_limit}. Skipping.")
        return p_init

    # Clone params and enable gradient only for I0
    p_current = p_init.clone()

    if "i0" not in p_current._params:
        logger.warning("Pre-fit I0: 'i0' not in params. Skipping.")
        return p_init

    i0_tensor = p_current._params["i0"]
    i0_tensor.requires_grad = True

    # Also fit bkg for better matching
    bkg_tensor = p_current._params.get("bkg")
    params_to_opt = [i0_tensor]
    if bkg_tensor is not None:
        bkg_tensor.requires_grad = True
        params_to_opt.append(bkg_tensor)

    # Convert observed data to linear scale for comparison
    r_obs_linear = 10**log_r_obs_gpu

    # [NEW] Estimate initial I0 from data scale ratio in plateau region
    # For 'none' mode, observed data can be very large (e.g., 1e6 counts)
    with torch.no_grad():
        # Same forward model as the stages (footprint, offset, resolution, i0, background)
        _, r_sim_raw, _ = model.forward_with_params(p_current)

        # Estimate scale from plateau region
        obs_plateau = r_obs_linear[mask].mean()
        sim_plateau = r_sim_raw[mask].mean()

        if sim_plateau > 1e-12:
            estimated_i0 = (p_current._params["i0"] * obs_plateau / sim_plateau).item()
            # Clamp to reasonable range
            estimated_i0 = max(0.01, min(estimated_i0, 1e6))
            p_current._params["i0"].fill_(estimated_i0)
            logger.info(
                f"  [Pre-fit] Initial I0 estimate: {estimated_i0:.4f} (obs/sim = {obs_plateau.item():.2e}/{sim_plateau.item():.2e})"
            )

    # Adaptive I0 range based on normalization mode
    norm_mode = xrr_config.data.normalization_mode.lower()
    if norm_mode == "none":
        # Raw data can have very different scales
        i0_min, i0_max = 0.01, 1e6
    else:
        # Normalized data: I0 should be close to 1.0
        i0_min, i0_max = 0.1, 5.0

    optimizer = optim.Adam(params_to_opt, lr=0.1)

    best_loss = float("inf")
    best_i0 = p_current._params["i0"].item()
    best_bkg = bkg_tensor.item() if bkg_tensor is not None else -10.0

    for _step in range(100):  # More iterations
        optimizer.zero_grad()

        # Simulate
        with torch.enable_grad():
            _, r_sim_scaled, _ = model.forward_with_params(p_current)  # includes i0 and bkg

            # Linear MSE in plateau region (match absolute scale)
            loss = torch.mean(((r_sim_scaled - r_obs_linear) ** 2) * mask)

            loss.backward()
            optimizer.step()

            # Clamp I0 to adaptive range
            with torch.no_grad():
                p_current._params["i0"].clamp_(i0_min, i0_max)
                if "bkg" in p_current._params:
                    p_current._params["bkg"].clamp_(-12.0, -3.0)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_i0 = p_current._params["i0"].item()
            if "bkg" in p_current._params:
                best_bkg = p_current._params["bkg"].item()

    # Set best values and detach
    p_current._params["i0"] = torch.tensor([[best_i0]], device=device)
    if "bkg" in p_current._params:
        p_current._params["bkg"] = torch.tensor([[best_bkg]], device=device)

    logger.info(
        f"✅ Pre-fit I0: {p_init._params['i0'].item():.4f} → {best_i0:.4f} "
        f"(bkg: {best_bkg:.2f}, Q < {q_limit})"
    )

    return p_current


def grid_search_thickness(model, p_init, log_r_obs, xrr_config, refine_config, stage_config):
    target_key = stage_config.grid_target_key
    if target_key not in p_init._params:
        return 0.0
    device = xrr_config.device
    q_grid = xrr_config.q_grid.to(device)
    log_r_obs_gpu = log_r_obs.to(device)
    r_obs_linear = 10 ** log_r_obs_gpu.squeeze()
    peak_idx = int(torch.argmax(r_obs_linear).item())
    q_critical = q_grid[peak_idx].item()
    q_start = q_critical + 0.01
    q_end = refine_config.q_cutoff
    mask = (log_r_obs_gpu > LOG_VALID_THRESHOLD) & (q_grid > q_start) & (q_grid < q_end)
    d_min, d_max = stage_config.grid_range
    d_grid = np.linspace(d_min, d_max, stage_config.grid_points)
    p_scan = p_init.clone()
    best_loss = 1e9
    best_d = p_init._params[target_key].item()
    for d_val in d_grid:
        p_scan._params[target_key].fill_(d_val)
        with torch.no_grad():
            if hasattr(model, "forward_with_params"):
                _, r_sim, _ = model.forward_with_params(p_scan)
            else:
                t, r, s = p_scan.assemble_structure()
                r_sim = model.simulator(q_grid, t, r, s)
            r_sim_norm = r_sim / (r_sim.max() + 1e-12)
            log_r_sim = torch.log10(torch.clamp(r_sim_norm, min=1e-12))
            sim_pattern = log_r_sim[mask]
            vx = sim_pattern - sim_pattern.mean()
            vy = log_r_obs_gpu[mask] - log_r_obs_gpu[mask].mean()
            corr = torch.sum(vx * vy) / (
                torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-9
            )
            loss = 1.0 - corr.item()
            if loss < best_loss:
                best_loss = loss
                best_d = d_val
    return best_d


def run_refinement(model, p_init, log_r_obs, xrr_config, refine_config=None):
    if refine_config is None:
        refine_config = DEFAULT_CONFIG
    p_current = p_init
    log_r_obs_gpu = log_r_obs.to(xrr_config.device)
    logger.info("Starting Hybrid Refinement Pipeline...")

    # [NEW] Stage 0: Pre-fit I0 (Only for non-anchor modes)
    # If Anchor mode is active, we already applied strict 0.2 deg correction in run_real_inference.
    # Running pre-fit here would overwrite that precise correction with a plateau average.
    if xrr_config.data.normalization_mode != "anchor":
        q_limit = getattr(xrr_config.data, "plateau_q_limit", 0.06)
        p_current = pre_fit_i0(model, p_current, log_r_obs, xrr_config, q_limit=q_limit)
    else:
        logger.info("⚓ [Refinement] Skipping Pre-fit I0 (Anchor Mode active)")

    # [Pre-fit I0 as initial estimate only - stages can still adjust]
    # fitted_i0 = p_current._params['i0'].item() if 'i0' in p_current._params else 1.0

    for _i, stage in enumerate(refine_config.stages):
        stage_name = stage.name
        p_start_stage = p_current.clone()
        fom_init = calculate_fom(
            model, p_current, log_r_obs_gpu, xrr_config, refine_config.q_cutoff
        )

        current_stage_config = deepcopy(stage)

        # [REMOVED I0 Locking - let stages optimize I0 if needed]

        # physics constraint: roughness <= 0.5 * thickness, to keep the fringes from washing out
        for layer_key in ["Target_Film", "Top_Layer"]:
            thick_key = f"{layer_key}.thickness"
            rough_key = f"{layer_key}.roughness"
            if thick_key in p_current._params and rough_key in current_stage_config.optimize_keys:
                t_val = p_current._params[thick_key].item()
                # Ensure roughness doesn't exceed 50% of thickness, but allow at least 2.0A
                r_max = max(2.0, t_val * 0.5)

                if not current_stage_config.temp_bounds:
                    current_stage_config.temp_bounds = {}

                # keep the intersection with existing bounds
                existing_bounds = current_stage_config.temp_bounds.get(rough_key)
                if existing_bounds:
                    new_max = min(existing_bounds[1], r_max)
                    current_stage_config.temp_bounds[rough_key] = (existing_bounds[0], new_max)
                else:
                    current_stage_config.temp_bounds[rough_key] = (0.0, r_max)

                # logger.info(f"  [Physics] Constraining {rough_key} <= {r_max:.2f} (Half-Thickness Rule)")

        if hasattr(stage, "relative_bounds") and stage.relative_bounds:
            if not current_stage_config.temp_bounds:
                current_stage_config.temp_bounds = {}
            for key, margin in stage.relative_bounds.items():
                if key in p_current._params:
                    current_val = p_current._params[key].item()
                    if any(x in key for x in ["offset", "bkg", "dq", "res", "beam"]):
                        p_min, p_max = current_val - margin, current_val + margin
                    else:
                        p_min, p_max = current_val * (1.0 - margin), current_val * (1.0 + margin)
                        if p_min > p_max:
                            p_min, p_max = p_max, p_min
                    current_stage_config.temp_bounds[key] = (p_min, p_max)
            stage_name += " (Smart-Lock)"

        if refine_config.trust_ai and stage.use_grid_search:
            # ... (Trust AI Grid logic same as before) ...
            target_key = stage.grid_target_key
            current_d = 0.0
            if target_key in p_current._params:
                current_d = p_current._params[target_key].item()
            if current_d > 10.0:
                margin = refine_config.ai_trust_range
                new_min = max(20.0, current_d * (1.0 - margin))
                new_max = current_d * (1.0 + margin)
                current_stage_config.grid_range = (new_min, new_max)
                stage_name += " (Trust Grid)"

        if current_stage_config.use_grid_search:
            # Type A: Grid Search (Thickness)
            best_val = grid_search_thickness(
                model, p_current, log_r_obs, xrr_config, refine_config, current_stage_config
            )
            p_current = p_current.clone()
            target_key = current_stage_config.grid_target_key
            if target_key in p_current._params:
                p_current._params[target_key].fill_(best_val)

        else:
            if current_stage_config.q_max is None:
                current_stage_config.q_max = refine_config.q_cutoff

            valid_keys = {k for k in current_stage_config.optimize_keys if k in p_current._params}

            if valid_keys:
                current_stage_config.optimize_keys = valid_keys

                # hybrid switching: early stages (1 Scale, 2 Anchor) use the global optimizer (DE)
                # because the initial position matters; this restores the low-angle fit
                is_early_stage = any(
                    x in stage_name for x in ["Scale", "Footprint", "Anchor", "Roundness"]
                )

                if is_early_stage:
                    # Legacy DE Optimizer (Scipy)
                    # DE needs no gradient, so it copes with a bad initial value
                    p_current = optimize_global(
                        model, p_current, log_r_obs, xrr_config, current_stage_config, verbose=False
                    )
                else:
                    # Robust GD Optimizer (Torch)
                    # stages 4 (Shape) and 5 (Polish) are fine tuning, so use fast and safe GD;
                    # gradient clipping prevents thickness explosion
                    p_current = optimize_robust(
                        model, p_current, log_r_obs, xrr_config, current_stage_config, verbose=False
                    )

        fom_final = calculate_fom(
            model, p_current, log_r_obs_gpu, xrr_config, refine_config.q_cutoff
        )
        report_stage_horizontal(stage_name, p_start_stage, p_current, fom_init, fom_final)

    logger.info("Hybrid Refinement Complete.")
    return p_current


@torch.no_grad()
def run_real_inference(
    checkpoint_path, data_path, output_dir="./results", refine_config=None, q_cutoff=None
):
    ckpt = load_checkpoint(checkpoint_path)
    config_data = ckpt.get("config")

    # [NEW] Force Anchor Normalization for Safety (User Request)
    if isinstance(config_data, dict):
        if "data" not in config_data:
            config_data["data"] = {}
        config_data["data"]["normalization_mode"] = "anchor"
        config_data["data"]["anchor_tth"] = 0.2
        logger.info("[New Pipeline] Forcing Anchor Normalization (0.2 deg)")

    xrr_config = XRefineConfig(**config_data) if isinstance(config_data, dict) else config_data

    # [NEW] Override data config from current config.yaml
    # This ensures normalization settings from config.yaml are used, not checkpoint
    from xrefine.config import CONFIG

    xrr_config.data = CONFIG.data
    logger.info(f"  [Config] Using normalization mode: {xrr_config.data.normalization_mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xrr_config.device = device
    xrr_config.q_grid = xrr_config.q_grid.to(device)
    model = get_model(xrr_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    simulator = AbelesMatrix(device=device)
    model.simulator = simulator
    input_tensor = load_real_data(data_path, xrr_config)
    q_grid_np = xrr_config.q_grid.cpu().numpy()
    actual_cutoff = q_cutoff if q_cutoff else 0.45
    logger.info(f"Cutoff: Q > {actual_cutoff:.3f}")
    if refine_config is None:
        refine_config = DEFAULT_CONFIG
    refine_config.q_cutoff = actual_cutoff
    mask_indices = q_grid_np > actual_cutoff
    masked_input = input_tensor.clone()
    masked_input[0, mask_indices] = LOG_MASK_VALUE
    masked_input_gpu = masked_input.to(device)
    out = model(masked_input_gpu, fixed_values=None)
    p_ai = out[0] if isinstance(out, tuple) else out

    # [TIP] Force Realistic Initial Beam Width
    if "beam_width" in p_ai._params:
        p_ai._params["beam_width"].fill_(0.1)

    # [USER REQUEST] Force AI to match Data at Anchor Point (0.2 deg)
    # This guarantees the curve intersects at the normalization point.
    try:
        if isinstance(xrr_config.data.anchor_tth, (float, int)):
            tth_anchor = float(xrr_config.data.anchor_tth)
            # Find Q index for anchor tth
            wav = xrr_config.instrument.wavelength
            q_anchor = 4 * np.pi * np.sin(np.radians(tth_anchor / 2)) / wav

            # Use closest grid point
            q_grid_gpu = xrr_config.q_grid.to(device)
            idx_anchor = torch.argmin(torch.abs(q_grid_gpu - q_anchor))

            # Simulate current AI guess at this point
            with torch.no_grad():
                # simulate R (normalized scale 0 to 1)
                t, r, s = p_ai.assemble_structure()
                dum_q = q_grid_gpu[idx_anchor].view(1, 1)  # Single Point

                # Full simulation at anchor point
                r_sim_anchor = abeles_constant_smearing(
                    q=dum_q,
                    thickness=t,
                    roughness=r,
                    sld=s,
                    sample_length=p_ai.L if hasattr(p_ai, "L") else None,
                    beam_width=p_ai.beam_width if hasattr(p_ai, "beam_width") else None,
                    dq=p_ai.dq if hasattr(p_ai, "dq") else torch.tensor([[0.0]], device=device),
                    gauss_num=11,
                    abeles_func=simulator,
                    wavelength=wav,
                )

                # Check current I0
                cur_i0 = p_ai.i0.item() if hasattr(p_ai, "i0") else 1.0

                # Sim Value = Correct_Sim * I0 (Background ignored at peak)
                current_val = r_sim_anchor.item() * cur_i0

                # Target Value is 1.0 (because we normalized input to 1.0 at anchor)
                # Correction Factor
                if current_val > 1e-12:
                    correction = 1.0 / current_val
                    new_i0 = cur_i0 * correction
                    if hasattr(p_ai, "i0"):
                        p_ai._params["i0"].fill_(new_i0)
                        logger.info(
                            f"⚓ [Anchor Correction] Adjusting I0: {cur_i0:.4f} -> {new_i0:.4f} (Match at {tth_anchor} deg)"
                        )
    except Exception as e:
        logger.warning(f"Failed to apply anchor correction: {e}")

    logger.info("AI Initial Guess Done.")

    p_refined = run_refinement(model, p_ai, masked_input, xrr_config, refine_config)

    def simulate_full(params):
        t, r, s = params.assemble_structure()
        L = params.L if hasattr(params, "L") else None
        beam_width = params.beam_width if hasattr(params, "beam_width") else None
        dq_val = params.dq if hasattr(params, "dq") else torch.tensor(0.0, device=device)
        r_smeared = abeles_constant_smearing(
            q=xrr_config.q_grid.unsqueeze(0),
            thickness=t,
            roughness=r,
            sld=s,
            sample_length=L,
            beam_width=beam_width,
            dq=dq_val.view(1, 1),
            gauss_num=51,
            abeles_func=simulator,
            wavelength=xrr_config.instrument.wavelength,
        )
        i0_val = params.i0.item() if hasattr(params, "i0") else 1.0
        bkg_val = params.bkg.item() if hasattr(params, "bkg") else -10.0
        r_final = r_smeared * i0_val + (10**bkg_val)
        return r_final.view(1, -1)

    r_final = simulate_full(p_refined)
    r_ai_sim = simulate_full(p_ai)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_name = Path(data_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _fig, ax = plt.subplots(figsize=(10, 6))
    q = q_grid_np
    y_obs = 10 ** masked_input[0].cpu().numpy()
    valid_mask = (masked_input[0].cpu().numpy() > LOG_VALID_THRESHOLD) & (q <= actual_cutoff)
    ax.semilogy(q[valid_mask], y_obs[valid_mask], "ko", ms=3, alpha=0.4, label="Measured Data")
    ax.semilogy(q, r_ai_sim[0].cpu().numpy(), "b--", lw=1, alpha=0.7, label="AI Init (Smeared)")
    ax.semilogy(q, r_final[0].cpu().numpy(), "r-", lw=2, label="Refined (Smeared)")
    ax.axvline(x=actual_cutoff, color="gray", linestyle=":", label="Cutoff Limit")
    ax.set_xlabel("Q ($Å^{-1}$)", fontsize=12)
    ax.set_ylabel("Reflectivity", fontsize=12)
    ax.set_title(f"XRefine Result: {data_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    if valid_mask.any():
        q_max = q[valid_mask].max()
        ax.set_xlim(0, max(q_max * 1.1, 0.2))
        y_min_data = y_obs[valid_mask].min()
        ax.set_ylim(bottom=max(y_min_data * 0.1, 1e-10), top=2.0)
    plt.tight_layout()
    plt.savefig(output_path / f"{data_name}_{timestamp}_result.png", dpi=150)
    plt.close()
    final_fom = calculate_fom(model, p_refined, masked_input_gpu, xrr_config, actual_cutoff)
    param_dict = {k: v.item() for k, v in p_refined._params.items()}
    param_dict.update(final_fom)
    pd.DataFrame([param_dict]).to_csv(
        output_path / f"{data_name}_{timestamp}_params.csv", index=False
    )
    return p_refined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Path to .dat file")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="./results")
    args = parser.parse_args()
    run_real_inference(args.checkpoint, args.data, args.out)
