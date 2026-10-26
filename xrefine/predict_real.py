import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks
from scipy.stats import qmc

from xrefine.checkpoint import load_checkpoint
from xrefine.config import XRefineConfig
from xrefine.core.structures import ParamSet
from xrefine.logger import logger
from xrefine.models import get_model
from xrefine.physics_utils import tth2q


def calculate_r2(y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute R² (coefficient of determination) over the masked points."""
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    if len(y_true_masked) == 0:
        return -1.0

    ss_res = torch.sum((y_true_masked - y_pred_masked) ** 2)
    ss_tot = torch.sum((y_true_masked - y_true_masked.mean()) ** 2)

    r2 = 1 - (ss_res / (ss_tot + 1e-12))
    return float(r2.item())


def load_real_data(file_path: str, config: XRefineConfig):
    """
    Load and Normalize Data using configurable normalization strategy.

    Normalization mode is controlled by config.data.normalization_mode.
    """
    from xrefine.data.normalization import apply_normalization

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

    # Configurable Normalization
    y_norm, scale = apply_normalization(y_raw, config, q_array=x_raw_q, tth_array=x_raw_tth)
    logger.info(f"  [Predict] Norm Scale: {scale:.4e}")
    log_y_raw = np.log10(np.maximum(y_norm, 1e-11))

    # Grid Interpolation
    q_target = config.q_grid.numpy()
    log_y_interp = np.interp(q_target, x_raw_q, log_y_raw)

    # mask points outside the measured q range with the log mask value
    log_y_interp[(q_target < x_raw_q.min()) | (q_target > x_raw_q.max())] = -15.0

    return torch.from_numpy(log_y_interp).float().view(1, -1)


def _optimize_subset(
    model,
    p_init: ParamSet,
    log_r_obs: torch.Tensor,
    config: XRefineConfig,
    optimize_keys: set[str],
    forced_params: dict[str, float] | None = None,
    q_min: float | None = None,
    q_max: float | None = None,
    use_gradient_loss: bool = False,
    use_linear_loss: bool = False,
    use_r2_loss: bool = False,
    maxiter: int = 100,
    popsize: int = 20,
) -> ParamSet:
    """Optimize a subset of the parameters with SciPy differential evolution."""
    target_device = next(model.parameters()).device
    range_config = config.ranges

    log_r_obs_gpu = log_r_obs.to(target_device)
    mask = log_r_obs_gpu > -14.0

    # Q-Range Cutting
    q_grid = config.q_grid.to(target_device)
    if q_max is not None:
        mask &= (q_grid < q_max).unsqueeze(0)
    if q_min is not None:
        mask &= (q_grid > q_min).unsqueeze(0)

    opt_param_info = []
    opt_guess_norm = []
    fixed_params = {}

    for k, v in p_init.items():
        if forced_params and k in forced_params:
            fixed_params[k] = torch.tensor(
                [[forced_params[k]]], dtype=torch.float32, device=target_device
            )
            continue

        p_min, p_max = getattr(range_config, k)
        val_flat = v.detach().cpu().numpy().flatten()

        if k in optimize_keys:
            opt_param_info.append({"key": k, "size": val_flat.size, "min": p_min, "max": p_max})
            # [DEBUG] Check bounds
            logger.info(f"🔎 Param {k}: Bounds [{p_min:.2f}, {p_max:.2f}]")

            norm_val = (val_flat - p_min) / (p_max - p_min + 1e-9)
            opt_guess_norm.extend(np.clip(norm_val, 0.0, 1.0).tolist())
        else:
            # keep the AI initial value
            fixed_params[k] = torch.tensor(
                val_flat, dtype=torch.float32, device=target_device
            ).view(1, -1)

    total_dim = len(opt_guess_norm)
    if total_dim == 0:
        return p_init

    def objective(trial_vector):
        try:
            p_dict = dict(fixed_params)
            cursor = 0
            for info in opt_param_info:
                k, size, p_min, p_max = info["key"], info["size"], info["min"], info["max"]
                phys_val = trial_vector[cursor : cursor + size] * (p_max - p_min) + p_min
                p_dict[k] = torch.tensor(phys_val, dtype=torch.float32, device=target_device).view(
                    1, size
                )
                cursor += size

            p_set = ParamSet(**p_dict)

            with torch.no_grad():
                _, r_sim, penalty = model.forward_with_params(p_set)
                log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))

                if use_linear_loss:
                    # linear R is sensitive to the fringe amplitude
                    r_obs_linear = 10**log_r_obs_gpu
                    mse_loss = torch.mean(((r_sim - r_obs_linear) ** 2) * mask)
                else:
                    mse_loss = torch.mean(((log_r_sim - log_r_obs_gpu) ** 2) * mask)
                loss = mse_loss

                # the gradient loss preserves the fringe shape
                if use_gradient_loss:
                    diff_sim = torch.diff(log_r_sim, dim=-1)
                    diff_obs = torch.diff(log_r_obs_gpu, dim=-1)
                    diff_mask = mask[:, 1:]
                    grad_loss = torch.mean(((diff_sim - diff_obs) ** 2) * diff_mask)
                    loss += grad_loss * 5.0

                if use_r2_loss:
                    r2 = calculate_r2(log_r_obs_gpu.squeeze(), log_r_sim.squeeze(), mask.squeeze())
                    loss = -r2

                r_pen = 0.0
                # no sigma > 5.0 penalty: higher roughness is allowed to match the damping

                # roughness < thickness / 2: a layer cannot persist when the roughness exceeds half
                # its thickness
                if p_set.sig_f.item() > (p_set.d_f.item() / 2.0):
                    r_pen += 100.0 + (p_set.sig_f.item() - p_set.d_f.item() / 2.0) * 10.0

                if p_set.sig_ox.item() > (p_set.d_ox.item() / 2.0):
                    r_pen += 100.0 + (p_set.sig_ox.item() - p_set.d_ox.item() / 2.0) * 10.0

                return float((loss + penalty + r_pen).item())

        except Exception:
            return 1e9

    sampler = qmc.LatinHypercube(d=total_dim)
    init_pop = sampler.random(n=popsize)
    init_pop[0] = np.array(opt_guess_norm)

    result = differential_evolution(
        objective,
        bounds=[(0, 1)] * total_dim,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        mutation=(0.5, 1.0),
        recombination=0.7,
        init=init_pop,
        disp=False,
        workers=1,
    )

    final_dict = {k: v.cpu() for k, v in fixed_params.items()}
    cursor = 0
    for info in opt_param_info:
        phys_val = (
            result.x[cursor : cursor + info["size"]] * (info["max"] - info["min"]) + info["min"]
        )
        final_dict[info["key"]] = torch.tensor(phys_val, dtype=torch.float32).view(1, info["size"])
        cursor += info["size"]

    return ParamSet(**final_dict)


def three_stage_refinement(model, p_ai: ParamSet, log_r_obs: torch.Tensor, config: XRefineConfig):
    """ALD-specific multi-stage refinement."""
    Q_CUTOFF = 0.45  # high-q region above this is too noisy

    # critical angle taken from the position of the reflectivity peak
    r_obs_linear = 10 ** log_r_obs.squeeze().cpu().numpy()
    q_grid = config.q_grid.numpy()
    peak_idx = np.argmax(r_obs_linear)
    q_critical = q_grid[peak_idx]
    logger.info(f"📍 임계각 위치 감지: q_c = {q_critical:.4f} Å⁻¹ (index {peak_idx})")

    logger.info("👣 [0단계] 풋프린트 & Background (Q < 0.15)...")
    p_stage0 = _optimize_subset(
        model,
        p_ai,
        log_r_obs,
        config,
        optimize_keys={"beam_width", "L", "i0", "bkg"},
        q_max=0.15,
        maxiter=50,
        popsize=20,
    )

    # 1D grid scan instead of gradient descent to reliably find the global minimum; metric is 1 -
    # correlation (pattern matching)
    logger.info("📐 [1단계] Thickness Grid Search (Correlation 기반, 10~1200A)...")

    d_grid = np.linspace(20, 1200, 500)  # 20 to 1200 A, about 2.4 A step
    best_corr_loss = 1e9
    best_d = p_stage0.d_f.item()

    p_scan = deepcopy(p_stage0)
    # remove oxide and roughness to see only the main fringe period
    p_scan.d_ox.fill_(0.0)
    p_scan.sig_ox.fill_(0.0)
    p_scan.rho_ox.fill_(0.0)
    p_scan.sig_f.fill_(2.0)

    # Q-Mask for Correlation (0.05 ~ cutoff)
    q_grid_gpu = config.q_grid.to(config.device)
    log_r_obs_gpu = log_r_obs.to(config.device)
    mask_corr = (log_r_obs_gpu > -14.0) & (q_grid_gpu > 0.05) & (q_grid_gpu < Q_CUTOFF)
    target_pattern = log_r_obs_gpu[mask_corr]
    target_mean = target_pattern.mean()

    scores = []

    for d_val in d_grid:
        p_scan.d_f.fill_(d_val)

        with torch.no_grad():
            _, r_sim, _ = model.forward_with_params(p_scan)
            r_sim_norm = r_sim / (r_sim.max() + 1e-12)
            log_r_sim = torch.log10(torch.clamp(r_sim_norm, min=1e-12))

            sim_pattern = log_r_sim[mask_corr]

            # Pearson Correlation Calculation
            vx = sim_pattern - sim_pattern.mean()
            vy = target_pattern - target_mean
            corr = torch.sum(vx * vy) / (
                torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-9
            )

            # Loss = 1 - Correlation
            c_loss = 1.0 - corr.item()
            scores.append(c_loss)

            if c_loss < best_corr_loss:
                best_corr_loss = c_loss
                best_d = d_val

    # [Smart Selection] Valley Detection to find true minima
    peaks, _ = find_peaks(-np.array(scores), prominence=0.005)

    # minimum thickness from the critical angle: artifacts empirically appear below 2.0/Qc (e.g.
    # Qc=0.04 -> 50 A)
    min_d_threshold = 2.0 / (q_critical + 1e-9)
    logger.info(
        f"🛡️ Dynamic Thickness Filter: d < {min_d_threshold:.1f}A 제외 (based on Qc={q_critical:.4f})"
    )

    candidates = []
    for idx in peaks:
        d_cand = d_grid[idx]
        loss_cand = scores[idx]

        if d_cand < min_d_threshold:
            logger.info(f"⚠️ 후보 제외 (Too Thin): {d_cand:.1f}A (Loss={loss_cand:.4f})")
            continue

        candidates.append((d_cand, loss_cand))

    if not candidates:
        logger.warning("🚫 적절한 Local Minima를 찾지 못함. Global Min 사용.")
        final_best_d = best_d
    else:
        candidates.sort(key=lambda x: x[1])
        final_best_d = candidates[0][0]
        logger.info(f"✅ 후보군 Top 3: {[f'{d:.1f}A({l:.4f})' for d, l in candidates[:3]]}")

    logger.info(f"🎯 최종 선택: d_f = {final_best_d:.1f} A")

    p_stage1 = deepcopy(p_stage0)
    p_stage1.d_f.fill_(final_best_d)
    # oxide and roughness are zero here and restored in stage 2
    p_stage1.d_ox.fill_(0.0)
    p_stage1.sig_ox.fill_(0.0)
    p_stage1.rho_ox.fill_(0.0)
    p_stage1.sig_f.fill_(3.0)

    logger.info(f"� [2단계] Linear R 피팅 (Fringe 진폭 맞추기, d_f={final_best_d:.1f}A 고정)...")

    p_combined = p_stage1
    p_combined.sig_f = torch.tensor([[3.0]], device=p_ai.d_f.device)
    p_combined.d_ox = p_ai.d_ox
    p_combined.sig_ox = p_ai.sig_ox
    p_combined.rho_ox = p_ai.rho_ox

    fixed_stage2 = {"d_f": final_best_d}

    p_stage2 = _optimize_subset(
        model,
        p_combined,
        log_r_obs,
        config,
        optimize_keys={"sig_f", "rho_f", "d_ox", "sig_ox", "rho_ox", "L", "i0", "bkg", "dq"},
        forced_params=fixed_stage2,
        q_max=Q_CUTOFF,
        use_linear_loss=True,
        maxiter=100,
        popsize=30,
    )

    logger.info("🔬 [3단계] Log R 미세 조정...")

    p_stage3 = _optimize_subset(
        model,
        p_stage2,
        log_r_obs,
        config,
        optimize_keys={"sig_f", "rho_f", "d_ox", "sig_ox", "rho_ox", "L", "i0", "bkg", "dq"},
        forced_params=fixed_stage2,
        q_max=Q_CUTOFF,
        use_linear_loss=False,
        maxiter=100,
        popsize=30,
    )

    logger.info(f"🎯 [4단계] 두께 미세 조정 (d_f ≈ {final_best_d:.1f} ± 50A)...")

    # temporarily narrow the d_f range in config
    original_d_range = config.ranges.d_f
    config.ranges.d_f = (max(10.0, final_best_d - 50.0), final_best_d + 50.0)

    p_final = _optimize_subset(
        model,
        p_stage3,
        log_r_obs,
        config,
        optimize_keys={"d_f", "sig_f", "rho_f", "d_ox", "sig_ox", "rho_ox", "L", "i0", "bkg", "dq"},
        q_max=Q_CUTOFF,
        use_linear_loss=False,
        maxiter=50,
        popsize=20,
    )

    config.ranges.d_f = original_d_range

    return p_final


@torch.no_grad()
def run_real_inference(checkpoint_path: str, data_path: str, output_dir: str = "./results"):
    ckpt = load_checkpoint(checkpoint_path)

    # Load Config from Checkpoint but OVERRIDE ranges from current code
    # (Checking point may have old narrow ranges like sig_f < 5.0)
    loaded_config_dict = ckpt["config"]
    config = XRefineConfig(**loaded_config_dict)

    # use the checkpoint config as is, including ranges

    model = get_model(config).to(config.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    data_name = Path(data_path).stem

    input_tensor = load_real_data(data_path, config)
    p_ai, r_ai, _ = model(input_tensor.to(config.device))

    p_refined = three_stage_refinement(model, p_ai, input_tensor, config)

    p_final, r_final, _ = model.forward_with_params(p_refined.to(config.device))

    r_final_norm = r_final / (r_final.max() + 1e-12)
    r_ai_norm = r_ai / (r_ai.max() + 1e-12)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / f"{data_name}_fit_{datetime.now().strftime('%H%M%S')}.png"

    _save_plot(
        config,
        input_tensor[0],
        r_ai_norm[0],
        r_final_norm[0],
        data_name,
        calculate_r2(
            input_tensor.to(config.device).squeeze(),
            torch.log10(torch.clamp(r_final_norm, min=1e-12)).squeeze(),
            (input_tensor.to(config.device) > -14).squeeze(),
        ),
        fig_path,
    )

    logger.info(f"✨ 완료. 결과 저장됨: {fig_path}")
    print("Refined Params:", p_final)


def _save_plot(config, y_meas_log, r_ai, r_ref, title, r2, save_path):
    q = config.q_grid.numpy()
    y_meas = y_meas_log.cpu().numpy()

    valid_mask = y_meas > -14.0
    q_valid = q[valid_mask]
    y_valid = 10 ** y_meas[valid_mask]
    r_ai_valid = r_ai.cpu().numpy()[valid_mask]
    r_ref_valid = r_ref.cpu().numpy()[valid_mask]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(q_valid, y_valid, "ko", ms=2, alpha=0.3, label="Measured")
    ax.semilogy(q_valid, r_ai_valid, "b--", alpha=0.6, label="AI Init")
    ax.semilogy(q_valid, r_ref_valid, "r-", lw=2, label=f"Refined (R²={r2:.3f})")
    ax.set_xlabel("Q (Å⁻¹)")
    ax.set_ylabel("Reflectivity")
    ax.set_title(f"XRefine: {title}")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    CHECKPOINT = Path(os.environ["DATA"], r"XRR_AI\xrefine\v1\best.pt")
    # DATA_FILE = Path(os.environ["DATA"], r"XRR_AI\XRR_data\KBJ\ALD_A_no_XRR.dat")
    DATA_FILE = Path(os.environ["DATA"], r"XRR_AI\XRR_data\XRR_data\#1_xrr.dat")

    run_real_inference(CHECKPOINT, DATA_FILE)
