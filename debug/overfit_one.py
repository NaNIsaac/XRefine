import torch
from tqdm import tqdm

from xrefine.config import CONFIG
from xrefine.data.generator import generate_synthetic_batch
from xrefine.models.fourier_model import FourierConvPINN
from xrefine.train_utils import compute_loss, save_pinn_report


def test_overfit(steps=1500):
    print("🧪 Overfitting Report: 8-Parameter System (inc. SiO2 SLD)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = CONFIG.model_copy()
    torch.autograd.set_detect_anomaly(True)
    # Initialize Model & Optimizer
    model = FourierConvPINN(config, device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    q_grid = config.q_grid.to(device)

    # Generate 1 Static Target
    # params_true: (thickness, roughness, sld, sample_len)
    r_target, params_true = generate_synthetic_batch(1, q_grid, config)
    target_log_r = torch.log10(torch.clamp(r_target, min=1e-9)).detach()

    loss_history = []
    pbar = tqdm(range(steps))

    for i in pbar:
        optimizer.zero_grad(set_to_none=True)

        # Unpack 3 outputs from PINN
        params_pred, r_recon, phys_penalty = model(target_log_r)

        # Call unified loss from train_utils
        total_loss, l_rec, l_par, l_phys = compute_loss(
            params_pred, r_recon, target_log_r, params_true, config, 1.0, phys_penalty
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        loss_history.append(total_loss.item())
        if i % 10 == 0:
            pbar.set_postfix({"loss": f"{total_loss.item():.6f}", "phys": f"{l_phys.item():.4f}"})

    # Save Visual Report
    save_pinn_report(
        q_grid,
        r_target,
        r_recon,
        loss_history,
        params_pred,
        params_true,
        "debug_overfit_8param.png",
        step=steps,
    )

    # Detailed Console Report

    print("\n" + "═" * 75)
    print(f"{'PHYSICAL PARAMETER':<25} | {'TRUE':<10} | {'PRED':<10} | {'DIFF'}")
    print("─" * 75)

    def rpt(name, t, p):
        t_val = t.item() if isinstance(t, torch.Tensor) else t
        p_val = p.item() if isinstance(p, torch.Tensor) else p
        print(f"{name:<25} | {t_val:<10.2f} | {p_val:<10.2f} | {p_val - t_val:<+10.2f}")

    rpt("Thickness (Film)", t_t[0, 0], d_f)
    rpt("Roughness (Film)", r_t[0, 0], sig_f)
    rpt("SLD (Film)", s_t[0, 1], rho_f)
    rpt("SiO2 Thickness", t_t[0, 1], d_ox)
    rpt("SiO2 SLD", s_t[0, 2], rho_ox)
    rpt("Sample Size (L)", l_t[0, 0], L)
    rpt("Background (log10)", config.augmentations.bg_range[0], log_bkg)
    print("═" * 75)


if __name__ == "__main__":
    test_overfit()
