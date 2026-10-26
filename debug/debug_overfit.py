import torch
from tqdm import tqdm

from xrefine.config import CONFIG
from xrefine.data.augmentations import XRRAugmentor
from xrefine.data.generator import generate_synthetic_batch
from xrefine.models.fourier_model import FourierConvPINN
from xrefine.train_utils import XRRWeightedLoss, save_pinn_report


def debug_overfit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🕵️ Debugging Overfit on {device}...")

    config = CONFIG.model_copy()
    config.model.n_layers = 1
    config.train.batch_size = 1
    config.train.lr = 5e-4
    config.device = str(device)

    # weight SLD more to force overfitting
    config.train.weights.rho_f = 5.0
    config.train.weights.rho_ox = 5.0

    q_grid = config.q_grid.to(device)
    torch.manual_seed(42)

    augmenter = XRRAugmentor(config).to(device)
    r_target, params_true = generate_synthetic_batch(
        batch_size=1, q_grid=q_grid, config=config, augmenter=augmenter
    )

    print(f"\n[TARGET PARAMS]\n{params_true}\n")

    model = FourierConvPINN(config=config, device=str(device)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=1e-4)
    criterion = XRRWeightedLoss(q_grid)

    input_log_r = torch.log10(torch.clamp(r_target, min=1e-11))

    losses = []
    model.train()

    print("🚀 Starting PINN Overfit Test (2000 steps)...")
    pbar = tqdm(range(2000))
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)

        # Forward
        params_pred, r_recon, phys_penalty = model(input_log_r)

        total_loss, l_recon, _l_param, _p_phys = criterion(
            params_pred,
            r_recon,
            input_log_r,
            params_true,
            config,
            curr_sup_w=100.0,
            phys_penalty=phys_penalty,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        losses.append(total_loss.item())
        if step % 10 == 0:
            pbar.set_postfix({"Loss": f"{total_loss.item():.6f}", "Recon": f"{l_recon.item():.6f}"})

    model.eval()
    with torch.no_grad():
        final_params, final_r, _ = model(input_log_r)

    print("\n✅ [OVERFIT RESULTS]")
    print(f"PRED: {final_params}")
    print(f"TRUE: {params_true}")

    save_pinn_report(
        q_grid=q_grid,
        target_log_r=input_log_r,
        r_recon=final_r,
        loss_history=losses,
        params_pred=final_params,
        params_true=params_true,
        save_path="debug_overfit_pinn.png",
        step=2000,
    )

    print("📊 Visualization saved to debug_overfit_pinn.png")


if __name__ == "__main__":
    debug_overfit()
