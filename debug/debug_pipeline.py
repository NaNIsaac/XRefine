import torch

from xrefine.config import CONFIG
from xrefine.data.augmentations import XRRAugmentations
from xrefine.data.generator import generate_synthetic_batch
from xrefine.models import XRR_Fourier_PINN


def debug():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = CONFIG
    config.n_layers = 1
    config.batch_size = 5

    print("\n[1] Testing Generator...")
    q_grid = config.q_grid.to(device)

    aug_cfg = config.augmentations
    augmenter = XRRAugmentations(
        delta_q=0.0005, intensity_noise_scale=aug_cfg.intensity_noise_scale, do_footprint=True
    )

    r, params = generate_synthetic_batch(5, q_grid, config, augmenter=augmenter)

    print(f"R shape: {r.shape}")
    print(f"R min/max/mean: {r.min().item():.2e}, {r.max().item():.2e}, {r.mean().item():.2e}")
    print(f"R std: {r.std().item():.2e}")

    if r.std() < 1e-9:
        print("❌ CRITICAL: Generator output is constant!")
    else:
        print("✅ Generator output varies.")

    t, rough, sld = params
    print(f"Thickness samples: {t.flatten().cpu().numpy()}")

    print("\n[2] Testing Model Forward...")
    model = XRR_Fourier_PINN(config=config, device=device).to(device)

    # Input logR
    inp = torch.log10(r + 1e-12)
    pred_params, r_recon = model(inp)

    pt, pr, ps = pred_params
    print(f"Pred Thickness: {pt.flatten().detach().cpu().numpy()}")

    if pt.std() < 1e-5:
        print(
            "⚠️ Model prediction is initially constant (Expected for untrained, but should vary slightly if initialized randomly?)"
        )
        # Kaiming init should give some variance.
    else:
        print("✅ Model prediction varies with input.")

    print("\n[3] Testing Gradient Flow...")
    loss_fn = torch.nn.MSELoss()
    target = torch.log10(r + 1e-12)  # Autoencoder style for test

    loss = loss_fn(torch.log10(r_recon + 1e-12), target)
    print(f"Initial Loss: {loss.item()}")

    loss.backward()

    grad_norms = []
    for _name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())

    print(f"Gradient Norms (Max): {max(grad_norms) if grad_norms else 0}")

    if max(grad_norms) == 0:
        print("❌ CRITICAL: Gradients are ZERO!")
    else:
        print("✅ Gradients exist.")


if __name__ == "__main__":
    debug()
