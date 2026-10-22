import matplotlib.pyplot as plt
import torch

from xrefine.config import XRefineConfig
from xrefine.data.augmentations import XRRAugmentor
from xrefine.data.generator import generate_synthetic_batch


def check_augmentation_severity():
    print("🔍 Checking Augmentation Severity...")

    # 1. Load Config
    config = XRefineConfig.load_yaml("config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device

    # 2. Setup Generator & Augmentor
    q_grid = config.q_grid.to(device)
    augmentor = XRRAugmentor(config).to(device)

    # 3. Generate Clean & Augmented Data
    # generate_synthetic_batch returns augmented data by default if augmenter is passed.
    # We need to access internal logic to get BOTH clean and augmented.
    # Or simply:

    # A. Generate Pure Physics (No Augmentation)
    r_clean, params = generate_synthetic_batch(
        batch_size=4,
        q_grid=q_grid,
        config=config,
        augmenter=None,  # Disable augmentation inside generator
    )

    # B. Apply Augmentation Manually
    r_aug = augmentor(q_grid.expand(4, -1), r_clean.to(device), params.to(device))

    # 4. Visualization
    r_clean_np = r_clean.detach().cpu().numpy()
    r_aug_np = r_aug.detach().cpu().numpy()
    q_np = q_grid.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]

        # Plot Log Scale
        ax.semilogy(q_np, r_clean_np[i], "k-", lw=1.5, alpha=0.6, label="Clean (Physics)")
        ax.semilogy(q_np, r_aug_np[i], "r-", lw=1.0, alpha=0.7, label="Augmented (Realistic)")

        # Calculate Correlation as a metric
        valid = r_aug_np[i] > 1e-10
        if valid.sum() > 10:
            corr = np.corrcoef(np.log10(r_clean_np[i][valid]), np.log10(r_aug_np[i][valid]))[0, 1]
        else:
            corr = 0.0

        ax.set_title(f"Sample #{i} (Corr: {corr:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Mark Noise Floor
        bkg = params.bkg[i].item()
        ax.axhline(10**bkg, color="gray", ls=":", label=f"Bkg: 10^{bkg:.1f}")

    plt.tight_layout()
    plt.savefig("check_aug_severity.png", dpi=150)
    print("✅ Check complete. Saved 'check_aug_severity.png'.")
    print(
        "   If 'Augmented' line is mostly noise (zigzag) and loses the 'Clean' dips, it's too severe."
    )


if __name__ == "__main__":
    import numpy as np  # Import locally if needed

    check_augmentation_severity()
