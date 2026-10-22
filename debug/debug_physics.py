import matplotlib.pyplot as plt
import numpy as np
import torch

from xrefine.config import CONFIG
from xrefine.core.interface import Simulator
from xrefine.data.augmentations import XRRAugmentations
from xrefine.models.fourier_model import FourierConvPINN


def debug_physics_mismatch():
    # Setup
    device = "cpu"  # Debug on CPU
    config = CONFIG
    config.n_layers = 1
    config.batch_size = 1

    # 1. Define specific parameters (Ground Truth)
    # [Thick, Rough, SLD]
    # Let's pick a simple case
    thickness = torch.tensor([[500.0]], device=device)
    roughness = torch.tensor([[5.0, 5.0]], device=device)  # Amb/L1, L1/Sub
    sld = torch.tensor([[20.0]], device=device)  # L1 SLD

    # Instrument Params
    s_len_val = 20.0  # Sample Length (mm)
    i0_val = 1.0  # Intensity Scale
    beam_width_val = 0.1  # mm
    wavelength = 1.54
    dq_val = 0.002  # Resolution

    # Q Grid
    q_grid = torch.linspace(0.01, 1.0, 500, device=device)

    # -------------------------------------------------------------
    # A. Generator Path (Numpy / XRRAugmentations)
    # -------------------------------------------------------------
    # Use Generator's logic: Simulator (Pure) -> Augmentations (Footprint+Smearing)

    sim = Simulator(method="abeles", device=device)

    # Param prep for simulator
    # Simulator expects: [B, N], [B, N+1], [B, N+2]
    amb = torch.zeros(1, 1)
    sub = torch.full((1, 1), 20.1)  # Si substrate
    sld_full = torch.cat([amb, sld, sub], dim=-1)

    # Pure Theory
    r_theory = sim(q_grid.unsqueeze(0), thickness, roughness, sld_full)
    r_theory_np = r_theory.squeeze().numpy()
    q_np = q_grid.numpy()

    # Augmenter
    # We want to force specific params to check match,
    # but Augmenter is random. We might need to subclass or mock.
    # Or just use the methods directly.
    aug = XRRAugmentations(
        do_footprint=True,
        delta_q=q_np[1] - q_np[0],
        intensity_noise_scale=0.0,  # Disable noise for comparison
    )

    # Manually Apply Footprint (Numpy)
    theta_deg = np.degrees(np.arcsin(np.clip(q_np * wavelength / (4 * np.pi), 0, 1)))
    footprint_corr = aug._gauss_footprint(theta_deg, beam_w=beam_width_val, sample_len=s_len_val)
    r_gen_foot = r_theory_np * footprint_corr

    # Manually Apply Smearing (Gaussian Filter)
    # sigma_px = res_sigma / delta_q
    # Let's say res_sigma = dq_val
    sigma_px = dq_val / aug.delta_q
    from scipy.ndimage import gaussian_filter1d

    r_gen_final = gaussian_filter1d(r_gen_foot, sigma=sigma_px, mode="nearest")

    # -------------------------------------------------------------
    # B. Model Path (Torch / DifferentiableFootprint)
    # -------------------------------------------------------------
    # Config override
    config.instrument.beam_width = beam_width_val
    config.instrument.res = dq_val

    model = FourierConvPINN(config, device=device)

    # Prep inputs for Model's internal physics engine
    # (thick, rough_full, sld_full)
    # Model forward logic:
    # 1. Simulator with DQ
    # 2. Footprint

    # Need to verify if Model passes dq to simulator now (yes, I added it)

    # Replicate Model's Physics Step
    dq_tensor = torch.full((1, 1), dq_val, device=device)

    # Note: Model forward usually takes PREDICTED params.
    # Here we feed TRUE params to see what the model WOULD see if it predicted perfectly.

    # Simulator Smearing
    r_model_sim = model.simulator(q_grid.unsqueeze(0), thickness, roughness, sld_full, dq=dq_tensor)

    # Footprint
    # We need to pass s_len tensor
    s_len_tensor = torch.tensor([[s_len_val]], device=device)  # Scaled?
    # Wait, model.forward scales s_len: out * 30 + 10.
    # Footprint module assumes physical value inputs?
    # Let's check DifferentiableFootprint.forward

    r_model_final = model.footprint(q_grid.unsqueeze(0), r_model_sim, s_len_tensor)

    r_model_final_np = r_model_final.squeeze().detach().numpy()

    # -------------------------------------------------------------
    # Compare
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.semilogy(q_np, r_theory_np, label="Pure Theory", linestyle=":", color="gray")
    plt.semilogy(q_np, r_gen_final, label="Generator (Numpy)", color="blue")
    plt.semilogy(q_np, r_model_final_np, label="Model Physics (Torch)", color="red", linestyle="--")
    plt.legend()
    plt.title(
        f"Physics Engine Mismatch Check\nThickness={thickness.item()}, Rough={roughness[0, 0].item()}"
    )
    plt.grid(True, which="both")
    plt.savefig("debug_physics_mismatch.png")
    print("Saved debug_physics_mismatch.png")

    # Error metrics
    mse = np.mean((np.log10(r_gen_final + 1e-12) - np.log10(r_model_final_np + 1e-12)) ** 2)
    print(f"Log MSE between Generator and Model Physics: {mse:.6f}")


if __name__ == "__main__":
    debug_physics_mismatch()
