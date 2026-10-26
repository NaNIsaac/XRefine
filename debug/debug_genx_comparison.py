import matplotlib.pyplot as plt
import numpy as np
import torch
from genx.models.spec_nx import Instrument, Layer, Sample, Specular, Stack

from xrefine.core.interface import Simulator

# Constants & Helpers
r_e = 2.8179403227e-5  # Angstroms


def get_f(sld_1e6: float) -> complex:
    """Convert SLD (10^-6 A^-2) to Scattering Factor (f) assuming dens=1.0"""
    return complex((sld_1e6 * 1e-6) / r_e, 0)


def make_genx_layer(d, sig, sld):
    return Layer(d=d, f=get_f(sld), dens=1.0, sigma=sig)


def run_genx_sim(qs, film_d, film_r, film_sld):

    # Ambient: Air (SLD=0)
    amb = Layer(d=0, f=complex(0, 0), dens=1.0, sigma=0)

    # Substrate: Si (Roughness 3.0, SLD 20.1)
    sub = make_genx_layer(d=0, sig=3.0, sld=20.1)

    # Surface SiO2 (Thickness 20.0, Roughness 3.0, SLD 18.0? SiO2 is ~18-19)
    sio2_layer = make_genx_layer(d=20.0, sig=3.0, sld=18.0)

    # Main Film (Variable)
    film_layer = make_genx_layer(d=film_d, sig=film_r, sld=film_sld)

    # Stack: SiO2 (Top) -> Film (Bottom)
    stack = Stack(Layers=[sio2_layer, film_layer], Repetitions=1)

    sample = Sample(Stacks=[stack], Ambient=amb, Substrate=sub)

    # Instrument
    # GenX spec_nx default assumes x is TwoTheta (degrees).
    inst = Instrument(wavelength=1.54, iback=0, i0=1.0, res=0)

    # Convert Q to TwoTheta
    # Q = 4pi sin(theta) / lambda
    # sin(theta) = Q * lambda / (4pi)
    wavelength = 1.54
    sin_theta = qs * wavelength / (4 * np.pi)
    # Clip for safety
    sin_theta = np.clip(sin_theta, 0, 1)
    theta_rad = np.arcsin(sin_theta)
    tth_deg = np.degrees(2 * theta_rad)

    refl = Specular(tth_deg, sample, inst)
    return refl


def run_xrefine_sim(qs, film_d, film_r, film_sld):
    device = "cpu"
    sim = Simulator(method="abeles", device=device)

    sld_tensor = torch.tensor([[0.0, 18.0, film_sld, 20.1]], device=device)  # [1, 4]

    thick_tensor = torch.tensor([[20.0, film_d]], device=device)  # [1, 2]

    rough_tensor = torch.tensor([[3.0, 3.0, 3.0]], device=device)  # [1, 3]

    q_tensor = torch.from_numpy(qs).float().view(1, -1).to(device)

    r_sim = sim(q_tensor, thick_tensor, rough_tensor, sld_tensor)
    return r_sim.squeeze().numpy()


def main():
    print("Running GenX vs XRefine Comparison (Simplified)...")

    qs = np.linspace(0.01, 0.5, 500)

    # Test Case: Single Layer on Substrate (No Roughness)
    d_val = 500.0
    r_val = 0.0  # No roughness first
    sld_film = 10.0
    sld_sub = 20.0

    # GenX: Amb -> Film -> Sub
    film_layer = make_genx_layer(d=d_val, sig=r_val, sld=sld_film)
    # Reuse previous helpers but simpler stack

    # Re-define genx runner for single layer
    amb = Layer(d=0, f=complex(0, 0), dens=1.0, sigma=0)
    sub = make_genx_layer(d=0, sig=0, sld=sld_sub)
    stack = Stack(Layers=[film_layer], Repetitions=1)
    sample = Sample(Stacks=[stack], Ambient=amb, Substrate=sub)
    inst = Instrument(wavelength=1.54, iback=0, i0=1.0, res=0)

    # Unit Conversion: Q (A^-1) -> TwoTheta (deg)
    # Q = 4pi sin(theta) / lambda
    wavelength = 1.54
    sin_theta = qs * wavelength / (4 * np.pi)
    sin_theta = np.clip(sin_theta, 0, 1)
    tth_deg = np.degrees(2 * np.arcsin(sin_theta))

    r_genx = Specular(tth_deg, sample, inst)

    # XRefine: Amb -> Film -> Sub
    # Thick: [500]
    # Rough: [0, 0] (Amb/Film, Film/Sub)
    # SLD: [0, 10, 20]

    thick_t = torch.tensor([[d_val]], dtype=torch.float64)
    rough_t = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    sld_t = torch.tensor([[0.0, sld_film, sld_sub]], dtype=torch.float64)
    q_t = torch.from_numpy(qs).double().view(1, -1)

    # We must call internal simulator directly to control tensors precisely
    # passing tensors to internal `abeles` function or use sim forward
    # Sim forward handles device but we are on CPU.

    # Import abeles directly to be sure
    from xrefine.core.abeles import abeles

    r_xref = abeles(q_t, thick_t, rough_t, sld_t).squeeze().numpy()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(qs, r_genx, "b-", label="GenX (Truth)", lw=2, alpha=0.7)
    plt.semilogy(qs, r_xref, "r--", label="XRefine (PINN)", lw=2, alpha=0.7)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.title(f"Physics Benchmark (1 Layer, No Roughness)\nFilm: d={d_val}, sld={sld_film}")
    plt.xlabel("q (A^-1)")
    plt.ylabel("Reflectivity")
    plt.savefig("debug_genx_comparison_simple.png")

    mse = np.mean((np.log10(r_genx + 1e-12) - np.log10(r_xref + 1e-12)) ** 2)
    print(f"Log MSE (1 Layer, No Roughness): {mse:.6e}")

    # Case 2: With Roughness
    r_val_2 = 5.0
    film_layer.sigma = r_val_2
    sub.sigma = r_val_2
    # GenX: Surface roughness is Ambient sigma? No, Top Layer sigma?
    # In GenX, Layer sigma is roughness of interface ON TOP of layer? Or below?
    # GenX spec_nx: sigma is roughness of the interface between this layer and the one ABOVE it.
    # Checks docs/source if possible. Common convention: sigma is Top interface.
    # Ambient sigma? Usually 0.

    # Let's set ALL roughness to 5.0
    # Amb layer (sigma=5? Amb is top. Amb has no interface above it.)

    # GenX:
    # Amb -> L1 (sigma is Amb/L1 interface?)
    # Let's assume GenX Layer.sigma is interface roughness between Previous and Current.

    # Let's refine based on results.

    r_genx_rough = Specular(tth_deg, sample, inst)

    rough_t_2 = torch.tensor([[r_val_2, r_val_2]], dtype=torch.float64)  # Amb/Film, Film/Sub
    r_xref_rough = abeles(q_t, thick_t, rough_t_2, sld_t).squeeze().numpy()

    plt.figure(figsize=(10, 6))
    plt.semilogy(qs, r_genx_rough, "b-", label="GenX (Rough)", lw=2, alpha=0.7)
    plt.semilogy(qs, r_xref_rough, "r--", label="XRefine (Rough)", lw=2, alpha=0.7)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.title(f"Physics Benchmark (1 Layer, Rough={r_val_2})")
    plt.savefig("debug_genx_comparison_rough.png")

    mse_rough = np.mean((np.log10(r_genx_rough + 1e-12) - np.log10(r_xref_rough + 1e-12)) ** 2)
    print(f"Log MSE (With Roughness): {mse_rough:.6e}")


if __name__ == "__main__":
    main()
