import torch

from xrefine.core.abeles import abeles


def run_stress_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔍 Testing Abeles Stability on {device}...")

    test_cases = {
        "Normal": {
            "q": torch.linspace(0.01, 0.5, 100),
            "thick": torch.tensor([[100.0]]),
            "rough": torch.tensor([[2.0, 2.0]]),
            "sld": torch.tensor([[0.0, 20.0, 20.0]]),
        },
        "Extreme Roughness": {
            "q": torch.linspace(0.01, 0.5, 100),
            "thick": torch.tensor([[10.0]]),
            "rough": torch.tensor([[50.0, 50.0]]),
            "sld": torch.tensor([[0.0, 60.0, 20.0]]),
        },
        "Low Q": {
            "q": torch.tensor([1e-9, 1e-8, 1e-7]),
            "thick": torch.tensor([[100.0]]),
            "rough": torch.tensor([[2.0, 2.0]]),
            "sld": torch.tensor([[0.0, 20.0, 20.0]]),
        },
    }

    for name, params in test_cases.items():
        q = (
            params["q"].to(device).unsqueeze(0)
            if params["q"].dim() == 1
            else params["q"].to(device)
        )
        t = params["thick"].to(device)
        r = params["rough"].to(device)
        s = params["sld"].to(device)

        try:
            res = abeles(q, t, r, s)
            has_nan = torch.isnan(res).any().item()
            has_inf = torch.isinf(res).any().item()

            print(
                f"[{name}] Result Range: {res.min().item():.3e} ~ {res.max().item():.3e} | NaN: {has_nan} | Inf: {has_inf}"
            )

            if has_nan or has_inf:
                print(f"   ⚠️ Issue detected in {name}!")
        except Exception as e:
            print(f"   ❌ Error in {name}: {e}")


if __name__ == "__main__":
    run_stress_test()
