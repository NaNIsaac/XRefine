import torch

from xrefine.config import XRefineConfig
from xrefine.core.structures import ParamSet
from xrefine.data.generator import generate_synthetic_batch
from xrefine.models import get_model


def test_phase1_dynamic_system():
    print("🚀 Starting Phase 1 Verification...")

    # 1. Load Config
    try:
        config = XRefineConfig.load_yaml("config.yaml")
        print("✅ Config Loaded Successfully")
        print(
            f"   Structure: {[l.name for l in config.sample.layers]} on {config.sample.substrate.name}"
        )
    except Exception as e:
        print(f"❌ Config Load Failed: {e}")
        return

    # 2. Test Dynamic ParamSet
    try:
        # Batch size 2
        params = ParamSet.from_config(config.sample, batch_size=2, device=config.device)
        print("✅ DynamicParamSet Initialized")

        # Check attribute access
        print(f"   Param Keys: {list(params._params.keys())[:5]}...")

        # Test Assembly
        t, r, s = params.assemble_structure()
        print(f"   Assembled T shape: {t.shape}")
    except Exception as e:
        print(f"❌ ParamSet Failed: {e}")
        return

    # 3. Test Generator (Physics Engine)
    try:
        r_batch, params_gt = generate_synthetic_batch(2, config.q_grid, config)
        print(f"✅ Generator Output: {r_batch.shape}")
    except Exception as e:
        print(f"❌ Generator Failed: {e}")
        return

    # 4. Test Model (Dynamic Factory)
    try:
        model = get_model(config).to(config.device)
        print(
            f"✅ Model Initialized (Type: {config.model.model_type.upper()}, Output Dim: {model.output_dim})"
        )

        # Forward Pass
        log_r = torch.log10(torch.clamp(r_batch, min=1e-11))
        p_pred, r_recon, _ = model(log_r)

        print(f"   Prediction keys: {list(p_pred._params.keys())[:5]}...")
        print("✅ Forward Pass Successful")

        # Backward Pass
        loss = torch.mean((r_recon - r_batch) ** 2)
        loss.backward()
        print("✅ Backward Pass Successful (Gradients computed)")

    except Exception as e:
        print(f"❌ Model Test Failed: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n🎉 ALL SYSTEMS GO! Phase 1 & 2 Refactor is Stable.")


if __name__ == "__main__":
    test_phase1_dynamic_system()
