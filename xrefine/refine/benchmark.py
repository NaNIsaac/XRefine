"""
Refinement Benchmark (Updated for Dynamic Params)
=================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from xrefine.checkpoint import load_checkpoint
from xrefine.config import XRefineConfig
from xrefine.core.abeles import AbelesMatrix
from xrefine.core.structures import ParamSet
from xrefine.data.generator import generate_synthetic_batch
from xrefine.logger import logger
from xrefine.models import get_model
from xrefine.refine.config import DEFAULT_CONFIG
from xrefine.refine.pipeline import run_refinement


def run_benchmark(
    checkpoint_path: str,
    target_params: dict[str, float] | None = None,
    n_random: int = 3,
    save_dir: str = "./benchmark_results",
):
    # 1. Load Setup
    ckpt = load_checkpoint(checkpoint_path)
    config_data = ckpt.get("config")
    if isinstance(config_data, dict):
        config = XRefineConfig(**config_data)
    else:
        config = config_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
    config.q_grid = config.q_grid.to(device)

    model = get_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    simulator = AbelesMatrix(device=device)
    # Inject simulator for optimize_step fallback
    model.simulator = simulator

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    # 2. Generate Cases
    test_cases = []

    # (A) Manual Case
    if target_params:
        # Generate random base then override
        _, p_base = generate_synthetic_batch(1, config.q_grid, config, simulator)

        # Override values
        for k, v in target_params.items():
            if k in p_base._params:
                p_base._params[k].fill_(v)

        # Simulate R
        t, r, s = p_base.assemble_structure()
        r_sim = simulator(config.q_grid, t, r, s)

        # Add Noise
        r_noisy = r_sim * (1 + 0.05 * torch.randn_like(r_sim))
        log_r = torch.log10(torch.clamp(r_noisy, min=1e-12))

        test_cases.append(("Manual", p_base, log_r))

    # (B) Random Cases
    if n_random > 0:
        r_batch, p_batch = generate_synthetic_batch(n_random, config.q_grid, config, simulator)
        log_r_batch = torch.log10(torch.clamp(r_batch, min=1e-12))

        for i in range(n_random):
            # Extract single item ParamSet
            single_dict = {k: v[i : i + 1] for k, v in p_batch._params.items()}
            p_single = ParamSet(single_dict, p_batch.layer_names, device)
            r_single = log_r_batch[i : i + 1]
            test_cases.append((f"Random_{i}", p_single, r_single))

    # 3. Run Benchmark
    for name, p_true, log_r in test_cases:
        logger.info(f"📦 Benchmarking: {name}")

        # AI Guess
        out = model(log_r, fixed_values=None)
        p_ai = out[0] if isinstance(out, tuple) else out

        # Refine
        p_ref = run_refinement(model, p_ai, log_r, config, DEFAULT_CONFIG)

        # Compare (Target Film only for brevity)
        t_thick = p_true._params["Target_Film.thickness"].item()
        p_thick = p_ref._params["Target_Film.thickness"].item()
        err = abs(p_thick - t_thick) / t_thick * 100

        logger.info(f"Result: True={t_thick:.1f}, Refined={p_thick:.1f}, Err={err:.2f}%")

        # Plot
        t, r, s = p_ref.assemble_structure()
        r_final = simulator(config.q_grid, t, r, s)

        plt.figure(figsize=(8, 5))
        q = config.q_grid.cpu().numpy()
        plt.semilogy(q, 10 ** log_r[0].cpu().numpy(), "k.", alpha=0.3)
        plt.semilogy(q, r_final[0].detach().cpu().numpy(), "r-")
        plt.title(f"{name}: Err {err:.1f}%")
        plt.savefig(save_path / f"{name}.png")
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XRefine Benchmark Tool")
    parser.add_argument("-c", "--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("-n", "--n_random", type=int, default=3, help="Number of random test cases")
    parser.add_argument("-o", "--output", default="./benchmark_results", help="Output directory")

    args = parser.parse_args()

    # [UPDATE] 복소수(흡수율)가 포함된 테스트 케이스 정의
    # 키 이름은 config.yaml의 레이어 이름과 일치해야 함 (동적 파라미터 구조 반영)
    manual_case = {
        "Target_Film.thickness": 354.0,
        "Target_Film.roughness": 5.0,
        "Target_Film.sld": 49.6,
        "Target_Film.sld_imag": 2.5,  # [New] 흡수율 테스트
        "Top_Layer.thickness": 18.3,
        "Top_Layer.sld": 14.1,
        "Top_Layer.sld_imag": 0.1,  # [New] 미세 흡수
    }

    run_benchmark(
        args.checkpoint, target_params=manual_case, n_random=args.n_random, save_dir=args.output
    )
