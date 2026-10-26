import torch
from torch import optim
from torch.profiler import ProfilerActivity, profile, record_function

from xrefine.config import CONFIG
from xrefine.core.abeles import AbelesMatrix
from xrefine.data.generator import generate_synthetic_batch
from xrefine.models.fourier_model import FourierConvPINN
from xrefine.train.utils import XRRWeightedLoss


def deep_profiling():
    device = torch.device("cuda")
    batch_size = 64

    config = CONFIG
    config.train.batch_size = batch_size
    config.device = device
    config.model.model_type = "fourier_conv"
    config.model.hidden_dim = 64

    print(f"🔬 정밀 프로파일링 시작 (Device: {device}, Batch: {batch_size})")
    print("   -> 분석 대상: Data Gen + Forward + Loss + Backward + Optimizer")

    model = FourierConvPINN(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = XRRWeightedLoss(config.q_grid).to(device)
    simulator = AbelesMatrix(device=device)
    q_grid = config.q_grid.to(device)

    print("🔥 Warming up (3 steps)...")
    for _ in range(3):
        r_meas, p_true = generate_synthetic_batch(batch_size, q_grid, config, simulator)
        log_r_obs = torch.log10(torch.clamp(r_meas, min=1e-15)).float()
        p_pred, r_recon, _ = model(log_r_obs)
        loss, _, _, _, _ = criterion(p_pred, r_recon, log_r_obs, p_true, config, 1.0, 0.0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    print("\n📸 Snapshot taking... (Running 5 steps)")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _i in range(5):
            with record_function("1_Data_Generation"):
                r_meas, p_true = generate_synthetic_batch(batch_size, q_grid, config, simulator)
                log_r_obs = torch.log10(torch.clamp(r_meas, min=1e-15)).float()

            with record_function("2_Forward_Pass"):
                p_pred, r_recon, _aux = model(log_r_obs)

            with record_function("3_Loss_Calc"):
                loss, _, _, _, _ = criterion(p_pred, r_recon, log_r_obs, p_true, config, 1.0, 0.0)

            with record_function("4_Backward_Pass"):
                loss.backward()

            with record_function("5_Optimizer"):
                optimizer.step()
                optimizer.zero_grad()

    print("\n" + "=" * 80)
    print("📊 프로파일링 결과 (시간 순 정렬)")
    print("=" * 80)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print("\n" + "=" * 80)
    print("🧠 CPU 시간 기준 상위 10개 (파이썬 병목 확인)")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


deep_profiling()
