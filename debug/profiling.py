import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import ProfilerActivity, profile, record_function

from xrefine.config import CONFIG
from xrefine.core.abeles import AbelesMatrix
from xrefine.data.generator import generate_synthetic_batch
from xrefine.models.fourier_model import FourierConvPINN
from xrefine.train.utils import XRRWeightedLoss


def deep_profiling():
    device = torch.device("cuda")
    batch_size = 64

    # Config 강제 설정
    config = CONFIG
    config.train.batch_size = batch_size
    config.device = device
    config.model.model_type = "fourier_conv"
    config.model.hidden_dim = 64  # 다이어트한 모델

    print(f"🔬 정밀 프로파일링 시작 (Device: {device}, Batch: {batch_size})")
    print("   -> 분석 대상: Data Gen + Forward + Loss + Backward + Optimizer")

    # 1. 컴포넌트 초기화
    model = FourierConvPINN(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = XRRWeightedLoss(config.q_grid).to(device)
    simulator = AbelesMatrix(device=device)
    q_grid = config.q_grid.to(device)

    # 2. Warm-up (초기화 오버헤드 제거)
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

    # 3. 실제 프로파일링
    print("\n📸 Snapshot taking... (Running 5 steps)")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _i in range(5):
            # [A] 데이터 생성 구간
            with record_function("1_Data_Generation"):
                r_meas, p_true = generate_synthetic_batch(batch_size, q_grid, config, simulator)
                log_r_obs = torch.log10(torch.clamp(r_meas, min=1e-15)).float()  # Float32 변환 명시

            # [B] Forward 구간
            with record_function("2_Forward_Pass"):
                p_pred, r_recon, aux = model(log_r_obs)

            # [C] Loss 계산 구간
            with record_function("3_Loss_Calc"):
                loss, _, _, _, _ = criterion(p_pred, r_recon, log_r_obs, p_true, config, 1.0, 0.0)

            # [D] Backward 구간 (여기가 가장 의심됨)
            with record_function("4_Backward_Pass"):
                loss.backward()

            # [E] Optimizer 구간
            with record_function("5_Optimizer"):
                optimizer.step()
                optimizer.zero_grad()

    # 4. 결과 출력
    print("\n" + "=" * 80)
    print("📊 프로파일링 결과 (시간 순 정렬)")
    print("=" * 80)

    # CUDA(GPU) 시간 기준 상위 15개 연산 출력
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print("\n" + "=" * 80)
    print("🧠 CPU 시간 기준 상위 10개 (파이썬 병목 확인)")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


deep_profiling()
