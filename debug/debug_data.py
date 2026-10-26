import torch
from tqdm import tqdm

from xrefine.config import CONFIG
from xrefine.data.generator import generate_synthetic_batch


def debug_data_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🕵️ 데이터 생성기 검증 시작 ({device})")

    for i in tqdm(range(300)):
        r_noisy, _params = generate_synthetic_batch(
            batch_size=CONFIG.batch_size, q_grid=CONFIG.q_grid.to(device), config=CONFIG
        )

        if torch.isnan(r_noisy).any():
            print(f"❌ [배치 {i}] R_noisy에서 NaN 발견!")
            return

        if (r_noisy <= 0).any():
            print(f"⚠️ [배치 {i}] 0 이하의 신호 발견 (Log 취할 때 NaN 위험)")
            print(f"최솟값: {r_noisy.min().item()}")

    print("✅ 데이터 생성기 단계는 안전합니다.")


if __name__ == "__main__":
    debug_data_pipeline()
