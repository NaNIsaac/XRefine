import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from xrefine.config import CONFIG
from xrefine.models.fourier_model import XRR_Fourier_PINN
from xrefine.data.generator import generate_synthetic_batch
from xrefine.data.augmentations import XRRAugmentations
from xrefine.train_utils import compute_loss

def print_param_set(param_set, title="ParamSet"):
    print(f"--- {title} ---")
    for key, value in param_set.items():
        print(f"{key}: {value.item():5.4f}", end=' | ')
    print("\n------------------\n")

def debug_overfit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Config 통합 및 튜닝
    config = CONFIG.model_copy()
    config.n_layers = 1
    config.batch_size = 1
    config.lr = 5e-4  # 오버피팅을 위해 조금 낮춤 (Fourier 모델 안정성)
    config.device = device
    
    # Sample Length 구속을 위한 가중치 설정 (가장 중요)
    config.sld_weight_factor = 5.0  # SLD 중요도 상승
    # 주의: CONFIG 객체에 L_weight가 없다면 compute_loss 내부에서 
    # 별도 가중치를 곱하는 로직을 통합해야 함. (아래 3번 참고)

    # 2. 고정된 샘플 생성
    q_grid = config.q_grid.to(device)
    torch.manual_seed(42)

    augmenter = XRRAugmentations(delta_q=0.0005, do_footprint=True)
    r_target, params_true = generate_synthetic_batch(1, q_grid, config, augmenter=augmenter)

    print_param_set(params_true, title="Target Params")

    model = XRR_Fourier_PINN(config=config, device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    
    input_log_r = torch.log10(torch.clamp(r_target, min=1e-9))
    
    losses = []
    model.train()
    
    print("🚀 Starting Unified Overfit Test...")
    pbar = tqdm(range(2000))
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)

        # PINN Forward
        pred_params, r_recon, phys_penalty = model(input_log_r)

        total_loss, l_recon, l_param, p_phys = compute_loss(
            pred_params, r_recon, input_log_r, 
            params_true, config, curr_sup_w=100.0, phys_penalty=phys_penalty
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        losses.append(total_loss.item())
        pbar.set_postfix({
            'Loss': f"{total_loss.item():.4f}", 
            'Recon': f"{l_recon.item():.4f}",
        })

    # 4. 결과 시각화
    model.eval()
    with torch.no_grad():
        final_params, final_r, _ = model(input_log_r)

    print("\n✅ [FINAL RESULTS]")
    print_param_set(final_params, title="Predicted Params")
    print_param_set(params_true, title="Target Params")
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.semilogy(q_grid.cpu(), r_target.cpu().flatten(), 'k.', alpha=0.3, label='Target')
    ax1.semilogy(q_grid.cpu(), final_r.cpu().flatten(), 'r-', lw=2, label='PINN Fit')
    ax1.set_title("Reflectivity Overfit")
    ax1.legend()
    
    ax2.plot(losses)
    ax2.set_yscale('log')
    ax2.set_title("Total Loss Convergence")
    
    plt.tight_layout()
    plt.savefig('debug_overfit.png')
    print("📊 Saved debug_overfit.png")

if __name__ == "__main__":
    debug_overfit()