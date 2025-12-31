import torch
from tqdm import tqdm
from pathlib import Path
from xrefine.models import XRR_Fourier_PINN
from xrefine.data.generator import generate_synthetic_batch
from xrefine.data.augmentations import XRRAugmentations
from xrefine.config import CONFIG, XRefineConfig

from xrefine.train_utils import compute_xrefine_loss, save_pinn_report

def train_streaming(config: XRefineConfig, steps: int = None):
    device = config.device
    save_dir = Path(config.save_root) / config.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(save_dir / "config.yaml")

    # 1. 모델 및 최적화
    model = XRR_Fourier_PINN(config=config, device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    total_steps = steps if steps is not None else config.train_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # 2. 증강 및 그리드
    q_grid = config.q_grid.to(device)
    augmenter = XRRAugmentations(delta_q=0.0005, do_footprint=True)

    loss_history = []
    best_loss = float('inf')
    pbar = tqdm(range(total_steps), desc="🚀 PINN Training")

    model.train()
    for step in pbar:
        # [A] Curriculum Weight (Linear Decay)
        ratio = min(step / config.warmup_steps, 1.0)
        curr_sup_w = config.sup_w_start * (1 - ratio) + config.sup_w_end * ratio

        # [B] 데이터 생성
        r_measured, params_true = generate_synthetic_batch(
            batch_size=config.batch_size, q_grid=q_grid, config=config, augmenter=augmenter
        )
        target_log_r = torch.log10(torch.clamp(r_measured, min=1e-9))

        # [C] Forward & Loss (중앙화된 함수 사용)
        optimizer.zero_grad(set_to_none=True)
        params_pred, r_recon = model(target_log_r)
        
        total_loss, l_rec, l_par, l_L = compute_xrefine_loss(
            params_pred, r_recon, target_log_r, params_true, config, 1.0
        )

        # [D] Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()

        # [E] 모니터링
        loss_history.append(total_loss.item())
        if step % 100 == 0:
            with torch.no_grad():
                d_f, _, rho_f, _, _, L, _ = params_pred
                t_t, _, s_t = params_true
                mae_t = torch.mean(torch.abs(d_f - t_t[:, 0:1])).item()
                mae_s = torch.mean(torch.abs(rho_f - s_t[:, 1:2])).item()
            
            pbar.set_description(f"R_loss:{l_rec.item():.3f} | d_err:{mae_t:.2f}Å | s_err:{mae_s:.2f} | L:{L.mean().item():.2f}")
            
            # 리포트 저장
            save_pinn_report(
                q_grid, r_measured[0:1], r_recon[0:1], loss_history, 
                params_pred, params_true, save_dir / "convergence.png", step
            )

            # 체크포인트
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                torch.save({'model_state_dict': model.state_dict(), 'config': config.model_dump()}, save_dir / "best.pt")

    print(f"✅ Training Complete. Best Loss: {best_loss:.6f}")

if __name__ == "__main__":
    train_streaming(CONFIG)