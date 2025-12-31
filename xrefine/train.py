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
    actual_warmup = min(config.warmup_steps, int(total_steps * 0.5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # 2. 증강기 초기화
    q_grid = config.q_grid.to(device)
    delta_q = (q_grid[1] - q_grid[0]).item()
    aug_cfg = config.augmentations
    base_augmenter = XRRAugmentations(
        delta_q=delta_q,
        intensity_noise_scale=aug_cfg.intensity_noise_scale,
        bg_range=aug_cfg.bg_range,
        res_sigma_range=aug_cfg.res_sigma_range,
        q_shift_sigma=aug_cfg.q_shift_sigma,
        beam_profile=aug_cfg.beam_profile,
        do_footprint=True
    )

    # [핵심 해결] 가변 인자 어댑터: 시뮬레이터(4~5개)와 증강기(2개) 호출을 모두 대응
    def universal_adapter(*args, **kwargs):
        """
        generator.py의 두 가지 호출 방식에 모두 대응:
        1. 시뮬레이터 모드: (q, t, r, s, dq=...) -> 4~5개 인자
        2. 증강기 모드: (q, r) -> 2개 인자
        """
        # XRRAugmentations는 dq 인자를 직접 받지 않으므로 제거
        kwargs.pop('dq', None)
        return base_augmenter(*args, **kwargs)

    # 3. 고정 Validation 세트 생성
    print("🧪 Generating fixed validation batch...")
    with torch.no_grad():
        v_r, v_params = generate_synthetic_batch(
            batch_size=config.batch_size, q_grid=q_grid, config=config, augmenter=universal_adapter
        )
        v_log_r = torch.log10(torch.clamp(v_r, min=1e-9)).detach()

    loss_history = []
    history = {'train': [], 'val': []}
    best_val_loss = float('inf')
    pbar = tqdm(range(total_steps), desc="🚀 PINN Training")

    model.train()
    for step in pbar:
        # [A] Curriculum Weight 계산 (Linear Decay)
        # 초반엔 Label 학습(sup_w_start)에 집중하여 뼈대를 잡습니다.
        ratio = min(step / actual_warmup, 1.0)
        curr_sup_w = config.sup_w_start * (1 - ratio) + config.sup_w_end * ratio

        # [B] 데이터 실시간 생성
        r_measured, params_true = generate_synthetic_batch(
            batch_size=config.batch_size, q_grid=q_grid, config=config, augmenter=universal_adapter
        )
        target_log_r = torch.log10(torch.clamp(r_measured, min=1e-9))

        # [C] Forward & Loss (curr_sup_w 실제 적용)
        optimizer.zero_grad(set_to_none=True)
        params_pred, r_recon = model(target_log_r)
        
        # [사용자 피드백 반영] 6번째 인자로 curr_sup_w를 명시적으로 전달
        total_loss, l_rec, l_par, l_L = compute_xrefine_loss(
            params_pred, r_recon, target_log_r, params_true, config, curr_sup_w
        )

        if step == actual_warmup:
            print(f"\n🎓 Phase Shift: Supervised -> Physics-Informed (Step {step})")

        # [D] Backward & Clipping
        total_loss.backward()
        # NaN 방지를 위해 0.1로 타이트하게 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        scheduler.step()

        # [E] 모니터링
        train_loss = total_loss.item()
        loss_history.append(train_loss)
        history['train'].append(train_loss)

        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                # Validation Loss 계산
                v_params_pred, v_r_recon = model(v_log_r)
                val_loss_total, _, _, _ = compute_xrefine_loss(
                    v_params_pred, v_r_recon, v_log_r, v_params, config, curr_sup_w
                )
                history['val'].append(val_loss_total.item())
                
                # [사용자 피드백 반영] mae_s(SLD 오차) 모니터링 활성화
                d_f, _, rho_f, _, _, L, _ = params_pred
                t_t, _, s_t = params_true
                mae_t = torch.mean(torch.abs(d_f - t_t[:, 0:1])).item()
                mae_s = torch.mean(torch.abs(rho_f - s_t[:, 1:2])).item() 
            
            pbar.set_description(
                f"W:{curr_sup_w:.0f} | T_L:{train_loss:.3f} | V_L:{val_loss_total.item():.3f} | d_err:{mae_t:.2f}Å | s_err:{mae_s:.2f}"
            )
            
            # 리포트(이미지) 저장
            save_pinn_report(
                q_grid, r_measured[0:1], r_recon[0:1], loss_history, 
                params_pred, params_true, save_dir / "convergence.png", step
            )

            # Best 체크포인트 저장
            if val_loss_total.item() < best_val_loss:
                best_val_loss = val_loss_total.item()
                checkpoint = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'history': history,
                    'config': config.model_dump()
                }
                torch.save(checkpoint, save_dir / "best.pt")
            
            model.train()

    print(f"✅ Training Complete. Best Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    train_streaming(CONFIG)