from pathlib import Path
from typing import Optional
import shutil
import time
import os

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from xrefine.config import XRefineConfig
from xrefine.data.generator import generate_synthetic_batch
from xrefine.data.augmentations import XRRAugmentor
from xrefine.train.utils import XRRWeightedLoss, save_pinn_report
from xrefine.core.abeles import AbelesMatrix
from xrefine.logger import logger

# PyTorch 2.9+ 최적화 설정
torch.backends.cudnn.benchmark = True

# Colab 환경 감지
def is_colab() -> bool:
    try:
        import google.colab
        return True
    except ImportError:
        return False

IN_COLAB = is_colab()

# tqdm 설정: 표준 tqdm 사용 (notebook 버전은 loguru와 충돌)
from tqdm import tqdm

# Colab에서 clear_output 사용 가능
if IN_COLAB:
    from IPython.display import clear_output

def log_epoch(msg: str):
    """tqdm과 충돌하지 않게 로그 출력"""
    tqdm.write(msg)

class XRefineTrainer:
    def __init__(self, model, config: XRefineConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        
        # 모델 및 그리드 설정
        self.model = model.to(self.device)
        self.q_grid = config.q_grid.to(self.device)
        
        # 유틸리티 초기화
        self.criterion = XRRWeightedLoss(self.q_grid).to(self.device)
        self.augmenter = XRRAugmentor(config).to(self.device)
        self.simulator = AbelesMatrix(device=self.device)  # Cached Simulator
        
        # 최적화 설정
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.train.lr, 
            weight_decay=config.train.weight_decay
        )
        # 스케줄러
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.train.lr,
            steps_per_epoch=config.train.steps_per_epoch,
            epochs=config.train.epochs,
            pct_start=0.1
        )
        # [FIX] OneCycleLR 경고 방지: dummy optimizer step 수행
        self.optimizer.step()
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None

        # [COLAB OPT] 로컬 저장 경로 vs Drive 백업 경로
        self.drive_save_dir = Path(config.save_root) / config.experiment_name
        if IN_COLAB:
            # Colab: /tmp에 로컬 저장, Drive는 백업용
            self.save_dir = Path("/tmp/xrefine_checkpoints") / config.experiment_name
            self.last_backup_time = time.time()
            self.backup_interval = 30 * 60  # 30분
        else:
            self.save_dir = self.drive_save_dir
            
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.drive_save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "logs"))
        
        # 상태 관리
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0  # Early stopping 카운터
        self.history = {"train_loss": [], "val_loss": [], "lr": [], "sup_weight": []}

    def _get_curriculum_weight(self, epoch: int) -> float:
        max_epochs = self.cfg.train.epochs
        progress = min(epoch / (max_epochs * 0.7), 1.0)
        w_start = self.cfg.train.sup_w_start
        w_end = self.cfg.train.sup_w_end
        return w_start * (1 - progress) + w_end * progress

    def train_epoch(self, epoch: int, curr_sup_w: float):
        self.model.train()
        total_loss = 0.0  # [FIX] 변수명 통일 (기존 total_loss_avg -> total_loss)
        steps = self.cfg.train.steps_per_epoch
        pbar = tqdm(range(steps), desc=f"Epoch {epoch:03d}", leave=False)
        
        for step in pbar:
            # 1. 데이터 생성
            r_meas, p_true = generate_synthetic_batch(
                q_grid=self.cfg.q_grid,
                batch_size=self.cfg.train.batch_size,
                config=self.cfg,
                simulator=self.simulator,  # Cached
                augmenter=self.augmenter
            )
            r_meas = r_meas.to(self.device)
            p_true = p_true.to(self.device)
            
            log_r_obs = torch.log10(torch.clamp(r_meas, min=1e-11))
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                # 2. Forward Pass
                p_pred, r_recon, aux_output = self.model(log_r_obs)
                
                # 3. VAE KL Loss Calculation
                kl_loss = torch.tensor(0.0, device=self.device)
                if isinstance(aux_output, tuple):
                    mu, logvar, phys_penalty = aux_output
                    # KL Divergence
                    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                    kl_loss = torch.mean(kl_div) * 0.001
                else:
                    phys_penalty = aux_output

                # 4. Criterion Calling
                loss_main, _, _, _, _ = self.criterion(
                    p_pred, r_recon, log_r_obs, p_true, 
                    self.cfg, curr_sup_w, phys_penalty
                )
                
                loss = loss_main + kl_loss

            if not torch.isfinite(loss):
                return float('nan')

            # 5. Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                current_scale = self.scaler.get_scale()
                skip_lr_sched = (current_scale > self.scaler.get_scale())
                if not skip_lr_sched:
                    self.scheduler.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

            # [FIX] 정의된 변수에 누적
            total_loss += loss.item()
            
            if step % 10 == 0:
                pbar.set_postfix({
                    "L": f"{loss.item():.4f}", 
                    "KL": f"{kl_loss.item():.4f}",
                    "LR": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
        return total_loss / steps

    @torch.no_grad()
    def validate(self, curr_sup_w: float):
        self.model.eval()
        
        r_meas, p_true = generate_synthetic_batch(
            batch_size=self.cfg.train.batch_size, 
            q_grid=self.q_grid, 
            config=self.cfg, 
            simulator=self.simulator,  # Cached
            augmenter=self.augmenter
        )
        r_meas = r_meas.to(self.device)
        p_true = p_true.to(self.device)
        
        log_r_obs = torch.log10(torch.clamp(r_meas, min=1e-11))
        
        p_pred, r_recon, aux_output = self.model(log_r_obs)
        
        kl_loss = 0.0
        if isinstance(aux_output, tuple):
            mu, logvar, phys_penalty = aux_output
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_loss = torch.mean(kl_div).item() * 0.001
        else:
            phys_penalty = aux_output
            
        # [FIX: Unpack 5 values here too]
        loss_main, _, _, _, loss_param = self.criterion(
            p_pred, r_recon, log_r_obs, p_true, 
            self.cfg, curr_sup_w, phys_penalty
        )
        
        metrics = {
            "loss": loss_main.item() + kl_loss,
            "loss_param": loss_param.item() if isinstance(loss_param, torch.Tensor) else loss_param,
            "kl_loss": kl_loss
        }
        
        return metrics, p_pred, r_recon, log_r_obs, p_true  # p_true 추가 반환

    def fit(self, resume: bool = False):
        if resume:
            self.load_checkpoint()
            
        max_epochs = self.cfg.train.epochs
        patience = getattr(self.cfg.train, 'early_stop_patience', 0)
        logger.info(f"Session Start: {self.cfg.experiment_name} | Device: {self.device} | Early Stop: {patience}")

        try:
            for epoch in range(self.start_epoch, max_epochs):
                curr_w = self._get_curriculum_weight(epoch)
                
                avg_train_loss = self.train_epoch(epoch, curr_w)
                val_metrics, v_pred, v_recon, v_target, v_true = self.validate(curr_w)  # p_true 추가
                val_loss = val_metrics["loss"]
                
                self.history["train_loss"].append(avg_train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["lr"].append(self.scheduler.get_last_lr()[0])
                self.history["sup_weight"].append(curr_w)
                
                self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
                self.writer.add_scalar("Loss/Val", val_loss, epoch)
                self.writer.add_scalar("Loss/Val_Param", val_metrics["loss_param"], epoch)
                self.writer.add_scalar("Loss/KL", val_metrics.get("kl_loss", 0), epoch)
                self.writer.add_scalar("Meta/Sup_Weight", curr_w, epoch)
                self.writer.add_scalar("Meta/LR", self.scheduler.get_last_lr()[0], epoch)

                self._save_checkpoint("last.pt", epoch)
                
                # Early Stopping Logic
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint("best.pt", epoch)
                else:
                    self.patience_counter += 1

                # Report at: every 50 epochs, epoch 0, or last epoch
                is_final = (epoch == max_epochs - 1)
                should_report = (epoch % 50 == 0) or is_final
                
                if should_report:
                    save_pinn_report(
                        self.q_grid, 
                        v_target[0:1],
                        v_recon[0:1], 
                        self.history["train_loss"], 
                        v_pred, 
                        v_true,  # p_true 전달
                        self.save_dir / f"report_E{epoch:03d}.png", 
                        step=epoch
                    )

                # Logging with more info (tqdm.write로 출력해서 프로그레스바와 충돌 방지)
                lr_now = self.scheduler.get_last_lr()[0]
                log_epoch(f"E{epoch:03d} | Loss: {val_loss:.4f} | Best: {self.best_val_loss:.4f} | LR: {lr_now:.2e} | Patience: {self.patience_counter}/{patience}")

                # Early Stop Check
                if patience > 0 and self.patience_counter >= patience:
                    logger.warning(f"Early stopping triggered at epoch {epoch}. No improvement for {patience} epochs.")
                    save_pinn_report(
                        self.q_grid, v_target[0:1], v_recon[0:1], 
                        self.history["train_loss"], v_pred, v_true, 
                        self.save_dir / f"report_FINAL_E{epoch:03d}.png", step=epoch
                    )
                    self.backup_to_drive(force=True)  # Early stop 시 강제 백업
                    break

                # [COLAB] 주기적 Drive 백업 (30분마다)
                self.backup_to_drive()

        except KeyboardInterrupt:
            logger.warning("⚠️ KeyboardInterrupt detected! Emergency saving...")
            self._save_checkpoint("emergency_interrupt.pt", epoch if 'epoch' in dir() else 0)
            self.backup_to_drive(force=True)
            logger.info("✅ Emergency backup complete.")
            raise
        except Exception as e:
            logger.error(f"❌ Error during training: {e}")
            try:
                self._save_checkpoint("emergency_error.pt", epoch if 'epoch' in dir() else 0)
                self.backup_to_drive(force=True)
                logger.info("✅ Emergency backup complete.")
            except:
                pass
            raise
        finally:
            # 학습 종료 후 최종 백업 (정상 종료 시)
            self.finalize_training()

    def _save_checkpoint(self, name: str, epoch: int):
        save_path = self.save_dir / name
        config_dump = self.cfg.model_dump(mode='json')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': config_dump, 
            'history': self.history
        }, save_path)

    def load_checkpoint(self, path: Optional[Path] = None):
        checkpoint_path = path or (self.save_dir / "last.pt")
        
        # [COLAB FIX] 로컬에 없으면 Drive에서 복원
        if IN_COLAB and not checkpoint_path.exists():
            drive_ckpt = self.drive_save_dir / "last.pt"
            if drive_ckpt.exists():
                logger.info(f"📥 Restoring checkpoint from Drive: {drive_ckpt}")
                shutil.copy2(drive_ckpt, checkpoint_path)
                # best.pt도 복원
                drive_best = self.drive_save_dir / "best.pt"
                if drive_best.exists():
                    shutil.copy2(drive_best, self.save_dir / "best.pt")
        
        if not checkpoint_path.exists():
            logger.warning(f"No checkpoint found at {checkpoint_path}. Starting fresh.")
            return

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        
        self.start_epoch = ckpt['epoch'] + 1
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.history = ckpt.get('history', self.history)
        logger.info(f"Checkpoint Loaded: Resume from Epoch {self.start_epoch}")

    def backup_to_drive(self, force: bool = False):
        """[COLAB] 로컬 체크포인트를 Drive로 백업"""
        if not IN_COLAB:
            return
        
        now = time.time()
        if not force and (now - self.last_backup_time) < self.backup_interval:
            return
        
        self.last_backup_time = now
        try:
            # 주요 파일만 백업 (last.pt, best.pt)
            for fname in ["last.pt", "best.pt"]:
                src = self.save_dir / fname
                if src.exists():
                    shutil.copy2(src, self.drive_save_dir / fname)
            logger.info(f"📦 Backup to Drive: {self.drive_save_dir}")
        except Exception as e:
            logger.warning(f"Backup failed: {e}")

    def finalize_training(self):
        """학습 종료 후 모든 파일을 Drive로 복사"""
        if not IN_COLAB:
            return
        
        try:
            # 전체 디렉토리 복사
            if self.save_dir.exists():
                for item in self.save_dir.iterdir():
                    dest = self.drive_save_dir / item.name
                    if item.is_file():
                        shutil.copy2(item, dest)
                    elif item.is_dir() and item.name != "logs":  # logs는 크므로 제외
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(item, dest)
            logger.info(f"✅ Final backup complete: {self.drive_save_dir}")
        except Exception as e:
            logger.error(f"Final backup failed: {e}")