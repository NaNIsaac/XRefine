import shutil
import time
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from xrefine.checkpoint import load_checkpoint
from xrefine.config import XRefineConfig
from xrefine.core.abeles import AbelesMatrix
from xrefine.data.augmentations import XRRAugmentor
from xrefine.data.generator import generate_synthetic_batch
from xrefine.logger import logger
from xrefine.train.utils import XRRWeightedLoss, save_pinn_report

torch.backends.cudnn.benchmark = True


def is_colab() -> bool:
    try:
        import google.colab

        return True
    except ImportError:
        return False


IN_COLAB = is_colab()

# standard tqdm; the notebook version conflicts with loguru
from tqdm import tqdm

if IN_COLAB:
    from IPython.display import clear_output


def log_epoch(msg: str):
    """Print a log line without clashing with tqdm."""
    tqdm.write(msg)


class XRefineTrainer:
    def __init__(self, model, config: XRefineConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        self.model = model.to(self.device)
        self.q_grid = config.q_grid.to(self.device)

        self.criterion = XRRWeightedLoss(self.q_grid).to(self.device)
        self.augmenter = XRRAugmentor(config).to(self.device)
        self.simulator = AbelesMatrix(device=self.device)  # Cached Simulator

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.train.lr,
            steps_per_epoch=config.train.steps_per_epoch,
            epochs=config.train.epochs,
            pct_start=0.1,
        )
        # a dummy optimizer step avoids the OneCycleLR warning
        self.optimizer.step()
        self.scaler = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None

        # local save path and Drive backup path
        self.drive_save_dir = Path(config.save_root) / config.experiment_name
        if IN_COLAB:
            # Colab: save locally under /tmp, Drive is only the backup
            self.save_dir = Path("/tmp/xrefine_checkpoints") / config.experiment_name
            self.last_backup_time = time.time()
            self.backup_interval = 30 * 60  # 30 minutes
        else:
            self.save_dir = self.drive_save_dir

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.drive_save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "logs"))

        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "lr": [], "sup_weight": []}

    def _get_curriculum_weight(self, epoch: int) -> float:
        max_epochs = self.cfg.train.epochs
        progress = min(epoch / (max_epochs * 0.7), 1.0)
        w_start = self.cfg.train.sup_w_start
        w_end = self.cfg.train.sup_w_end
        return w_start * (1 - progress) + w_end * progress

    def train_epoch(self, epoch: int, curr_sup_w: float):
        # [ENHANCED] Support curriculum learning in enhanced models
        if hasattr(self.model, "set_epoch"):
            self.model.set_epoch(epoch)

        self.model.train()
        total_loss = 0.0
        total_param_loss = 0.0

        steps = self.cfg.train.steps_per_epoch
        pbar = tqdm(
            range(steps),
            desc=f"Epoch {epoch:03d}",
            leave=False,
            total=steps,
            ncols=100,
            ascii=" >=",
            mininterval=1.0,
        )

        for step in pbar:
            r_meas, p_true = generate_synthetic_batch(
                q_grid=self.cfg.q_grid,
                batch_size=self.cfg.train.batch_size,
                config=self.cfg,
                simulator=self.simulator,
                augmenter=self.augmenter,
            )

            log_r_obs = torch.log10(torch.clamp(r_meas, min=1e-11))
            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.scaler is not None):
                # fixed_values=None in training so the network predicts by itself; feeding p_true
                # would copy the answer (loss = 0)
                p_pred, r_recon, aux_output = self.model(log_r_obs, fixed_values=None)

                kl_loss = torch.tensor(0.0, device=self.device)
                if isinstance(aux_output, tuple):
                    mu, logvar, phys_penalty = aux_output
                    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                    kl_loss = torch.mean(kl_div) * 0.001
                else:
                    phys_penalty = aux_output

                loss_main, _, _, _, loss_param = self.criterion(
                    p_pred, r_recon, log_r_obs, p_true, self.cfg, curr_sup_w, phys_penalty
                )

                loss = loss_main + kl_loss

            if not torch.isfinite(loss):
                return float("nan")

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # OneCycleLR Scale safe-guard
                current_scale = self.scaler.get_scale()
                skip_lr_sched = current_scale > self.scaler.get_scale()
                if not skip_lr_sched:
                    self.scheduler.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

            total_loss += loss.item()
            total_param_loss += loss_param.item()

            if step % 10 == 0:
                pbar.set_postfix(
                    {
                        "L": f"{loss.item():.4f}",
                        "P_L": f"{loss_param.item():.4f}",
                        "KL": f"{kl_loss.item():.4f}",
                        "LR": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    }
                )

        return total_loss / steps

    @torch.no_grad()
    def validate(self, curr_sup_w: float):
        self.model.eval()

        r_meas, p_true = generate_synthetic_batch(
            batch_size=self.cfg.train.batch_size,
            q_grid=self.q_grid,
            config=self.cfg,
            simulator=self.simulator,  # Cached
            augmenter=self.augmenter,
        )

        log_r_obs = torch.log10(torch.clamp(r_meas, min=1e-15))

        # no p_true injection (fixed_values=None) in validation, to measure the real predictive
        # performance
        p_pred, r_recon, aux_output = self.model(log_r_obs, fixed_values=None)

        kl_loss = 0.0
        phys_penalty = torch.tensor(0.0, device=self.device)

        if isinstance(aux_output, tuple):
            mu, logvar, phys_penalty = aux_output
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_loss = torch.mean(kl_div).item() * 0.001
        else:
            phys_penalty = aux_output

        # XRRWeightedLoss returns: (total, base, fringe, qc, param)
        loss_main, _, _, _, loss_param = self.criterion(
            p_pred, r_recon, log_r_obs, p_true, self.cfg, curr_sup_w, phys_penalty
        )

        metrics = {
            "loss": loss_main.item() + kl_loss,
            "loss_param": loss_param.item() if isinstance(loss_param, torch.Tensor) else loss_param,
            "kl_loss": kl_loss,
        }

        # [FIX] Return signature updated: p_true added
        return metrics, p_pred, r_recon, log_r_obs, p_true

    def fit(self, resume: bool = False):
        if resume:
            self.load_checkpoint()

        max_epochs = self.cfg.train.epochs
        patience = getattr(self.cfg.train, "early_stop_patience", 0)
        logger.info(
            f"Session Start: {self.cfg.experiment_name} | Device: {self.device} | Early Stop: {patience}"
        )

        try:
            for epoch in range(self.start_epoch, max_epochs):
                curr_w = self._get_curriculum_weight(epoch)

                avg_train_loss = self.train_epoch(epoch, curr_w)
                val_metrics, v_pred, v_recon, v_target, v_true = self.validate(curr_w)
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
                is_final = epoch == max_epochs - 1
                is_best = val_loss < self.best_val_loss  # best_val_loss is updated above
                should_report = (epoch % 50 == 0) or is_final or is_best

                if should_report:
                    # [Performance] Only save file if best or final. Otherwise just TensorBoard.
                    save_path = None
                    if is_best or is_final:
                        save_path = self.save_dir / f"report_E{epoch:03d}.png"

                    fig = save_pinn_report(
                        self.q_grid,
                        v_target[0:1],
                        v_recon[0:1],
                        self.history["train_loss"],
                        v_pred,
                        v_true,
                        save_path=save_path,  # Can be None
                        step=epoch,
                    )

                    if fig:
                        self.writer.add_figure("Validation/Fit", fig, epoch)
                        import matplotlib.pyplot as plt

                        plt.close(fig)

                # log through tqdm.write so it does not clash with the progress bar
                lr_now = self.scheduler.get_last_lr()[0]
                log_epoch(
                    f"E{epoch:03d} | Loss: {val_loss:.4f} | Best: {self.best_val_loss:.4f} | LR: {lr_now:.2e} | Patience: {self.patience_counter}/{patience}"
                )

                # Early Stop Check
                if patience > 0 and self.patience_counter >= patience:
                    logger.warning(
                        f"Early stopping triggered at epoch {epoch}. No improvement for {patience} epochs."
                    )
                    save_pinn_report(
                        self.q_grid,
                        v_target[0:1],
                        v_recon[0:1],
                        self.history["train_loss"],
                        v_pred,
                        v_true,
                        self.save_dir / f"report_FINAL_E{epoch:03d}.png",
                        step=epoch,
                    )
                    self.backup_to_drive(force=True)
                    break

                self.backup_to_drive()

        except KeyboardInterrupt:
            logger.warning("⚠️ KeyboardInterrupt detected! Emergency saving...")
            self._save_checkpoint("emergency_interrupt.pt", epoch if "epoch" in dir() else 0)
            self.backup_to_drive(force=True)
            logger.info("✅ Emergency backup complete.")
            raise
        except Exception as e:
            logger.error(f"❌ Error during training: {e}")
            try:
                self._save_checkpoint("emergency_error.pt", epoch if "epoch" in dir() else 0)
                self.backup_to_drive(force=True)
                logger.info("✅ Emergency backup complete.")
            except:
                pass
            raise
        finally:
            self.finalize_training()

    def _save_checkpoint(self, name: str, epoch: int):
        save_path = self.save_dir / name
        config_dump = self.cfg.model_dump(mode="json")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "config": config_dump,
                "history": self.history,
            },
            save_path,
        )

    def load_checkpoint(self, path: Path | None = None):
        checkpoint_path = path or (self.save_dir / "last.pt")

        if IN_COLAB and not checkpoint_path.exists():
            drive_ckpt = self.drive_save_dir / "last.pt"
            if drive_ckpt.exists():
                logger.info(f"📥 Restoring checkpoint from Drive: {drive_ckpt}")
                shutil.copy2(drive_ckpt, checkpoint_path)
                drive_best = self.drive_save_dir / "best.pt"
                if drive_best.exists():
                    shutil.copy2(drive_best, self.save_dir / "best.pt")

        if not checkpoint_path.exists():
            logger.warning(f"No checkpoint found at {checkpoint_path}. Starting fresh.")
            return

        try:
            ckpt = load_checkpoint(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

            self.start_epoch = ckpt["epoch"] + 1
            self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
            self.history = ckpt.get("history", self.history)
            logger.info(f"Checkpoint Loaded: Resume from Epoch {self.start_epoch}")
        except RuntimeError as e:
            logger.warning(f"⚠️  [Incompatible Checkpoint] Structure mismatch detected: {e}")
            logger.warning("🔄 Starting a NEW training session from scratch (Epoch 0).")
            return
        except Exception as e:
            logger.warning(f"⚠️  Failed to load checkpoint: {e}")
            logger.warning("🔄 Starting a NEW training session from scratch.")
            return

    def backup_to_drive(self, force: bool = False):
        """Back up the local checkpoints to Drive (Colab only)."""
        if not IN_COLAB:
            return

        now = time.time()
        if not force and (now - self.last_backup_time) < self.backup_interval:
            return

        self.last_backup_time = now
        try:
            logger.info("⏳ Starting Drive backup...")
            for fname in ["last.pt", "best.pt"]:
                src = self.save_dir / fname
                if src.exists():
                    shutil.copy2(src, self.drive_save_dir / fname)
            logger.info(f"📦 Backup to Drive complete: {self.drive_save_dir}")
        except Exception as e:
            logger.warning(f"Backup failed: {e}")

    def finalize_training(self):
        """Copy all files to Drive after training ends (Colab only)."""
        if not IN_COLAB:
            return

        try:
            if self.save_dir.exists():
                for item in self.save_dir.iterdir():
                    dest = self.drive_save_dir / item.name
                    if item.is_file():
                        shutil.copy2(item, dest)
                    elif item.is_dir() and item.name != "logs":  # logs is large, so it is excluded
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(item, dest)
            logger.info(f"✅ Final backup complete: {self.drive_save_dir}")
        except Exception as e:
            logger.error(f"Final backup failed: {e}")
