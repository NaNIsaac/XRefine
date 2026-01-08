from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from xrefine.config import CONFIG, XRefineConfig
from xrefine.data.generator import generate_synthetic_batch
from xrefine.data.augmentations import XRRAugmentor
from xrefine.train.utils import XRRWeightedLoss, save_pinn_report
from xrefine.logger import logger

# PyTorch 2.9+ 최적화 설정
torch.backends.cudnn.benchmark = True

class XRefineTrainer:
    def __init__(self, model, config: XRefineConfig = CONFIG):
        self.cfg = config
        self.device = torch.device(config.device)
        
        # 모델 및 그리드 설정
        self.model = model.to(self.device)
        self.q_grid = config.q_grid.to(self.device)
        
        # 유틸리티 초기화
        self.criterion = XRRWeightedLoss(self.q_grid).to(self.device)
        self.augmenter = XRRAugmentor(config).to(self.device)
        
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

        # 경로 및 로거 설정
        self.save_dir = Path(config.save_root) / config.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "logs"))
        
        # 상태 관리
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {"train_loss": [], "val_loss": []}

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
        return metrics, p_pred, r_recon, log_r_obs

    def fit(self, resume: bool = False):
        if resume:
            self.load_checkpoint()
            
        max_epochs = self.cfg.train.epochs
        logger.info(f"Session Start: {self.cfg.experiment_name} | Device: {self.device}")

        for epoch in range(self.start_epoch, max_epochs):
            curr_w = self._get_curriculum_weight(epoch)
            
            avg_train_loss = self.train_epoch(epoch, curr_w)
            val_metrics, v_pred, v_recon, v_target = self.validate(curr_w)
            val_loss = val_metrics["loss"]
            
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)
            
            self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Loss/Val_Param", val_metrics["loss_param"], epoch)
            self.writer.add_scalar("Loss/KL", val_metrics.get("kl_loss", 0), epoch)
            self.writer.add_scalar("Meta/Sup_Weight", curr_w, epoch)
            self.writer.add_scalar("Meta/LR", self.scheduler.get_last_lr()[0], epoch)

            self._save_checkpoint("last.pt", epoch)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best.pt", epoch)

            if epoch % 20 == 0:
                save_pinn_report(
                    self.q_grid, 
                    v_target[0:1],  # [FIX] 진짜 정답 데이터 전달
                    v_recon[0:1], 
                    self.history["train_loss"], 
                    v_pred, 
                    None,  # p_true는 validate에서 받아오지 않았으므로 None (또는 추가 반환)
                    self.save_dir / f"report_E{epoch:03d}.png", 
                    step=epoch
                )

            logger.info(f"E{epoch:03d} | Loss: {val_loss:.4f} | KL: {val_metrics.get('kl_loss', 0):.4f} | Best: {self.best_val_loss:.4f}")

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
        if not checkpoint_path.exists():
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