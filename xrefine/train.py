from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from xrefine.config import CONFIG, XRefineConfig
from xrefine.data.generator import generate_synthetic_batch
from xrefine.data.augmentations import XRRAugmentor
from xrefine.train_utils import XRRWeightedLoss, save_pinn_report
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
        # 스케줄러 추가 (Warmup 대응)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.train.lr,
            steps_per_epoch=config.train.steps_per_epoch,
            epochs=config.train.epochs,
            pct_start=0.1 # 초기 10% 구간 warmup
        )
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
        """에폭 진행에 따른 파라미터 지도 학습 가중치 계산 (Linear Decay)"""
        max_epochs = self.cfg.train.epochs
        progress = min(epoch / (max_epochs * 0.7), 1.0) # 70% 시점까지 선형 증가
        w_start = self.cfg.train.sup_w_start
        w_end = self.cfg.train.sup_w_end
        return w_start * (1 - progress) + w_end * progress

    def train_epoch(self, epoch: int, curr_sup_w: float):
        self.model.train()
        total_loss = 0
        steps = self.cfg.train.steps_per_epoch
        pbar = tqdm(range(steps), desc=f"Epoch {epoch:03d}", leave=False)
        
        for step in pbar:
            # 1. 데이터 생성 (CPU에서 생성됨)
            r_meas, p_true = generate_synthetic_batch(
                q_grid=self.cfg.q_grid,
                batch_size=self.cfg.train.batch_size,
                config=self.cfg,
                augmenter=self.augmenter
            )
            
            # [수정 포인트] 데이터를 모델이 있는 장치(GPU)로 이동시켜야 합니다!
            r_meas = r_meas.to(self.device)
            p_true = p_true.to(self.device) # ParamSet의 .to() 메서드 활용
            
            log_r_obs = torch.log10(torch.clamp(r_meas, min=1e-11))
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                # 이제 log_r_obs가 GPU에 있으므로 fourier 연산(@ self.B)이 가능해집니다.
                p_pred, r_recon, phys_penalty = self.model(log_r_obs)
                loss, _, _, _ = self.criterion(
                    p_pred, r_recon, log_r_obs, p_true, 
                    self.cfg, curr_sup_w, phys_penalty
                )

            if not torch.isfinite(loss): # [Best Practice]
                return float('nan')

            if self.scaler:
                # [필수!] step()을 부르기 전에 반드시 scale()과 backward()가 먼저 와야 합니다.
                self.scaler.scale(loss).backward() 
                
                # 가중치 클리핑을 위해 unscale 먼저 수행
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # 이제 안전하게 step() 호출
                current_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # 스케줄러 경고 방지 로직 (사용자님이 추가하신 부분)
                skip_lr_sched = current_scale > self.scaler.get_scale()
                if not skip_lr_sched:
                    self.scheduler.step()
            else:
                # 일반적인 float32 학습 시
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

            total_loss += loss.item()
            if step % 10 == 0:
                pbar.set_postfix({"L": f"{loss.item():.4f}", "LR": f"{self.scheduler.get_last_lr()[0]:.2e}"})
                
        return total_loss / steps

    @torch.no_grad()
    def validate_epoch(self, epoch: int):
        self.model.eval()
        losses = [
            self.criterion(*self.model(torch.log10(generate_synthetic_batch(q_grid=self.cfg.q_grid, batch_size=self.cfg.train.batch_size, config=self.cfg)[0])), 
            torch.log10(generate_synthetic_batch(q_grid=self.cfg.q_grid, batch_size=self.cfg.train.batch_size, config=self.cfg)[0]), 
            generate_synthetic_batch(q_grid=self.cfg.q_grid, batch_size=self.cfg.train.batch_size, config=self.cfg)[1], self.cfg, 1.0, 0)[0].item() 
            for _ in range(5)
        ]
        return np.mean(losses)

    @torch.no_grad()
    def validate(self, curr_sup_w: float):
        self.model.eval()
        r_meas, p_true = generate_synthetic_batch(
            batch_size=self.cfg.train.batch_size, 
            q_grid=self.q_grid, 
            config=self.cfg, 
            augmenter=self.augmenter
        )
        # 데이터를 GPU로 이동
        r_meas = r_meas.to(self.device)
        p_true = p_true.to(self.device)
        
        log_r_obs = torch.log10(torch.clamp(r_meas, min=1e-11))
        p_pred, r_recon, phys_penalty = self.model(log_r_obs)
        loss, _, _, _ = self.criterion(p_pred, r_recon, log_r_obs, p_true, self.cfg, curr_sup_w, phys_penalty)
        return loss.item(), p_pred, r_recon

    def fit(self, resume: bool = False):
        if resume:
            self.load_checkpoint()
            
        max_epochs = self.cfg.train.epochs
        logger.info(f"Session Start: {self.cfg.experiment_name} | Device: {self.device}")

        for epoch in range(self.start_epoch, max_epochs):
            curr_w = self._get_curriculum_weight(epoch)
            
            avg_train_loss = self.train_epoch(epoch, curr_w)
            val_loss, v_pred, v_recon = self.validate(curr_w)
            
            # 히스토리 기록
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)
            
            # TensorBoard 기록
            self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Meta/Sup_Weight", curr_w, epoch)
            self.writer.add_scalar("Meta/LR", self.scheduler.get_last_lr()[0], epoch)

            # 체크포인트 관리
            self._save_checkpoint("last.pt", epoch)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best.pt", epoch)

            # 시각화 리포트 (5에폭마다)
            if epoch % 20 == 0:
                save_pinn_report(
                    self.q_grid, torch.log10(v_recon[0:1]), v_recon[0:1], 
                    self.history["train_loss"], v_pred, None, 
                    self.save_dir / f"report_E{epoch:03d}.png", step=epoch
                )

            logger.info(f"E{epoch:03d} | Loss: {val_loss:.4f} | Best: {self.best_val_loss:.4f} | W: {curr_w:.1f}")

    def _save_checkpoint(self, name: str, epoch: int):
        save_path = self.save_dir / name
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.cfg.model_dump(),
            'history': self.history
        }, save_path)

    def load_checkpoint(self, path: Optional[Path] = None):
        checkpoint_path = path or (self.save_dir / "last.pt")
        if not checkpoint_path.exists(): return

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        
        self.start_epoch = ckpt['epoch'] + 1
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.history = ckpt.get('history', self.history)
        logger.info(f"Checkpoint Loaded: Resume from Epoch {self.start_epoch}")