import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from typing import Optional

from xrefine.config import CONFIG
from xrefine.data.generator import generate_synthetic_batch
from xrefine.data.augmentations import XRRAugmentor
from xrefine.train_utils import XRRWeightedLoss, save_pinn_report
from xrefine.logger import logger

torch.backends.cudnn.enabled = False

class XRefineTrainer:
    def __init__(self, model, config=CONFIG):
        self.cfg = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.q_grid = config.q_grid.to(self.device)
    
        self.criterion = XRRWeightedLoss(self.q_grid).to(self.device)
        self.augmenter = XRRAugmentor(config).to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None
        
        self.save_dir = Path(config.save_root) / config.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "logs"))
        
        # --- 상태 관리 변수 ---
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {"train_loss": [], "val_loss": []}

    def _get_curriculum_weight(self, epoch, max_epochs):
        progress = min(epoch / (max_epochs * 0.5), 1.0)
        return self.cfg.sup_w_start * (1 - progress) + self.cfg.sup_w_end * progress

    def train_epoch(self, epoch, steps, curr_sup_w):
        self.model.train()
        total_loss = 0
        for _ in range(steps):
            r_meas, p_true = generate_synthetic_batch(
                self.cfg.batch_size, self.q_grid, self.cfg, augmenter=self.augmenter
            )
            log_r_obs = torch.log10(torch.clamp(r_meas, min=1e-11))
            
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                p_pred, r_recon, phys_penalty = self.model(log_r_obs)
                loss, _, _, _ = self.criterion(
                    p_pred, r_recon, log_r_obs, p_true, 
                    self.cfg, curr_sup_w, phys_penalty
                )

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()
        return total_loss / steps

    @torch.no_grad()
    def validate(self, curr_sup_w: float):
        self.model.eval()
        r_meas, p_true = generate_synthetic_batch(
            self.cfg.batch_size, self.q_grid, self.cfg, augmenter=self.augmenter
        )
        log_r_obs = torch.log10(torch.clamp(r_meas, min=1e-11))
        p_pred, r_recon, phys_penalty = self.model(log_r_obs)
        loss, _, _, _ = self.criterion(
            p_pred, r_recon, log_r_obs, p_true, 
            self.cfg, curr_sup_w, phys_penalty
        )
        return loss.item(), p_pred, r_recon

    def load_checkpoint(self, path: Optional[Path] = None):
        """저장된 체크포인트를 불러와 모델, 옵티마이저, 학습 기록 복원"""
        checkpoint_path = path or (self.save_dir / "last.pt")
        
        if not checkpoint_path.exists():
            # 파일이 없으면 그냥 리턴하여 self.start_epoch = 0 유지
            logger.info(f"No checkpoint found at {checkpoint_path}. Starting from E000.")
            return

        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # 1. 모델 및 옵티마이저 상태 복원
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            
            # 2. 에폭 정보 업데이트 (저장된 에폭 다음부터 시작)
            self.start_epoch = ckpt['epoch'] + 1
            
            # 3. 기타 상태 복원
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            self.history = ckpt.get('history', self.history)
            
            logger.info(f"✅ Resume success: Starting from Epoch {self.start_epoch:03d}")
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}. Starting from scratch.")

    def fit(self, max_epochs=200, steps_per_epoch=100, resume=False):
        # 1. Resume 요청 시 체크포인트 로드 시도
        if resume:
            self.load_checkpoint()

        logger.info(f"🚀 XRefine Session Start: {self.cfg.experiment_name} | Device: {self.device}")

        pbar = tqdm(range(self.start_epoch, max_epochs), desc="Training")
        
        for epoch in pbar:
            curr_w = self._get_curriculum_weight(epoch, max_epochs)
            
            # --- Training & Validation ---
            avg_train_loss = self.train_epoch(epoch, steps_per_epoch, curr_w)
            val_loss, v_pred, v_recon = self.validate(curr_w)
            
            # 결과 기록
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)
            
            # TensorBoard
            self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Meta/Sup_Weight", curr_w, epoch)

            # --- [중요] 체크포인트 저장 ---
            # last.pt는 매 에폭 덮어쓰기 (Resume용)
            self._save_checkpoint("last.pt", epoch)

            # 성능 갱신 시 best.pt 저장 (최종 결과물용)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best.pt", epoch)

            # 시각화 리포트 (10에폭마다)
            if epoch % 10 == 0:
                report_name = f"report_E{epoch:03d}.png"
                save_pinn_report(
                    self.q_grid, torch.log10(v_recon[0:1]), v_recon[0:1], 
                    self.history["train_loss"], v_pred, None, 
                    self.save_dir / report_name, step=epoch
                )

            # 진행 표시줄 업데이트
            pbar.set_postfix({
                "Loss": f"{val_loss:.4f}", 
                "Best": f"{self.best_val_loss:.4f}",
                "W": f"{curr_w:.1f}"
            })

    def _save_checkpoint(self, name, epoch):
        save_path = self.save_dir / name
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.cfg.model_dump(),
            'history': self.history
        }, save_path)