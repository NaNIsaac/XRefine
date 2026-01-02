import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from xrefine.data.generator import generate_synthetic_batch, get_xrr_dataloader
from xrefine.config import XRefineConfig
from xrefine.train_utils import compute_loss, save_pinn_report, plot_learning_curves
from xrefine.core.structures import ParamSet
from xrefine.logger import logger

class XRefineTrainer:
    def __init__(self, model: torch.nn.Module, config: XRefineConfig, patience: int = 30):
        self.cfg = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.q_grid = config.q_grid.to(self.device)
        
        # 1. 경로 설정 및 TensorBoard Writer (코랩 사용 시 드라이브 마운트 경로 권장)
        self.checkpoint_dir = config.save_root / config.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / "logs"))

        # 2. 최적화 및 Mixed Precision 설정
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None
        
        # 3. 상태 관리 변수
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_patience = patience
        self.history = {"train_loss": [], "val_loss": []}

        self.val_data = self._prepare_val_set()

    def _prepare_val_set(self) -> tuple[torch.Tensor, ParamSet[torch.Tensor]]:
        """학습 내내 변하지 않는 검증 세트 구성"""
        logger.info("🧪 Generating fixed validation set for monitoring...")
        with torch.no_grad():
            # r: [B, Q], p: ParamSet[Tensor]
            r, p = generate_synthetic_batch(
                batch_size=self.cfg.batch_size * 2,
                q_grid=self.q_grid, 
                config=self.cfg
            )
            # Log scale로 변환하여 저장
            log_r = torch.log10(torch.clamp(r, min=1e-9)).to(self.device)
            p_gpu = {k: v.to(self.device) for k, v in p.items()}
            return log_r, p_gpu

    def load_checkpoint(self, path: Optional[Path] = None):
        """저장된 최신 상태를 불러와 학습 재개 (Resume)"""
        path = path or (self.checkpoint_dir / "latest.pt")
        if not path.exists():
            logger.info(f"⚠️ No checkpoint found at {path}. Starting fresh.")
            return

        logger.info(f"🔄 Resuming training from {path}...")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_val_loss = ckpt.get('val_loss', float('inf'))
        self.history = ckpt.get('history', self.history)
        logger.info(f"✅ Ready to train from E{self.start_epoch}")

    def _get_curriculum_weight(self, epoch: int, max_epochs: int) -> float:
        """에폭에 따라 지도 학습 가중치를 서서히 줄여 물리적 피팅 유도"""
        ratio = min(epoch / (max_epochs * 0.4), 1.0)
        return self.cfg.sup_w_start * (1 - ratio) + self.cfg.sup_w_end * ratio

    def train_step(self, target_log_r: torch.Tensor, params_true: ParamSet[torch.Tensor], curr_sup_w: float):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=self.scaler is not None):
            # Model Forward: params_pred는 딕셔너리 형태의 ParamSet[Tensor]
            params_pred, r_recon, phys_penalty = self.model(target_log_r)
            
            # Key-based Loss 계산
            total_loss, l_recon, l_param, p_phys = compute_loss(
                params_pred, r_recon, target_log_r, 
                params_true, self.cfg, curr_sup_w, phys_penalty
            )

        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        return total_loss.item(), params_pred, l_recon

    @torch.no_grad()
    def _validate(self, curr_sup_w: float):
        self.model.eval()
        v_log_r, v_params_true = self.val_data
        
        params_pred, r_recon, phys_penalty = self.model(v_log_r)
        val_loss, _, _, _ = compute_loss(
            params_pred, r_recon, v_log_r, 
            v_params_true, self.cfg, curr_sup_w, phys_penalty
        )
        return val_loss.item(), params_pred, r_recon

    def fit(self, max_epochs: int = 200, steps_per_epoch: int = 100, resume: bool = False):
        if resume:
            self.load_checkpoint()

        train_loader = get_xrr_dataloader(self.cfg, self.cfg.q_grid.cpu(), steps_per_epoch)
        logger.info(f"🔥 Starting XRefine session (Device: {self.device})")
        
        pbar = tqdm(range(self.start_epoch, max_epochs), desc="Training")
        for epoch in pbar:
            curr_w = self._get_curriculum_weight(epoch, max_epochs)
            epoch_loss = 0
            
            for step, (r_meas, p_true) in enumerate(train_loader):
                log_r = torch.log10(torch.clamp(r_meas, min=1e-9)).to(self.device)
                p_true_gpu = {k: v.to(self.device) for k, v in p_true.items()}
                
                loss_val, p_pred, l_recon = self.train_step(log_r, p_true_gpu, curr_w)
                epoch_loss += loss_val
                
                # TensorBoard Step 단위 로깅 (20스텝마다)
                if step % 20 == 0:
                    g_step = epoch * steps_per_epoch + step
                    self.writer.add_scalar("Step/Total_Loss", loss_val, g_step)
                    self.writer.add_scalar("Step/Recon_Loss", l_recon.item(), g_step)

            # 에폭 종료 후 성적표 작성
            avg_train = epoch_loss / steps_per_epoch
            val_loss, v_pred, v_recon = self._validate(curr_w)
            
            self.history['train_loss'].append(avg_train)
            self.history['val_loss'].append(val_loss)

            # --- TensorBoard Epoch Logging ---
            self.writer.add_scalar("Epoch/Train_Loss", avg_train, epoch)
            self.writer.add_scalar("Epoch/Val_Loss", val_loss, epoch)
            self.writer.add_scalar("Params/I0_mean", v_pred['i0'].mean().item(), epoch)
            self.writer.add_scalar("Params/L_mean", v_pred['L'].mean().item(), epoch)
            self.writer.add_scalar("Curriculum/Weight", curr_w, epoch)

            # 5에폭마다 물리적 피팅 결과 시각화
            if epoch % 5 == 0:
                self._log_visual_reports(epoch, v_recon, v_pred)

            # --- Checkpointing & Early Stopping ---
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best_pinn.pt", epoch)
            else:
                self.patience_counter += 1

            self._save_checkpoint("latest.pt", epoch)
            pbar.set_postfix({'val_loss': f'{val_loss:.4f}', 'patience': self.patience_counter})

            if self.patience_counter >= self.early_stop_patience:
                logger.info(f"🛑 Early stopping at E{epoch}. Performance plateaued.")
                break

    def _log_visual_reports(self, epoch: int, v_recon: torch.Tensor, v_pred: ParamSet[torch.Tensor]):
        """이미지 기반 리포트를 TensorBoard와 디스크에 저장"""
        report_path = self.checkpoint_dir / f"report_E{epoch:03d}.png"
        v_log_r_target, v_params_true = self.val_data
        
        # 1. 디스크 저장
        save_pinn_report(
            self.q_grid, v_log_r_target, v_recon, 
            self.history['train_loss'], v_pred, v_params_true, 
            report_path, step=epoch
        )
        plot_learning_curves(self.history, self.checkpoint_dir / "learning_curves.png")
        
        # 2. TensorBoard Figure로 추가 (Matplotlib 객체 생성)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.semilogy(self.q_grid.cpu(), 10**v_log_r_target[0].cpu(), 'ko', alpha=0.2, label='Target')
        ax.semilogy(self.q_grid.cpu(), v_recon[0].cpu(), 'r-', lw=2, label='PINN Fit')
        ax.legend()
        ax.set_title(f"Epoch {epoch} Fit Result")
        self.writer.add_figure("Plots/Reflectivity_Fit", fig, epoch)

    def _save_checkpoint(self, name: str, epoch: int):
        save_path = self.checkpoint_dir / name
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.cfg.model_dump()
        }, save_path)

    def __del__(self):
        if hasattr(self, 'writer'):
            self.writer.close()
