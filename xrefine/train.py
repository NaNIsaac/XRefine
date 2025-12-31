from pathlib import Path
import torch
from torch import optim
from tqdm import tqdm

from xrefine.data.generator import generate_synthetic_batch, get_xrr_dataloader
from xrefine.config import XRefineConfig
from xrefine.train_utils import compute_loss, QWeightedLogMSE

class XRefineTrainer:
    def __init__(self, model, config: XRefineConfig, save_dir: Path | str, patience: int = 30):
        self.cfg = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.q_grid = config.q_grid.to(self.device)
        
        # 1. Losses & Optimizer
        self.criterion_recon = QWeightedLogMSE(self.q_grid, q_cut=0.5)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None
        
        # 2. Early Stopping
        self.early_stop_patience = patience
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        
        # 3. Data & Checkpointing
        self.val_data = self._prepare_val_set()
        self.checkpoint_dir = Path(save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"train_loss": [], "val_loss": []}

    def _prepare_val_set(self):
        print("🧪 Generating fixed validation set for Early Stopping...")
        with torch.no_grad():
            r, p = generate_synthetic_batch(
                batch_size=self.cfg.batch_size * 2, 
                q_grid=self.q_grid, 
                config=self.cfg
            )
            # v_log_r, v_params_true
            return torch.log10(torch.clamp(r, min=1e-9)).detach(), p

    def _get_curriculum_weight(self, epoch, max_epochs):
        ratio = min(epoch / (max_epochs * 0.4), 1.0)
        return self.cfg.sup_w_start * (1 - ratio) + self.cfg.sup_w_end * ratio

    def train_step(self, target_log_r, params_true, curr_sup_w):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=self.scaler is not None):
            params_pred, r_recon, phys_penalty = self.model(target_log_r)
            
            # Unified loss from train_utils
            total_loss, _, _, _ = compute_loss(
                params_pred, r_recon, target_log_r, 
                params_true, self.cfg, curr_sup_w, phys_penalty
            )

        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()

        return total_loss.item()

    @torch.no_grad()
    def validate(self, curr_sup_w):
        self.model.eval()
        v_log_r, v_params_true = self.val_data
        
        params_pred, r_recon, phys_penalty = self.model(v_log_r)
        
        val_loss, _, _, _ = compute_loss(
            params_pred, r_recon, v_log_r, 
            v_params_true, self.cfg, curr_sup_w, phys_penalty
        )
        return val_loss.item()

    def fit(self, max_epochs=200, steps_per_epoch=100):
        # Parallel Data Loading (No Disk I/O, Pure CPU Simulation)
        q_grid_cpu = self.cfg.q_grid.cpu()
        train_loader = get_xrr_dataloader(self.cfg, q_grid_cpu, steps_per_epoch)

        print(f"🔥 Starting Training (Device: {self.device})")
        pbar = tqdm(range(max_epochs), leave=False)
        for epoch in pbar:
            pbar.desc = f"E{epoch:03d}"
            curr_w = self._get_curriculum_weight(epoch, max_epochs)
            epoch_loss = 0
            
            for r_meas, params_true in train_loader:
                target_log_r = torch.log10(torch.clamp(r_meas, min=1e-9)).to(self.device)
                params_true = {k: v.to(self.device) for k, v in params_true.items()}
                loss_val = self.train_step(target_log_r, params_true, curr_w)
                epoch_loss += loss_val
                pbar.set_postfix({'loss': f'{loss_val:.4f}'})

            avg_train = epoch_loss / steps_per_epoch
            val_loss = self.validate(curr_w)
            
            self.history['train_loss'].append(avg_train)
            self.history['val_loss'].append(val_loss)
            
            print(f"E{epoch:03d} | Train: {avg_train:.4f} | Val: {val_loss:.4f} | W_sup: {curr_w:.1f}")

            # Early Stopping Logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best_pinn.pt", epoch)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    print(f"🛑 Early stopping at epoch {epoch}. Restoration of physics complete.")
                    break
            
            self._save_checkpoint("latest.pt", epoch)

    def _save_checkpoint(self, name, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.cfg.model_dump()
        }, self.checkpoint_dir / name)