import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Any, Optional

from xrefine.models import XRR_Fourier_PINN
from xrefine.data.generator import generate_synthetic_batch
from xrefine.data.augmentations import XRRAugmentor
from xrefine.config import XRefineConfig
from xrefine.train_utils import plot_learning_curves
from xrefine.logger import logger
from xrefine.core.structures import ParamSet

class XRREvaluator:
    """
    XRR PINN 모델의 물리적 타당성 및 예측 정확도를 정량적/시각적으로 평가하는 클래스.
    OOP 아키텍처를 통해 상태 관리와 평가 로직을 분리함.
    """
    def __init__(self, checkpoint_path: str, batch_size: int = 1000):
        self.checkpoint_path = Path(checkpoint_path)
        self.exp_dir = self.checkpoint_path.parent
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 상태 변수 초기화
        self.config: Optional[XRefineConfig] = None
        self.model: Optional[torch.nn.Module] = None
        self.results_df: Optional[pd.DataFrame] = None
        self.metrics: Dict[str, Dict] = {}

        # 안전한 경로 객체 허용 (PyTorch 2.6 대응)
        import pathlib
        torch.serialization.add_safe_globals([pathlib.WindowsPath, pathlib.PosixPath])

    def setup(self):
        """모델 및 설정을 로드하고 환경을 준비함"""
        logger.info(f"🔍 Loading checkpoint from {self.checkpoint_path}...")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        # Config 복원
        config_data = ckpt.get('config')
        self.config = XRefineConfig(**config_data) if isinstance(config_data, dict) else config_data
        
        # 모델 초기화 및 가중치 로드
        self.model = XRR_Fourier_PINN(config=self.config, device=self.device).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        
        # 학습 곡선 출력
        history = ckpt.get('history', {})
        if history:
            plot_learning_curves(history, self.exp_dir / "learning_curves.png")
        
        logger.info(f"✅ Model loaded: {self.config.model_type}")

    @torch.no_grad()
    def run_inference(self) -> tuple[torch.Tensor, ParamSet, ParamSet]:
        """테스트 데이터를 생성하고 추론을 수행함"""
        q_grid = self.config.q_grid.to(self.device)
        augmenter = XRRAugmentor(delta_q=0.0005, do_footprint=False)
        
        logger.info(f"🧪 Generating {self.batch_size} test samples...")
        r_measured, p_true = generate_synthetic_batch(
            self.batch_size, q_grid, config=self.config, augmenter=augmenter
        )
        
        target_log_r = torch.log10(torch.clamp(r_measured, min=1e-9))
        p_pred, r_recon, _ = self.model(target_log_r)
        
        return r_measured, p_true, p_pred, r_recon

    def calculate_statistics(self, p_true: ParamSet, p_pred: ParamSet):
        """모든 파라미터에 대해 동적으로 지표를 산출함 (Dynamic Discovery)"""
        all_results = []
        
        for key in p_true.keys():
            if key not in p_pred: continue
            
            y_true = p_true[key].cpu().numpy().flatten()
            y_pred = p_pred[key].cpu().numpy().flatten()
            
            # Metrics 계산 (Vectorized)
            err = y_pred - y_true
            mae = np.mean(np.abs(err))
            rmse = np.sqrt(np.mean(err**2))
            
            # R2 Score: $1 - \frac{SS_{res}}{SS_{tot}}$
            ss_res = np.sum(err**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            r2 = 1 - (ss_res / (ss_tot + 1e-9))
            
            self.metrics[key] = {"MAE": mae, "RMSE": rmse, "R2": r2}
            
            all_results.append(pd.DataFrame({
                'Parameter': key, 'True': y_true, 'Pred': y_pred, 'Error': err
            }))
            
        self.results_df = pd.concat(all_results)
        self.results_df.to_csv(self.exp_dir / "eval_results_full.csv", index=False)

    def report(self):
        """최종 리포트를 콘솔에 출력함"""
        header = f"{'Parameter':<15} | {'MAE':<10} | {'RMSE':<10} | {'R2':<10}"
        logger.info("\n" + "="*50 + f"\n{header}\n" + "-"*50)
        for k, v in self.metrics.items():
            logger.info(f"{k:<15} | {v['MAE']:<10.4f} | {v['RMSE']:<10.4f} | {v['R2']:<10.4f}")
        logger.info("="*50)

    def save_visuals(self, q_grid, r_true, r_recon):
        """상관관계도 및 물리 복원도 시각화"""
        # 1. Correlation Plots (Dynamic Subplots)
        keys = list(self.metrics.keys())
        n_params = len(keys)
        fig, axes = plt.subplots(2, n_params, figsize=(4 * n_params, 8))
        if n_params == 1: axes = axes.reshape(2, 1)

        for i, key in enumerate(keys):
            sub_df = self.results_df[self.results_df['Parameter'] == key]
            # Scatter
            axes[0, i].scatter(sub_df['True'], sub_df['Pred'], alpha=0.3, s=10)
            axes[0, i].plot([sub_df['True'].min(), sub_df['True'].max()], 
                           [sub_df['True'].min(), sub_df['True'].max()], 'r--')
            axes[0, i].set_title(f"{key}\n$R^2$: {self.metrics[key]['R2']:.4f}")
            # Error Hist
            axes[1, i].hist(sub_df['Error'], bins=30, color='green', alpha=0.6)
            axes[1, i].axvline(0, color='red', linestyle='--')

        plt.tight_layout()
        plt.savefig(self.exp_dir / "eval_correlation.png", dpi=120)

        # 2. Physics Reconstruction (Sample Visualization)
        self._plot_reconstruction(q_grid, r_true, r_recon)
        plt.close('all')

    def _plot_reconstruction(self, q, r_t, r_r, n_samples=6):
        """물리 곡선 매칭 시각화 (Internal helper)"""
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, hspace=0.3)
        indices = np.random.choice(len(r_t), n_samples, replace=False)
        
        for i, idx in enumerate(indices):
            sub_gs = gridspec.GridSpecFromSubplotSpec(2, 1, gs[i], height_ratios=[3, 1], hspace=0.05)
            ax_t, ax_b = fig.add_subplot(sub_gs[0]), fig.add_subplot(sub_gs[1])
            
            ax_t.semilogy(q, r_t[idx].cpu(), 'ko', ms=2, alpha=0.2)
            ax_t.semilogy(q, r_r[idx].cpu(), 'r-', lw=1.5)
            
            # Log Residuals
            res = torch.log10(r_r[idx]+1e-10) - torch.log10(r_t[idx]+1e-10)
            ax_b.plot(q, res.cpu(), color='blue', alpha=0.5)
            ax_b.axhline(0, color='k', ls='--')
            ax_b.set_ylim(-0.5, 0.5)
            
        plt.savefig(self.exp_dir / "eval_reconstruction.png", dpi=120)

# -----------------------------------------------------------------------------
# CLI Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    evaluator = XRREvaluator(args.checkpoint, args.batch_size)
    evaluator.setup()
    r_meas, p_true, p_pred, r_recon = evaluator.run_inference()
    evaluator.calculate_statistics(p_true, p_pred)
    evaluator.report()
    evaluator.save_visuals(evaluator.config.q_grid, r_meas, r_recon)
    logger.info("✅ Evaluation Task Finished.")