import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Optional, Tuple

from xrefine.models import XRR_Fourier_PINN
from xrefine.data.generator import generate_synthetic_batch
from xrefine.data.augmentations import XRRAugmentor
from xrefine.config import XRefineConfig
from xrefine.train_utils import plot_learning_curves
from xrefine.logger import logger
from xrefine.core.structures import ParamSet

class XRREvaluator:
    """
    XRR PINN 모델의 성능을 평가하는 클래스.
    GPU 가속을 활용하여 수천 개의 샘플에 대한 통계 지표를 즉시 산출합니다.
    """
    def __init__(self, checkpoint_path: str, batch_size: int = 1000):
        self.checkpoint_path = Path(checkpoint_path)
        self.exp_dir = self.checkpoint_path.parent
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 상태 변수
        self.config: Optional[XRefineConfig] = None
        self.model: Optional[XRR_Fourier_PINN] = None
        self.results_df: Optional[pd.DataFrame] = None
        self.metrics: Dict[str, Dict] = {}

        # PyTorch 2.6+ 보안 로드 설정
        import pathlib
        torch.serialization.add_safe_globals([pathlib.WindowsPath, pathlib.PosixPath, XRefineConfig])

    def setup(self):
        """체크포인트로부터 모델과 설정을 완벽히 복원"""
        logger.info(f"🔍 Loading checkpoint from {self.checkpoint_path}...")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # 1. Config 복구
        config_dict = ckpt.get('config')
        if isinstance(config_dict, dict):
            self.config = XRefineConfig(**config_dict)
        else:
            self.config = config_dict
            
        # 2. 모델 초기화 및 가중치 주입
        self.model = XRR_Fourier_PINN(config=self.config, device=self.device).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        
        # 3. 학습 히스토리 시각화 (존재할 경우)
        history = ckpt.get('history', {})
        if history:
            plot_learning_curves(history, self.exp_dir / "eval_learning_curves.png")
        
        logger.info(f"✅ Evaluation setup complete. Target Layers: {self.config.n_layers}")

    @torch.no_grad()
    def run_inference(self) -> Tuple[torch.Tensor, ParamSet, ParamSet, torch.Tensor]:
        """고속 증강기를 사용하여 테스트 셋 생성 및 추론 수행"""
        q_grid = self.config.q_grid.to(self.device)
        
        # 평가용 증강기 (실제 측정 환경과 유사하게 노이즈 주입)
        augmentor = XRRAugmentor(self.config).to(self.device)
        
        logger.info(f"🧪 Running inference on {self.batch_size} synthetic samples...")
        
        # 1. 데이터 생성 (GPU 위에서 즉시 생성)
        r_measured, p_true = generate_synthetic_batch(
            self.batch_size, q_grid, config=self.config, augmenter=augmentor
        )
        
        # 2. 모델 입력용 로그 변환
        log_r_input = torch.log10(torch.clamp(r_measured, min=1e-11))
        
        # 3. 모델 추론
        p_pred, r_recon, _ = self.model(log_r_input)
        
        return r_measured, p_true, p_pred, r_recon

    def calculate_statistics(self, p_true: ParamSet, p_pred: ParamSet):
        """파라미터별 오차 통계(MAE, RMSE, R2) 계산"""
        all_results = []
        logger.info("📊 Calculating statistical metrics...")

        for key in p_true.keys():
            if key not in p_pred: continue
            
            # Tensor to Numpy (flatten for statistics)
            y_true = p_true[key].detach().cpu().numpy().flatten()
            y_pred = p_pred[key].detach().cpu().numpy().flatten()
            
            err = y_pred - y_true
            mae = np.mean(np.abs(err))
            rmse = np.sqrt(np.mean(err**2))
            
            # R2 Score
            ss_res = np.sum(err**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            r2 = 1 - (ss_res / (ss_tot + 1e-9))
            
            self.metrics[key] = {"MAE": mae, "RMSE": rmse, "R2": r2}
            
            # 개별 데이터 포인트 저장용 DataFrame
            all_results.append(pd.DataFrame({
                'Parameter': key, 'True': y_true, 'Pred': y_pred, 'Error': err
            }))
            
        self.results_df = pd.concat(all_results)
        self.results_df.to_csv(self.exp_dir / "eval_metrics_detailed.csv", index=False)

    def report(self):
        """성적표 출력"""
        line = "=" * 65
        header = f"{'Parameter':<15} | {'MAE':<12} | {'RMSE':<12} | {'R2 Score':<12}"
        logger.info(f"\n{line}\n{header}\n{'-' * 65}")
        
        for k, v in self.metrics.items():
            logger.info(f"{k:<15} | {v['MAE']:<12.4f} | {v['RMSE']:<12.4f} | {v['R2']:<12.4f}")
        logger.info(f"{line}\n")

    def save_visuals(self, q_grid: torch.Tensor, r_true: torch.Tensor, r_recon: torch.Tensor):
        """평가 결과 시각화 리포트 저장"""
        keys = list(self.metrics.keys())
        n_params = len(keys)
        
        # 1. 상관관계 산점도 (Correlation Plots)
        fig, axes = plt.subplots(2, n_params, figsize=(3.5 * n_params, 7))
        if n_params == 1: axes = axes.reshape(2, 1)

        for i, key in enumerate(keys):
            sub_df = self.results_df[self.results_df['Parameter'] == key]
            
            # Scatter Plot
            axes[0, i].scatter(sub_df['True'], sub_df['Pred'], alpha=0.2, s=8, color='#34495e')
            # 1:1 Line
            m_min = min(sub_df['True'].min(), sub_df['Pred'].min())
            m_max = max(sub_df['True'].max(), sub_df['Pred'].max())
            axes[0, i].plot([m_min, m_max], [m_min, m_max], 'r--', lw=1.5)
            axes[0, i].set_title(f"{key}\n$R^2$: {self.metrics[key]['R2']:.4f}")
            
            # Error Distribution (Histogram)
            axes[1, i].hist(sub_df['Error'], bins=40, color='#27ae60', alpha=0.7)
            axes[1, i].axvline(0, color='black', linestyle='-', lw=1)
            axes[1, i].set_xlabel("Error")

        plt.tight_layout()
        plt.savefig(self.exp_dir / "eval_correlation_matrix.png", dpi=150)

        # 2. 물리 곡선 복원도 (Physics Reconstruction Samples)
        self._plot_sample_reconstruction(q_grid, r_true, r_recon)
        plt.close('all')

    def _plot_sample_reconstruction(self, q, r_t, r_r, n_samples=6):
        """랜덤하게 샘플을 선택하여 시뮬레이션 결과와 실제 데이터 매칭 시각화"""
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.25)
        indices = np.random.choice(len(r_t), n_samples, replace=False)
        q_np = q.cpu().numpy()
        
        for i, idx in enumerate(indices):
            sub_gs = gridspec.GridSpecFromSubplotSpec(2, 1, gs[i], height_ratios=[3, 1], hspace=0.08)
            ax_main = fig.add_subplot(sub_gs[0])
            ax_res = fig.add_subplot(sub_gs[1])
            
            # Main Reflectivity Plot
            y_true = r_t[idx].cpu().numpy()
            y_pred = r_r[idx].cpu().numpy()
            
            ax_main.semilogy(q_np, y_true, 'ko', ms=2.5, alpha=0.15, label='Measured')
            ax_main.semilogy(q_np, y_pred, 'r-', lw=1.8, label='PINN Recon')
            ax_main.set_title(f"Sample Index: {idx}")
            if i == 0: ax_main.legend()
            
            # Log Residuals (Physics Accuracy)
            res = np.log10(np.clip(y_pred, 1e-11, None)) - np.log10(np.clip(y_true, 1e-11, None))
            ax_res.plot(q_np, res, color='#2980b9', lw=1.2)
            ax_res.axhline(0, color='black', ls='--', alpha=0.5)
            ax_res.set_ylim(-0.5, 0.5)
            ax_res.set_ylabel("$\\Delta$ LogR")

        plt.savefig(self.exp_dir / "eval_physics_samples.png", dpi=150)

# -----------------------------------------------------------------------------
# CLI Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="XRefine PINN Model Evaluator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of test samples")
    args = parser.parse_args()

    evaluator = XRREvaluator(args.checkpoint, args.batch_size)
    evaluator.setup()
    
    # 추론 및 통계 산출
    r_meas, p_true, p_pred, r_recon = evaluator.run_inference()
    evaluator.calculate_statistics(p_true, p_pred)
    
    # 리포트 출력 및 시각화 저장
    evaluator.report()
    evaluator.save_visuals(evaluator.config.q_grid, r_meas, r_recon)
    
    logger.info(f"📊 Evaluation files saved in: {evaluator.exp_dir}")