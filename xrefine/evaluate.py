import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import fields
from tqdm import tqdm

from xrefine.models import XRR_Fourier_PINN
from xrefine.data.generator import generate_synthetic_batch
from xrefine.data.augmentations import XRRAugmentor
from xrefine.config import XRefineConfig
from xrefine.train_utils import plot_learning_curves
from xrefine.logger import logger
from xrefine.core.structures import ParamSet

class XRREvaluator:
    """
    XRR PINN 모델의 다각도 성능 분석 및 고품질 리포트 생성기.
    """
    def __init__(self, checkpoint_path: str, batch_size: int = 1000):
        self.checkpoint_path = Path(checkpoint_path)
        self.exp_dir = self.checkpoint_path.parent
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.config: Optional[XRefineConfig] = None
        self.model: Optional[XRR_Fourier_PINN] = None
        self.results_df: Optional[pd.DataFrame] = None
        self.sample_metrics: Optional[pd.DataFrame] = None 
        self.metrics_summary: Dict[str, Dict] = {}

    def setup(self):
        """환경 설정 및 모델 로드"""
        logger.info(f"🔍 Initializing Evaluator from {self.checkpoint_path}...")
        import pathlib
        torch.serialization.add_safe_globals([pathlib.WindowsPath, pathlib.PosixPath, XRefineConfig])
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        match ckpt.get('config'):
            case dict() as d: self.config = XRefineConfig(**d)
            case XRefineConfig() as c: self.config = c
            case _: raise ValueError("Invalid config format")

        self.model = XRR_Fourier_PINN(config=self.config, device=self.device).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        
        if history := ckpt.get('history'):
            plot_learning_curves(history, self.exp_dir / "eval_learning_curves.png")
        logger.info(f"✅ Setup complete. N_Layers: {self.config.model.n_layers}")

    @torch.no_grad()
    def run_inference(self) -> Tuple[torch.Tensor, ParamSet, ParamSet, torch.Tensor]:
        """배치 분할 추론 수행"""
        q_grid = self.config.q_grid.to(self.device)
        augmentor = XRRAugmentor(self.config).to(self.device)
        
        split_size = 50
        num_iters = self.batch_size // split_size
        r_m_list, r_r_list, p_t_list, p_p_list = [], [], [], []

        for _ in tqdm(range(num_iters), desc="Running Inference"):
            r_m, p_t = generate_synthetic_batch(split_size, q_grid, self.config, augmenter=augmentor)
            log_r_in = torch.log10(torch.clamp(r_m, min=1e-11))
            p_p, r_r, _ = self.model(log_r_in)

            r_m_list.append(r_m.cpu())
            r_r_list.append(r_r.cpu())
            p_t_list.append(p_t)
            p_p_list.append(p_p)

        return torch.cat(r_m_list), ParamSet.cat(p_t_list), ParamSet.cat(p_p_list), torch.cat(r_r_list)

    def calculate_statistics(self, r_t: torch.Tensor, r_r: torch.Tensor, p_t: ParamSet, p_p: ParamSet):
        """다차원 통계 지표 산출"""
        logger.info("📊 Calculating global and per-sample statistics...")
        
        # 1. Physics Fidelity: Log-MSE
        log_t = torch.log10(torch.clamp(r_t, min=1e-11))
        log_r = torch.log10(torch.clamp(r_r, min=1e-11))
        mse_per_sample = torch.mean((log_t - log_r)**2, dim=1).numpy()
        self.sample_metrics = pd.DataFrame({'idx': np.arange(self.batch_size), 'log_mse': mse_per_sample})

        # 2. Parameter Accuracy
        all_res = []
        for fld in fields(ParamSet):
            key = fld.name
            y_t = getattr(p_t, key).detach().cpu().numpy().flatten()
            y_p = getattr(p_p, key).detach().cpu().numpy().flatten()
            
            err = y_p - y_t
            # evaluate.py의 calculate_statistics 내부
            self.metrics_summary[key] = {
                "MAE": np.mean(np.abs(err)),
                "MAPE": np.mean(np.abs(err) / (np.abs(y_t) + 1e-9)) * 100,
                "RMSE": np.sqrt(np.mean(err**2)),  # <--- 이 줄이 있는지 확인!
                "R2": 1 - (np.sum(err**2) / (np.sum((y_t - np.mean(y_t))**2) + 1e-9))
            }
            all_res.append(pd.DataFrame({'Parameter': key, 'True': y_t, 'Pred': y_p, 'Error': err, 'AbsError': np.abs(err)}))
            
        self.results_df = pd.concat(all_res)
        self.results_df.to_csv(self.exp_dir / "eval_metrics_full.csv", index=False)

    def save_visual_report(self, q: torch.Tensor, r_t: torch.Tensor, r_r: torch.Tensor):
        """이해관계자용 종합 시각화 리포트 생성"""
        q_np = q.cpu().numpy()
        sns.set_theme(style="whitegrid", palette="muted")

        # --- Chart 1: The "Classic" Correlation Matrix ---
        # 1:1 산점도를 통해 전체적인 예측 경향성 확인
        keys = list(self.metrics_summary.keys())
        fig1, axes1 = plt.subplots(2, len(keys), figsize=(4 * len(keys), 8))
        for i, key in enumerate(keys):
            sub = self.results_df[self.results_df['Parameter'] == key]
            axes1[0, i].scatter(sub['True'], sub['Pred'], alpha=0.15, s=6, color='#2c3e50')
            mi, ma = min(sub['True'].min(), sub['Pred'].min()), max(sub['True'].max(), sub['Pred'].max())
            axes1[0, i].plot([mi, ma], [mi, ma], 'r--', lw=1.5)
            axes1[0, i].set_title(f"{key}\n$R^2$: {self.metrics_summary[key]['R2']:.3f}", fontsize=12, fontweight='bold')
            sns.histplot(sub['Error'], ax=axes1[1, i], color='#27ae60', kde=True)
            axes1[1, i].axvline(0, color='black', lw=1)
        plt.tight_layout()
        plt.savefig(self.exp_dir / "report_correlation_matrix.png", dpi=150)

        # --- Chart 2: Ranked Performance (Best to Worst) ---
        # 물리적 적합도에 따른 대표 케이스 추출
        sorted_idx = self.sample_metrics.sort_values('log_mse')['idx'].values
        target_idx = [sorted_idx[0], sorted_idx[len(sorted_idx)//2], sorted_idx[int(len(sorted_idx)*0.85)], sorted_idx[-1]]
        titles = ["Best (Exemplary Fit)", "Median (Typical Case)", "Struggling (Challenging Case)", "Worst (Critical Failure)"]
        
        fig2 = plt.figure(figsize=(18, 11))
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.22)
        for i, (idx, title) in enumerate(zip(target_idx, titles)):
            s_gs = gridspec.GridSpecFromSubplotSpec(2, 1, gs[i], height_ratios=[3, 1], hspace=0.05)
            ax_t, ax_b = fig2.add_subplot(s_gs[0]), fig2.add_subplot(s_gs[1])
            yt, yr = r_t[idx].numpy(), r_r[idx].numpy()
            mse = self.sample_metrics.iloc[idx]['log_mse']
            
            ax_t.semilogy(q_np, yt, 'ko', ms=2.5, alpha=0.1, label='Actual')
            ax_t.semilogy(q_np, yr, 'r-', lw=1.8, label='XRefine Fit')
            ax_t.set_title(f"{title} | MSE: {mse:.5f}", fontsize=13, loc='left')
            if i == 0: ax_t.legend()
            
            res = np.log10(np.clip(yr, 1e-11, None)) - np.log10(np.clip(yt, 1e-11, None))
            ax_b.plot(q_np, res, color='#2980b9', lw=1); ax_b.axhline(0, color='black', ls='--', alpha=0.5)
            ax_b.set_ylim(-0.4, 0.4)
            ax_b.set_ylabel("$\\Delta$ LogR", fontsize=9)
        plt.savefig(self.exp_dir / "report_ranked_cases.png", dpi=200)

        # --- Chart 3: Parameter Entanglement Heatmap ---
        plt.figure(figsize=(11, 9))
        df_corr = self.results_df.pivot(columns='Parameter', values='Pred').corr()
        sns.heatmap(df_corr, annot=True, fmt=".2f", cmap='RdBu_r', center=0, square=True)
        plt.title("Parameter Dependency Analysis (Entanglement)", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.exp_dir / "report_entanglement_heatmap.png")
        plt.close('all')

    def report(self):
        """콘솔 출력용 리포트"""
        line = "=" * 88
        logger.info(f"\n{line}\n{'Parameter':<18} | {'MAE':<10} | {'MAPE(%)':<10} | {'RMSE':<10} | {'R2 Score':<10}\n{'-' * 88}")
        for k, v in self.metrics_summary.items():
            logger.info(f"{k:<18} | {v['MAE']:<10.4f} | {v['MAPE']:<10.2f} | {v.get('RMSE', 0):<10.4f} | {v['R2']:<10.4f}")
        logger.info(f"{line}\n")

    def evaluate_all(self):
        """단일 진입점: 전체 파이프라인 자동 실행"""
        rm, pt, pp, rr = self.run_inference()
        self.calculate_statistics(rm, rr, pt, pp)
        self.report()
        self.save_visual_report(self.config.q_grid, rm, rr)
        logger.info(f"Professional Evaluation Complete. Artifacts at: {self.exp_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    evaluator = XRREvaluator(args.checkpoint, args.batch_size)
    evaluator.setup()
    evaluator.evaluate_all()