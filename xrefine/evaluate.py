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

from xrefine.models import XRR_Fourier_PINN, XRR_Transformer_PINN, XRR_ViT_PINN
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
        self.model = None # Type depends on config
        self.results_df: Optional[pd.DataFrame] = None
        self.sample_metrics: Optional[pd.DataFrame] = None 
        self.metrics_summary: Dict[str, Dict] = {}

    def setup(self):
        """환경 설정 및 모델 로드"""
        logger.info(f"🔍 Initializing Evaluator from {self.checkpoint_path}...")
        import pathlib
        torch.serialization.add_safe_globals([pathlib.WindowsPath, pathlib.PosixPath, XRefineConfig])

        # 1. 체크포인트 로드
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # 2. Config 복원 (여기서 100k 설정이 불러와짐)
        match ckpt.get('config'):
            case dict() as d:
                self.config = XRefineConfig(**d)
            case XRefineConfig() as c:
                self.config = c
            case _:
                raise ValueError("Invalid config format in checkpoint")

        # [FIX] 체크포인트 config를 그대로 사용 (q_len 포함)
        # 체크포인트의 q_grid를 재생성하여 device에 올림
        self.config.q_grid = self.config.q_grid.to(self.device)

        # 3. 모델 로드 (model_type에 따라 선택)
        model_type = getattr(self.config.model, 'model_type', 'cnn').lower()
        
        # [Compatibility] Fourier 없이 학습된 체크포인트 감지
        # 체크포인트에 'fourier.B' 키가 없으면 mapping_size=0으로 강제
        state_dict = ckpt['model_state_dict']
        if model_type == 'vit':
            has_fourier = 'fourier.B' in state_dict
            if not has_fourier:
                logger.info("⚠️ Checkpoint trained without Fourier. Setting mapping_size=0")
                self.config.model.mapping_size = 0
            self.model = XRR_ViT_PINN(config=self.config).to(self.device)
        elif model_type == 'transformer':
            self.model = XRR_Transformer_PINN(config=self.config).to(self.device)
        else:
            self.model = XRR_Fourier_PINN(config=self.config, device=self.device).to(self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        logger.info(f"Setup complete. Grid: {self.config.q_len} pts, Fourier: {getattr(self.config.model, 'mapping_size', 0) > 0}")

    @torch.no_grad()
    def run_inference(self) -> Tuple[torch.Tensor, ParamSet, ParamSet, torch.Tensor]:
        """배치 분할 추론 수행"""
        q_grid = self.config.q_grid.to(self.device)
        augmentor = XRRAugmentor(self.config).to(self.device)
        
        split_size = 50 # OOM 방지용 작은 배치
        num_iters = self.batch_size // split_size
        if self.batch_size % split_size != 0: num_iters += 1
        
        r_m_list, r_r_list, p_t_list, p_p_list = [], [], [], []

        for _ in tqdm(range(num_iters), desc="Running Inference"):
            # Generator는 CPU/GPU 호환되게 설계됨
            r_m, p_t = generate_synthetic_batch(split_size, q_grid, self.config, augmenter=augmentor)
            
            # GPU 이동
            r_m = r_m.to(self.device)
            p_t.to(self.device)
            
            log_r_in = torch.log10(torch.clamp(r_m, min=1e-11))
            
            # Forward Pass (Tuple Unpacking 처리)
            # Transformer: (p, r, (mu, logvar, penalty))
            # CNN: (p, r, penalty)
            out = self.model(log_r_in)
            p_p, r_r = out[0], out[1] # 첫 두 개는 항상 Params, Recon

            r_m_list.append(r_m.to("cpu"))
            r_r_list.append(r_r.to("cpu"))
            p_t_list.append(p_t.to("cpu"))
            p_p_list.append(p_p.to("cpu"))

        # Concatenate Results
        # ParamSet.cat 구현에 의존
        return torch.cat(r_m_list), ParamSet.cat(p_t_list), ParamSet.cat(p_p_list), torch.cat(r_r_list)

    def calculate_statistics(self, r_t: torch.Tensor, r_r: torch.Tensor, p_t: ParamSet, p_p: ParamSet):
        """다차원 통계 지표 산출"""
        logger.info("📊 Calculating global and per-sample statistics...")
        
        # 1. Physics Fidelity: Log-MSE
        log_t = torch.log10(torch.clamp(r_t, min=1e-11))
        log_r = torch.log10(torch.clamp(r_r, min=1e-11))
        mse_per_sample = torch.mean((log_t - log_r)**2, dim=1).numpy()
        
        # Clip batch size if needed (last batch might be smaller)
        actual_size = len(mse_per_sample)
        self.sample_metrics = pd.DataFrame({'idx': np.arange(actual_size), 'log_mse': mse_per_sample})

        # 2. Parameter Accuracy
        all_res = []
        for fld in ParamSet.names():
            key = fld
            # Detach and convert to numpy
            val_t = getattr(p_t, key)
            val_p = getattr(p_p, key)
            
            if isinstance(val_t, torch.Tensor): val_t = val_t.detach().cpu().numpy()
            if isinstance(val_p, torch.Tensor): val_p = val_p.detach().cpu().numpy()
            
            y_t = val_t.flatten()
            y_p = val_p.flatten()
            
            err = y_p - y_t
            
            self.metrics_summary[key] = {
                "MAE": np.mean(np.abs(err)),
                "MAPE": np.mean(np.abs(err) / (np.abs(y_t) + 1e-9)) * 100,
                "RMSE": np.sqrt(np.mean(err**2)),
                "R2": 1 - (np.sum(err**2) / (np.sum((y_t - np.mean(y_t))**2) + 1e-9))
            }
            all_res.append(pd.DataFrame({'Parameter': key, 'True': y_t, 'Pred': y_p, 'Error': err, 'AbsError': np.abs(err)}))
            
        self.results_df = pd.concat(all_res)
        self.results_df.to_csv(self.exp_dir / "eval_metrics_full.csv", index=False)

    def save_visual_report(self, q: torch.Tensor, r_t: torch.Tensor, r_r: torch.Tensor):
        """이해관계자용 종합 시각화 리포트 생성"""
        q_np = q.cpu().numpy()
        sns.set_theme(style="whitegrid", palette="muted")

        # --- Chart 1: Correlation Matrix ---
        keys = list(self.metrics_summary.keys())
        n_cols = 4 # 한 줄에 4개씩
        n_rows = (len(keys) + n_cols - 1) // n_cols
        
        fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes1 = axes1.flatten()
        
        for i, key in enumerate(keys):
            if i >= len(axes1): break
            sub = self.results_df[self.results_df['Parameter'] == key]
            
            ax = axes1[i]
            ax.scatter(sub['True'], sub['Pred'], alpha=0.15, s=6, color='#2c3e50')
            
            mi, ma = min(sub['True'].min(), sub['Pred'].min()), max(sub['True'].max(), sub['Pred'].max())
            ax.plot([mi, ma], [mi, ma], 'r--', lw=1.5)
            
            r2 = self.metrics_summary[key]['R2']
            ax.set_title(f"{key}\nR²: {r2:.3f}", fontsize=11, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=8)
            
        plt.tight_layout()
        plt.savefig(self.exp_dir / "report_correlation_matrix.png", dpi=150)
        plt.close()

        # --- Chart 2: Ranked Performance ---
        if self.sample_metrics is not None and not self.sample_metrics.empty:
            sorted_idx = self.sample_metrics.sort_values('log_mse')['idx'].values
            n_samples = len(sorted_idx)
            
            # Select 4 representative cases
            target_idx = [
                sorted_idx[0], 
                sorted_idx[n_samples//2], 
                sorted_idx[int(n_samples*0.85)], 
                sorted_idx[-1]
            ]
            titles = ["Best Fit", "Median Fit", "Top 15% Error", "Worst Fit"]
            
            fig2 = plt.figure(figsize=(18, 11))
            gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.22)
            
            for i, (idx, title) in enumerate(zip(target_idx, titles)):
                s_gs = gridspec.GridSpecFromSubplotSpec(2, 1, gs[i], height_ratios=[3, 1], hspace=0.05)
                ax_t, ax_b = fig2.add_subplot(s_gs[0]), fig2.add_subplot(s_gs[1])
                
                yt, yr = r_t[idx].numpy(), r_r[idx].numpy()
                mse = self.sample_metrics.iloc[idx]['log_mse']
                
                ax_t.semilogy(q_np, yt, 'ko', ms=2.5, alpha=0.3, label='Observed')
                ax_t.semilogy(q_np, yr, 'r-', lw=1.5, alpha=0.9, label='Predicted')
                ax_t.set_title(f"{title} | Log-MSE: {mse:.5f}", fontsize=12, loc='left', fontweight='bold')
                if i == 0: ax_t.legend()
                ax_t.grid(True, which='both', alpha=0.2)
                
                # Residuals
                res = np.log10(np.clip(yr, 1e-12, None)) - np.log10(np.clip(yt, 1e-12, None))
                ax_b.plot(q_np, res, color='#2980b9', lw=1)
                ax_b.axhline(0, color='black', ls='--', alpha=0.5)
                ax_b.set_ylim(-0.5, 0.5)
                ax_b.set_ylabel("Error (Log)", fontsize=9)
                ax_b.grid(True, alpha=0.2)
                
            plt.savefig(self.exp_dir / "report_ranked_cases.png", dpi=200)
            plt.close()

        # --- Chart 3: Entanglement Heatmap ---
        # Limit correlation calculation to numerical columns only
        try:
            plt.figure(figsize=(12, 10))
            pivot_df = self.results_df.pivot(columns='Parameter', values='Pred')
            # 같은 인덱스끼리만 상관관계를 봐야 하므로 인덱스 정렬 필요할 수 있음 (concat 순서대로임)
            
            df_corr = pivot_df.corr()
            mask = np.triu(np.ones_like(df_corr, dtype=bool))
            
            sns.heatmap(df_corr, annot=True, fmt=".2f", cmap='RdBu_r', center=0, 
                        square=True, mask=mask, cbar_kws={"shrink": .8})
            plt.title("Parameter Entanglement (Correlation of Predictions)", fontsize=14)
            plt.tight_layout()
            plt.savefig(self.exp_dir / "report_entanglement_heatmap.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Skipping heatmap generation due to error: {e}")

    def report(self):
        lines = [f"{'Parameter':<18} | {'MAE':<10} | {'MAPE(%)':<10} | {'RMSE':<10} | {'R2 Score':<10}\n{'-' * 95}"]
        for k, v in self.metrics_summary.items():
            lines.append(f"{k:<18} | {v['MAE']:<10.4f} | {v['MAPE']:<10.2f} | {v.get('RMSE', 0):<10.4f} | {v['R2']:<10.4f}")
        lines.append("=" * 95)
        logger.info("\n"+"\n".join(lines))

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