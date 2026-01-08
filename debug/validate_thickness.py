import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from scipy.signal import find_peaks, detrend

from xrefine.config import XRefineConfig
from xrefine.models import XRR_Fourier_PINN
from xrefine.core.structures import ParamSet
from xrefine.predict_real import load_real_data, _optimize_subset
from xrefine.logger import logger

class ThicknessValidator:
    def __init__(self, checkpoint_path: str, data_path: str):
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Model & Config
        self.ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.config = XRefineConfig(**self.ckpt['config'])
        self.model = XRR_Fourier_PINN(config=self.config).to(self.device)
        self.model.load_state_dict(self.ckpt['model_state_dict'])
        self.model.eval()
        
        # Load Data
        self.log_r_obs = load_real_data(data_path, self.config).to(self.device)
        self.q_grid = self.config.q_grid.to(self.device)
        
        # Prepare Mask (Typical valid range)
        self.mask = (self.log_r_obs > -14.0) & (self.q_grid > 0.03) & (self.q_grid < 0.45)
        
    def _get_base_params(self):
        """AI 초기값 + Background 최적화된 파라미터 반환"""
        with torch.no_grad():
            p_ai, _, _ = self.model(self.log_r_obs)
            
        # Background & Footprint Optimization
        p_base = _optimize_subset(
            self.model, p_ai, self.log_r_obs, self.config,
            optimize_keys={'beam_width', 'L', 'i0', 'bkg'},
            q_max=0.15, maxiter=30, popsize=10
        )
        return p_base

    def scan_loss_landscape(self, d_range=np.linspace(10, 1200, 500)):
        """Correlation vs Loss Landscape 스캔"""
        p_scan = self._get_base_params()
        # Oxide 제거 (순수 Film 두께만 보기 위함)
        p_scan.d_ox.fill_(0.0)
        p_scan.sig_ox.fill_(0.0)
        p_scan.rho_ox.fill_(0.0)
        p_scan.sig_f.fill_(3.0) # 고정 Roughness
        
        results = []
        
        for d_val in d_range:
            p_scan.d_f.fill_(d_val)
            
            with torch.no_grad():
                _, r_sim, _ = self.model.forward_with_params(p_scan)
                log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
                
                # 1. MSE
                mse = torch.mean(((log_r_sim - self.log_r_obs)**2) * self.mask).item()
                
                # 2. Correlation (Inverse)
                sim_m = log_r_sim[self.mask]
                obs_m = self.log_r_obs[self.mask]
                vx = sim_m - sim_m.mean()
                vy = obs_m - obs_m.mean()
                corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-9)
                corr_inv = 1.0 - corr.item()
                
                results.append({
                    'd': d_val,
                    'mse': mse,
                    'corr': corr_inv
                })
                
        return results

    def analyze_fft(self):
        """FFT 기반 두께 추정"""
        q = self.q_grid.cpu().numpy()
        log_r = self.log_r_obs.cpu().numpy().flatten()
        
        # Valid region only
        valid = (log_r > -14.0) & (q > 0.05) & (q < 0.35)
        q_v = q[valid]
        r_v = log_r[valid]
        
        # Detrend
        r_detrend = detrend(r_v)
        
        # FFT
        n = len(q_v)
        dq = np.mean(np.diff(q_v))
        fft_vals = np.fft.fft(r_detrend)
        freq = np.fft.fftfreq(n, d=dq)
        
        mask = freq > 0
        freq = freq[mask]
        power = np.abs(fft_vals[mask])**2
        
        # Freq to Thickness: d = 2π * freq
        thickness = 2 * np.pi * freq
        
        # Find Peaks
        peaks, _ = find_peaks(power, prominence=np.max(power)*0.1)
        peak_info = [(thickness[p], power[p]) for p in peaks]
        peak_info.sort(key=lambda x: x[1], reverse=True)
        
        return peak_info, (thickness, power)

    def run_validation(self, output_dir='debug_output'):
        """종합 검증 실행"""
        results = self.scan_loss_landscape()
        
        d_vals = [r['d'] for r in results]
        corr_vals = [r['corr'] for r in results]
        mse_vals = [r['mse'] for r in results]
        
        # Find Minima
        corr_peaks, _ = find_peaks(-np.array(corr_vals), prominence=0.005)
        mse_peaks, _ = find_peaks(-np.array(mse_vals), prominence=0.005)
        
        corr_cands = [(d_vals[i], corr_vals[i]) for i in corr_peaks]
        mse_cands = [(d_vals[i], mse_vals[i]) for i in mse_peaks]
        
        corr_cands.sort(key=lambda x: x[1])
        mse_cands.sort(key=lambda x: x[1])
        
        # FFT Validation
        fft_peaks, (d_fft, p_fft) = self.analyze_fft()
        
        # Print Report
        print(f"\n===== Thickness Validation: {Path(self.data_path).name} =====")
        print(f"{'Correlation Minima (Top 3)':<35} | {'MSE Minima (Top 3)':<35}")
        print("-" * 75)
        for i in range(3):
            c_str = f"{corr_cands[i][0]:.1f}A ({corr_cands[i][1]:.4f})" if i < len(corr_cands) else ""
            m_str = f"{mse_cands[i][0]:.1f}A ({mse_cands[i][1]:.4f})" if i < len(mse_cands) else ""
            print(f"{c_str:<35} | {m_str:<35}")
            
        print("\n[FFT Peaks]")
        for d, p in fft_peaks[:3]:
            print(f"  d ≈ {d:.1f}A (power={p:.1f})")
            
        self._plot_results(d_vals, corr_vals, mse_vals, corr_cands, mse_cands, output_dir)
        
    def _plot_results(self, d, corr, mse, c_cands, m_cands, out_dir):
        plt.figure(figsize=(10, 8))
        
        # 1. Norm Landscape
        plt.subplot(2, 1, 1)
        c_norm = (corr - np.min(corr)) / (np.max(corr) - np.min(corr))
        m_norm = (mse - np.min(mse)) / (np.max(mse) - np.min(mse))
        
        plt.plot(d, c_norm, 'g-', label='Correlation (1-r)', alpha=0.8)
        plt.plot(d, m_norm, 'r--', label='MSE Loss', alpha=0.6)
        
        # Mark Top 1
        if c_cands:
            plt.axvline(c_cands[0][0], color='g', alpha=0.5, linestyle=':')
            plt.text(c_cands[0][0], 0, f"{c_cands[0][0]:.0f}A", color='g', fontweight='bold')
            
        plt.title(f"Landscape Comparison: {Path(self.data_path).name}")
        plt.xlabel("Thickness (Å)")
        plt.ylabel("Normalized Score")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 2. Fit of Top 1 Corr vs Top 1 MSE
        plt.subplot(2, 1, 2)
        q = self.q_grid.cpu().numpy()
        obs = self.log_r_obs.cpu().numpy().flatten()
        plt.plot(q, obs, 'k.', alpha=0.2, label='Observed')
        
        # Plot Top 1 Corr Candidate
        if c_cands:
            d_best = c_cands[0][0]
            sim = self._get_sim_for_d(d_best)
            plt.plot(q, sim, 'g-', linewidth=1.5, label=f'Best Corr: {d_best:.0f}A')
            
        plt.legend()
        plt.title("Fit Comparison")
        plt.grid(alpha=0.3)
        
        Path(out_dir).mkdir(exist_ok=True)
        save_path = Path(out_dir) / f"validate_{Path(self.data_path).stem}.png"
        plt.savefig(save_path)
        print(f"\n📊 Plot saved to: {save_path}")
        plt.close()

    def _get_sim_for_d(self, d_val):
        p = self._get_base_params()
        p.d_f.fill_(d_val)
        p.sig_f.fill_(3.0)
        p.d_ox.fill_(0.0)
        with torch.no_grad():
            _, r, _ = self.model.forward_with_params(p)
        return torch.log10(torch.clamp(r, min=1e-12)).cpu().numpy().flatten()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    args = parser.parse_args()
    
    CHECKPOINT = r"D:\03_Resources\Data\XRR_AI\xrefine\v1\best.pt"
    validator = ThicknessValidator(CHECKPOINT, args.data)
    validator.run_validation()
