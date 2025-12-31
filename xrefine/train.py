import torch
import torch.nn as nn
from tqdm import tqdm

from xrefine.model import XRR_PINN_V3
from xrefine.data.generator import generate_synthetic_batch
from xrefine.config import Config

def train_streaming():
    device = Config.DEVICE
    pinn = XRR_PINN_V3(device=device).to(device)
    optimizer = torch.optim.AdamW(pinn.parameters(), lr=Config.LR)

    criterion = nn.MSELoss()

    print("🔥 PINN Streaming Training Started on Colab...")
    
    for step in tqdm(range(10000)):
        r_measured, _ = generate_synthetic_batch(Config.BATCH_SIZE, Config.Q_GRID.to(device), Config.__dict__)

        optimizer.zero_grad()

        params, r_reconstructed = pinn(torch.log10(r_measured + 1e-12))

        loss = criterion(torch.log10(r_reconstructed + 1e-12), torch.log10(r_measured + 1e-12))

        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            tqdm.write(f"Step {step} | Loss: {loss.item():.6f} | d_pred: {params[0][0].item():.2f}")

if __name__ == "__main__":
    train_streaming()