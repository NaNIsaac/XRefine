import torch
import torch.nn as nn
from xrefine.simulator.interface import Simulator

class XRRParameterEncoder(nn.Module):
    def __init__(self, n_layers=1):
        super().__init__()
        self.n_layers = n_layers

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(8)
        )

        self.regressor = nn.Linear(64 * 8, 3 * n_layers + 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        feat = self.feature_extractor(x).view(x.size(0), -1)
        params = torch.sigmoid(self.regressor(feat))
        
        thick = params[:, :self.n_layers] * 1000.0
        rough = params[:, self.n_layers : 2*self.n_layers+1] * 30.0
        sld = params[:, 2*self.n_layers+1 :] * 100.0
        
        return thick, rough, sld

class XRR_PINN(nn.Module):
    def __init__(self, q_grid, n_layers=1, device='cpu'):
        super().__init__()
        self.q_grid = q_grid.to(device)
        self.encoder = XRRParameterEncoder(n_layers=n_layers)
        self.simulator = Simulator(method='abeles', device=device)
        self.device = device

    def forward(self, r_obs):
        thick, rough, sld = self.encoder(r_obs)
        
        q_batch = self.q_grid.unsqueeze(0).repeat(r_obs.size(0), 1)
        
        r_sim = self.simulator(q_batch, thick, rough, sld)
        
        return (thick, rough, sld), r_sim

def train_pinn_step(pinn_model, r_measured, optimizer):
    pinn_model.train()
    optimizer.zero_grad()

    params, r_reconstructed = pinn_model(r_measured)

    loss = nn.MSELoss()(torch.log10(r_measured + 1e-12), torch.log10(r_reconstructed + 1e-12))

    loss.backward()
    optimizer.step()

    return loss.item(), params
