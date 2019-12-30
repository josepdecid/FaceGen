import torch
from torch import nn


class MSEKLDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, x1, x2, mu, log_var):
        MSE = self.mse_loss(x1, x2)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD
