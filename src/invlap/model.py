import torch
import torch.nn as nn
import torch.nn.functional as F


class MassNN(nn.Module):
    def __init__(self, T, N):
        super().__init__()
        self.T = T
        self.N = N 

        self.net = nn.Sequential(
            nn.Linear(T, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2*N)
        )

    def forward(self, x):
        y = self.net(x)

        A = F.softplus(y[:, :self.N])
        e = y[:,self.N:]

        dE = F.softplus(e)
        E = torch.cumsum(dE, dim=1)

        return A, E
    