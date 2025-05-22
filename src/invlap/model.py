import torch
import torch.nn as nn
import torch.nn.functional as F


class MassNN(nn.Module):
    def __init__(self, T, N):
        super().__init__()
        self.T = T
        self.N = N 

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3),
            nn.Tanh(),
            nn.Flatten()
        )
        
        self.dense_net = nn.Sequential(
            nn.Linear((T-2) * 5, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2*N)
        )

    def forward(self, x):
        y = x.unsqueeze(1)
        y = self.conv(y)
        y = self.dense_net(y)

        A = F.softplus(y[:, :self.N])
        e = y[:,self.N:]

        dE = F.softplus(e)
        E = torch.cumsum(dE, dim=1)

        return A, E
    


    