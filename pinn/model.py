# Defining our model
import torch
import torch.nn as nn

class MLP(nn.Module): 
    def __init__(self, in_dim=2, out_dim=2, hidden_dim=64, n_layers=6):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for i in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        # Ensuring no issues with shape of our tensors
        if x.ndim == 1:
            x = x.unsqueeze(1)
        if t.ndim == 1:
            t = t.unsqueeze(1)
        inp = torch.cat((x, t), dim=1)
        out = self.net(inp)  # [N, 2]
        rho = out[:, 0:1]  # [N, 1]
        u = out[:, 1:2]    # [N, 1]
        return rho, u