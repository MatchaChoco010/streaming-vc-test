import torch
from torch import nn

from src.module.wn import WN


class ResidualCouplingLayer(nn.Module):
    def __init__(self):
        super(ResidualCouplingLayer, self).__init__()
        self.pre = nn.Conv1d(128, 512, 1)
        self.enc = WN(hidden_dim=512, kernel_size=5, dilation_rate=1, n_layers=4)
        self.post = nn.Conv1d(512, 256, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)

        h = self.pre(x1)
        h = self.enc(h)
        h = self.post(h)
        mu, log_sigma = h.chunk(2, dim=1)

        x2 = mu + x2 * torch.exp(log_sigma)

        x = torch.cat([x1, x2], dim=1)
        logdet = torch.sum(log_sigma, [1, 2])

        return x, logdet

    def reverse(self, z):
        x1, x2 = z.chunk(2, dim=1)

        h = self.pre(x1)
        h = self.enc(h)
        h = self.post(h)
        mu, log_sigma = h.chunk(2, dim=1)

        x2 = (x2 - mu) * torch.exp(-log_sigma)

        x = torch.cat([x1, x2], dim=1)

        return x


class Flip(nn.Module):
    def forward(self, x):
        x = torch.flip(x, [1])
        logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
        return x, logdet

    def reverse(self, x):
        x = torch.flip(x, [1])
        return x


class ResidualCouplingBlock(nn.Module):
    def __init__(self):
        super(ResidualCouplingBlock, self).__init__()
        self.flows = nn.ModuleList()
        for _ in range(4):
            self.flows.append(ResidualCouplingLayer())
            self.flows.append(Flip())

    def forward(self, x):
        for flow in self.flows:
            x, _ = flow(x)
        return x

    def reverse(self, z):
        for flow in reversed(self.flows):
            z = flow.reverse(z)
        return z
