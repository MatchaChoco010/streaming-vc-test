import torch
from torch import nn
from torch.nn import functional as F

from src.module.wn import WN


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=192,
        hidden_channels=256,
        kernel_size=5,
        dilation_rate=1,
        n_layers=16,
    ):
        super(PosteriorEncoder, self).__init__()
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers)
        self.proj_mu = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj_log_sigma = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x):
        h = self.pre(x)
        h = self.enc(h)
        mu = self.proj_mu(h)
        log_sigma = self.proj_log_sigma(h)
        z = mu + torch.rand_like(mu) * torch.exp(log_sigma)
        return z, mu, log_sigma
