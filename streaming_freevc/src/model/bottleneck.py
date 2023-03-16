from typing import Tuple

import torch
import torch.nn as nn
from src.module.fft_block import FFTBlockST


class Bottleneck(nn.Module):
    """
    FastSpeechのFFTBlockを積み重ねたモデル
    """

    def __init__(self):
        super(Bottleneck, self).__init__()
        self.layers = nn.Sequential(
            FFTBlockST(512, 1024),
            nn.LeakyReLU(),
            FFTBlockST(1024, 1024),
            nn.LeakyReLU(),
            FFTBlockST(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 192),
        )
        self.bn = nn.BatchNorm1d(192)
        self.proj_mu = nn.Conv1d(192, 256, 1)
        self.proj_log_sigma = nn.Conv1d(192, 256, 1)

    def forward(
        self, xs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xs = self.bn(self.layers(xs).transpose(1, 2))
        mu = self.proj_mu(xs)
        log_sigma = self.proj_log_sigma(xs)
        z = mu + torch.rand_like(mu) * torch.exp(log_sigma)
        return z, mu, log_sigma
