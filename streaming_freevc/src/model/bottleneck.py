from typing import Tuple

import torch
import torch.nn as nn
from src.module.fft_block import FFTBlockST, FFTBlock


class Bottleneck(nn.Module):
    """
    FastSpeechのFFTBlockを積み重ねたモデル
    """

    def __init__(self):
        super(Bottleneck, self).__init__()
        # self.layers = nn.Sequential(
        #     FFTBlockST(512, 1024),
        #     nn.LeakyReLU(),
        #     FFTBlockST(1024, 1024),
        #     nn.LeakyReLU(),
        #     FFTBlockST(1024, 1024),
        #     nn.LeakyReLU(),
        #     nn.Linear(1024, 32),
        # )
        # self.bn = nn.BatchNorm1d(32)
        # self.proj_mu = nn.Conv1d(32, 256, 1)
        # self.proj_log_sigma = nn.Conv1d(32, 256, 1)
        self.pre_conv = nn.Conv1d(1024, 192, kernel_size=5, padding=2)
        self.f0_embed = nn.Embedding(256, 192)
        self.atten = nn.MultiheadAttention(192, 4)
        self.proj_mu = nn.Conv1d(192, 192, 1)
        self.proj_log_sigma = nn.Conv1d(192, 192, 1)

    def forward(
        self,
        xs: torch.Tensor,
        f0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # xs = self.bn(self.layers(xs).transpose(1, 2))
        xs = xs.transpose(1, 2)
        xs = self.pre_conv(xs)
        xs = xs + self.f0_embed(f0).transpose(1, 2)[:, :, : xs.size(2)]
        xs = xs.transpose(1, 2)
        xs = self.atten(xs, xs, xs)[0]
        xs = xs.transpose(1, 2)
        mu = self.proj_mu(xs)
        log_sigma = self.proj_log_sigma(xs)
        z = mu + torch.rand_like(mu) * torch.exp(log_sigma)
        return z, mu, log_sigma


class BottleneckDiscriminator(nn.Module):
    def __init__(self):
        super(BottleneckDiscriminator, self).__init__()
        self.proj_mu = nn.Conv1d(256, 1024, 1)
        self.proj_sigma = nn.Conv1d(256, 1024, 1)
        self.layers = nn.Sequential(
            FFTBlock(1024, 1024),
            nn.LeakyReLU(),
            FFTBlock(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        xs = self.proj_mu(mu) + self.proj_sigma(sigma)
        return self.layers(xs.transpose(1, 2))
