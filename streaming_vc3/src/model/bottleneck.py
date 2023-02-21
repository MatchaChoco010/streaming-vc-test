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
            FFTBlockST(128, 512),
            FFTBlockST(512, 512),
            FFTBlockST(512, 128),
        )
        self.proj_mu = nn.Linear(128, 128)
        self.proj_log_sigma = nn.Linear(128, 128)

    def forward(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = self.layers(xs)
        mu = torch.tanh(self.proj_mu(xs)).transpose(1, 2)
        log_sigma = self.proj_log_sigma(xs).transpose(1, 2)
        return mu, log_sigma
