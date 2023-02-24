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
            nn.Linear(512, 24),
        )
        self.proj_mu = nn.Conv1d(24, 256, 1)
        self.proj_log_sigma = nn.Conv1d(24, 256, 1)

    def forward(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = self.layers(xs).transpose(1, 2)
        mu = self.proj_mu(xs)
        log_sigma = self.proj_log_sigma(xs)
        return mu, log_sigma
