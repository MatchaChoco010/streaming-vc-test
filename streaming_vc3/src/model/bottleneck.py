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
            FFTBlockST(80, 512),
            FFTBlockST(512, 512),
            FFTBlockST(512, 512),
        )
        self.fc_s = nn.Linear(512, 80)
        self.fc_t = nn.Linear(512, 80)

    def forward(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = self.layers(xs)
        mu = torch.tanh(self.fc_s(xs))
        log_sigma = self.fc_t(xs)
        return mu, log_sigma
