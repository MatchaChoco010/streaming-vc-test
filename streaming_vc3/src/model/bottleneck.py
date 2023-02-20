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
            FFTBlockST(512, 512),
        )
        self.fc_s = nn.Linear(512, 40)
        self.fc_t = nn.Linear(512, 40)

    def forward(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = self.layers(xs)
        mu = torch.tanh(self.fc_s(xs)).transpose(1, 2)
        log_sigma = self.fc_t(xs).transpose(1, 2)
        return mu, log_sigma
