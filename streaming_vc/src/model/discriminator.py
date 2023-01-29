import torch
import torch.nn as nn
from src.module.fft_block import FFTBlock


class Discriminator(nn.Module):
    """
    FastSpeechのFFTBlockを積み重ねたモデル
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            FFTBlock(80, 512),
            FFTBlock(512, 512),
            FFTBlock(512, 512),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        xs = self.layers(xs)
        return xs.transpose(1, 2)
