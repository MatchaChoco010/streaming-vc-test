import torch
import torch.nn as nn
from src.module.fft_block import FFTBlock


class DiscriminatorFeat(nn.Module):
    """
    FastSpeechのFFTBlockを積み重ねたモデル
    """

    def __init__(self):
        super(DiscriminatorFeat, self).__init__()
        self.layers = nn.Sequential(
            FFTBlock(32, 512),
            FFTBlock(512, 512),
            FFTBlock(512, 512),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        xs = self.layers(xs)
        return xs.transpose(1, 2)


class DiscriminatorMel(nn.Module):
    """
    FastSpeechのFFTBlockを積み重ねたモデル
    """

    def __init__(self):
        super(DiscriminatorMel, self).__init__()
        self.layers = nn.Sequential(
            FFTBlock(80, 512),
            FFTBlock(512, 512),
            FFTBlock(512, 512),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        xs = self.layers(xs.transpose(1, 2))
        return xs.transpose(1, 2)
