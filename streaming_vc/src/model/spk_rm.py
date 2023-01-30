import torch
import torch.nn as nn
from src.module.fft_block import FFTBlockST


class SpeakerRemoval(nn.Module):
    """
    FastSpeechのFFTBlockを積み重ねたモデル
    """

    def __init__(self):
        super(SpeakerRemoval, self).__init__()
        self.layers = nn.Sequential(
            FFTBlockST(32, 512),
            FFTBlockST(512, 512),
            FFTBlockST(512, 512),
        )
        self.bn = nn.BatchNorm1d(512)
        self.after_layer = nn.Linear(512, 32)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        xs = self.layers(xs)
        xs = self.bn(xs.transpose(1, 2))
        xs = self.after_layer(xs.transpose(1, 2))
        # TODO Sigmoid
        return xs
