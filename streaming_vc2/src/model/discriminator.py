import torch
import torch.nn as nn
from src.module.fft_block import FFTBlock


class DiscriminatorMel(nn.Module):
    """
    FastSpeechのFFTBlockを積み重ねたモデル
    """

    def __init__(self):
        super(DiscriminatorMel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(80, 512, kernel_size=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, kernel_size=7),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, kernel_size=5),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, kernel_size=5),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 1, kernel_size=3),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        xs = self.conv(xs)
        xs = xs.squeeze(1)
        xs = self.sigmoid(xs)
        return xs


class DiscriminatorFeat(nn.Module):
    """
    FastSpeechのFFTBlockを積み重ねたモデル
    """

    def __init__(self):
        super(DiscriminatorFeat, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(128, 512, kernel_size=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, kernel_size=7),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, kernel_size=5),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, kernel_size=5),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 1, kernel_size=3),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        xs = self.conv(xs.transpose(1, 2))
        xs = xs.squeeze(1)
        xs = self.sigmoid(xs)
        return xs
