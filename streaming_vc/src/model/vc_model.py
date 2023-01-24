import torch
import torch.nn as nn
from src.module.fft_block import FFTBlock
from src.module.log_melspectrogram import log_melspectrogram


class VCModel(nn.Module):
    """
    FastSpeechのFFTBlockを積み重ねたモデル
    """

    def __init__(self):
        super(VCModel, self).__init__()
        self.layers = nn.Sequential(
            FFTBlock(128, 512),
            FFTBlock(512, 512),
            FFTBlock(512, 512),
            FFTBlock(512, 512),
            nn.Linear(512, 80),
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            xs: Tensor (batch, seq_length, input_feature_size)
                入力のオーディオ特徴量
        Returns:
            xs: Tensor (batch, mel_size, seq_length)
                出力の特徴量
        """
        xs = self.layers(xs)
        return xs.transpose(1, 2)
