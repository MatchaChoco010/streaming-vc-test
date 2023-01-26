import torch
import torch.nn as nn
from src.module.fft_block import FFTBlock


class MelGenerateModel(nn.Module):
    """
    FastSpeechのFFTBlockを積み重ねたモデル
    """

    def __init__(self):
        super(MelGenerateModel, self).__init__()
        self.first_layer = FFTBlock(32, 512)
        self.layers = nn.Sequential(
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
        first_xs = self.first_layer(xs)
        xs = self.layers(first_xs)
        return xs.transpose(1, 2)

    def forward_first_layer(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            xs: Tensor (batch, seq_length, input_feature_size)
                入力のオーディオ特徴量
        Returns:
            first_xs: Tensor (batch, seq_length, 512)
                最初のFFTBlockの出力
        """
        first_xs = self.first_layer(xs)
        return first_xs
