import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    話者のDiscriminator
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(32, 512, 1, 7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, 1, 7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, 1, 7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, 1, 7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 1, 1, 7),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            xs: Tensor (batch, seq_len, 32)
                入力のオーディオ特徴量
        Returns:
            xs: Tensor (batch, seq_len / 32, 1)
                出力の特徴量
        """
        xs = self.conv(xs.transpose(1, 2))
        xs = xs.squeeze(1)
        xs = self.sigmoid(xs)
        return xs
