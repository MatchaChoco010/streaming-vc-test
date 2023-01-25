import torch
import torch.nn as nn


class VCDiscriminator(nn.Module):
    """
    話者のDiscriminator
    """

    def __init__(self):
        super(VCDiscriminator, self).__init__()
        self.conv = nn.Conv1d(512, 1, 7)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            xs: Tensor (batch, seq_len, 512)
                入力のオーディオ特徴量
        Returns:
            xs: Tensor (batch, seq_len, 1)
                出力の特徴量
        """
        xs = self.conv(xs.transpose(1, 2))
        xs = self.sigmoid(xs)
        return xs
