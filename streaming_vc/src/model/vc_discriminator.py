import torch
import torch.nn as nn


class VCDiscriminator(nn.Module):
    """
    MLPによるDiscriminator
    """

    def __init__(self):
        super(VCDiscriminator, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Linear(80, 512),
            nn.Linear(512, 512),
        )
        self.conv = nn.Conv1d(512, 1, 1)
        self.layers2 = nn.Sequential(
            nn.Linear(128, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 1),
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            xs: Tensor (batch, 128, 80)
                入力のオーディオ特徴量
        Returns:
            xs: Tensor (batch, 1)
                出力の特徴量
        """
        xs = self.layers1(xs.transpose(1, 2))
        xs = self.conv(xs.transpose(1, 2)).squeeze(1)
        xs = self.layers2(xs)
        return xs
