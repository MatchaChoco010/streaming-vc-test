import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    話者のDiscriminator
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(32, 512, kernel_size=7),
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
