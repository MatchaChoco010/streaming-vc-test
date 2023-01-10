import torch
import torch.nn.functional as F


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_logmel, y_logmel):
        """
        Arguments:
            x_logmel: torch.Tensor (batch_size, seq_len, mel_size)
                logメルスペクトログラムの推定値
            y_logmel: torch.Tensor (batch_size, seq_len, mel_size)
                logメルスペクトログラムの正解値
        Returns:
            loss: torch.Tensor (batch_size,)
                spectral convergence loss
        """
        y_mag = torch.exp(y_logmel)
        x_mag = torch.exp(x_logmel)
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
