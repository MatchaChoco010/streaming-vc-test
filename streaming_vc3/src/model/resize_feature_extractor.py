import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
import torchvision.transforms.functional
import torchaudio
from convmelspec.stft import ConvertibleSpectrogram as Spectrogram
from src.module.mask import length_mask


class Delta(torch.nn.Module):
    """
    特徴量の微分を特徴量に追加するモジュール。
    速度と加速度をチャンネルとして追加する。
    """

    def __init__(self):
        super(Delta, self).__init__()
        self.order = 2
        self.window_size = 2
        filters = self._create_filters()
        self.register_buffer("filters", filters)
        self.padding = (0, (filters.shape[-1] - 1) // 2)

    def forward(self, xs) -> torch.Tensor:
        """
        Arguments:
            xs: torch.Tensor (batch, feature_size, seq_len)
                入力の特徴量
        Returns:
            xs: torch.Tensor (batch, 3, feature_size, seq_len)
        """
        xs = xs.unsqueeze(1)
        weight: torch.Tensor = self.filters  # type: ignore
        return F.conv2d(xs, weight=weight, padding=self.padding)

    def _create_filters(self) -> torch.Tensor:
        scales = [[1.0]]
        for i in range(1, 3):
            prev_offset = (len(scales[i - 1]) - 1) // 2
            offset = prev_offset + 2

            current = [0.0] * (len(scales[i - 1]) + 2 * 2)
            normalizer = 0.0
            for j in range(-2, 2 + 1):
                normalizer += j * j
                for k in range(-prev_offset, prev_offset + 1):
                    current[k + offset + j] += j * scales[i - 1][k + prev_offset]
            current = [x / normalizer for x in current]
            scales.append(current)

        max_len = len(scales[-1])
        for i, scale in enumerate(scales[:-1]):
            padding = (max_len - len(scale)) // 2
            scales[i] = [0] * padding + scale + [0] * padding

        # out_channel, in_channel, kernel_height, kernel_width
        return torch.tensor(scales).unsqueeze(1).unsqueeze(1)


class FeatureExtractor(nn.Module):
    """
    audioの特徴量を抽出するモジュール。
    """

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.melspec = Spectrogram(
            sr=24000,
            n_fft=1024,
            hop_size=256,
            n_mel=80,
            padding=512,
        )
        self.delta = Delta()

    def forward(self, xs: torch.Tensor, scale: int) -> torch.Tensor:
        """
        Arguments:
            xs: torch.Tensor (batch, max(audio_len))
                オーディオの入力
            scale: int
                スケール後のmelのサイズ（80基準)
        Returns:
            (ys, y_lengths): Tuple[torch.Tensor, torch.Tensor]
            ys: torch.Tensor (batch, seq_len, feature_size)
                特徴量
        """
        ys = self.melspec(xs)[:, :, :-1]

        # メルスペクトログラムをDBベースにするためにlogを取る
        ys = torch.log(torch.clamp(ys, min=1e-5))

        # 縦方向にリサイズする
        height = scale
        ys = torchvision.transforms.functional.resize(
            img=ys, size=(height, ys.shape[2])
        )
        if scale < 80:
            ys = torchvision.transforms.Pad(
                padding=(0, 0, 0, 80 - ys.shape[1]), padding_mode="edge"
            )(ys)
        else:
            ys = ys[:, :80, :]

        ys = self.delta(ys)

        # (batch, channel, feature_size, seq_len) -> (batch, feature_size, seq_len)
        ys = ys.reshape(ys.shape[0], -1, ys.shape[3])

        # (batch, feature_size, seq_len) -> (batch, seq_len, feature_size)
        ys = ys.permute(0, 2, 1)

        return ys
