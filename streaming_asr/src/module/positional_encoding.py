import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional Encodingモジュール。
    """

    def __init__(self, d_model: int):
        """
        Arguments:
            d_model: int
                モデルの次元数
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xs_scale = math.sqrt(self.d_model)
        self.pe: torch.Tensor | None = None

    def _extend_pe(self, xs: torch.Tensor):
        """
        内部に保存しているPositional Encodingをxsと同じサイズまで拡張する

        Arguments:
            xs: torch.Tensor (batch_size, seq_len, d_model)
                入力の特徴量
        """
        # すでに保存しているPositional Encodingがxsと同じサイズ以上の場合
        if self.pe is not None:
            if self.pe.size(1) >= xs.size(1):
                if self.pe.dtype != xs.dtype or self.pe.device != xs.device:
                    self.pe = self.pe.to(dtype=xs.dtype, device=xs.device)
                return

        # 保存しているPositional Encodingがxsより小さい場合、新たにpeを作る
        pe = torch.zeros(xs.size(1), self.d_model)
        position = torch.arange(0, xs.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=xs.device, dtype=xs.dtype)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            xs: torch.Tensor (batch_size, seq_len, d_model)
                入力の特徴量
        Returns:
            xs: torch.Tensor (batch_size, seq_len, d_model)
                Positional Encodingを加えた特徴量
        """
        self._extend_pe(xs)
        return xs * self.xs_scale + self.pe[:, : xs.size(1)]
