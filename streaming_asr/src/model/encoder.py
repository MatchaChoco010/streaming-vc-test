from typing import Tuple

import torch
import torch.nn as nn
from src.module.mask import chunk_mask, length_mask
from src.module.multi_head_attention import MultiHeadAttention
from src.module.positional_encoding import PositionalEncoding

CHUNK_SIZE = 6


class EncoderLayer(nn.Module):
    """
    ASR Transformerのエンコーダーの一つのレイヤー。
    self attentionとfeed forwardをresidual結合したレイヤーとなっている。
    """

    def __init__(self, feature_size: int):
        """
        Arguments:
            feature_size: int
                入力の特徴量の次元数
        """
        super(EncoderLayer, self).__init__()
        self.out_feature_dim = feature_size

        # Attention層
        self.norm1 = nn.LayerNorm(feature_size)
        self.attention = MultiHeadAttention(
            4, feature_size, feature_size, feature_size, feature_size
        )
        self.dropout1 = nn.Dropout(0.1)

        # Feed Forward層
        self.norm2 = nn.LayerNorm(feature_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, feature_size),
        )
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, xs, mask) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            xs: torch.Tensor (batch_size, seq_len, feature_size)
                入力の特徴量
            mask: torch.Tensor (batch_size, seq_len)
                Attentionのマスク
        Returns:
            (xs, mask): Tuple[torch.Tensor, torch.Tensor]

            xs: torch.Tensor (batch_size, seq_len, feature_size)
                出力の特徴量
            mask: torch.Tensor (batch_size, 1, seq_len)
                出力の長さのマスク
        """
        # batch_size = xs.shape[0]
        # seq_length = xs.shape[1]

        # self attentionの計算
        residual = xs
        xs = self.norm1(xs)
        xs = residual + self.dropout1(self.attention(xs, xs, xs, mask))

        # feed forwardの計算
        residual = xs
        xs = self.norm2(xs)
        xs = residual + self.dropout2(self.feed_forward(xs))

        # assert xs.size() == (batch_size, seq_length, self.out_feature_dim)
        # assert mask.size() == (batch_size, seq_length, seq_length)

        return xs, mask


class Encoder(nn.Module):
    """
    ASR Transformerのエンコーダー。
    self attentionとfeed forwardを繰り返す構造となっている。
    """

    def __init__(self, input_feature_size: int, decoder_feature_size: int):
        """
        Arguments:
            input_feature_size: int
                入力のオーディオ特徴量の次元数
            decoder_feature_size: int
                出力の特徴量の次元数
        """
        super(Encoder, self).__init__()
        self.out_feature_dim = decoder_feature_size

        # 入力のembedding
        self.embed = nn.Sequential(
            nn.Linear(input_feature_size, self.out_feature_dim),
            nn.LayerNorm(self.out_feature_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            PositionalEncoding(self.out_feature_dim),
        )

        # # Encoderのレイヤー
        encoder_modules = []
        for _ in range(6):
            encoder_modules.append(EncoderLayer(feature_size=self.out_feature_dim))
        self.encoders = nn.ModuleList(encoder_modules)

        # Normalizationレイヤー
        self.after_norm = nn.LayerNorm(self.out_feature_dim)

    def forward(
        self, input_features: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            input_features: Tensor (batch, seq_length, input_feature_size)
                入力のオーディオ特徴量
            input_lengths: Tensor (batch)
                各バッチの入力の長さのTensor
        Returns:
            (tuple): (xs, output_lengths)

            xs: Tensor (batch, seq_length, decoder_feature_size)
                出力の特徴量
            output_length: Tensor (batch)
                出力の各バッチの出力の長さのTensor
        """
        # batch_size = input_features.shape[0]
        seq_length = input_features.shape[1]

        # 入力の長さのマスクを作る
        mask_len = (
            length_mask(input_lengths, seq_length)
            .to(input_features.device)
            .to(torch.bool)
        )
        # チャンクの先読みを封じるマスクを作る
        mask_chunk = chunk_mask(seq_length, 6).to(input_features.device, torch.bool)
        # マスクを掛け合わせる
        mask = mask_len * mask_chunk

        # 入力のembedding
        xs = self.embed(input_features)

        # Encoderのレイヤーを繰り返す
        for encoder_layer in self.encoders:
            xs, mask = encoder_layer(xs, mask)

        # Normalization
        xs = self.after_norm(xs)
        # assert xs.shape == (batch_size, seq_length, self.out_feature_dim)

        # 出力の長さの計算
        output_lengths = mask_len.squeeze(1).sum(dim=1)
        # assert output_lengths.shape == (batch_size,)

        return xs, output_lengths
