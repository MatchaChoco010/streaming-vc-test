from typing import Tuple

import torch
import torch.nn as nn
from src.module.multi_head_attention import MultiHeadAttention
from src.module.positional_encoding import PositionalEncoding

CHUNK_SIZE = 6


class EncoderLayer(nn.Module):
    """
    ASR Transformerのエンコーダーの一つのレイヤー。
    self attentionとfeed forwardをresidual結合したレイヤーとなっている。
    """

    def __init__(self, feature_size: int, history_length: int):
        """
        Arguments:
            feature_size: int
                入力の特徴量の次元数
        """
        super(EncoderLayer, self).__init__()
        self.out_feature_dim = feature_size
        self.history_length = history_length

        # Attention層
        self.norm1 = nn.LayerNorm(feature_size)
        self.attention = MultiHeadAttention(
            4, feature_size, feature_size, feature_size, feature_size
        )

        # Feed Forward層
        self.norm2 = nn.LayerNorm(feature_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, feature_size),
        )

    def forward(self, chunk, history) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            chunk: torch.Tensor (batch_size, chunk_size, feature_size)
                新しく入力されたチャンク
            history: torch.Tensor (batch_size, seq_len, feature_size)
                チャンクのヒストリー
        Returns:
            (chunk, history): Tuple[torch.Tensor, torch.Tensor]

            chunk: torch.Tensor (batch_size, chunk_size, feature_size)
                出力の特徴量
            history: torch.Tensor (batch_size, seq_len, feature_size)
                新しくチャンクを加えたヒストリ
        """
        new_history = torch.cat([history, chunk], dim=1)
        new_history = new_history[:, -self.history_length :, :]

        batch_size = history.shape[0]
        history_length = new_history.shape[1]
        dummy_mask = torch.ones(
            (batch_size, CHUNK_SIZE, history_length), dtype=torch.bool
        ).to(chunk.device)

        residual = chunk
        chunk = self.norm1(chunk)
        chunk = residual + self.attention(chunk, new_history, new_history, dummy_mask)

        residual = chunk
        chunk = self.norm2(chunk)
        chunk = residual + self.feed_forward(chunk)

        return chunk, new_history


class Encoder(nn.Module):
    """
    ASR Transformerのエンコーダー。
    self attentionとfeed forwardを繰り返す構造となっている。
    """

    def __init__(
        self, input_feature_size: int, decoder_feature_size: int, history_length: int
    ):
        """
        Arguments:
            input_feature_size: int
                入力のオーディオ特徴量の次元数
            decoder_feature_size: int
                出力の特徴量の次元数
        """
        super(Encoder, self).__init__()
        self.out_feature_dim = decoder_feature_size
        self.history_length = history_length

        # 入力のembedding
        self.chunk_embed = nn.Sequential(
            nn.Linear(input_feature_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            PositionalEncoding(512),
        )

        # Encoderのレイヤー
        encoder_modules = []
        for _ in range(6):
            encoder_modules.append(
                EncoderLayer(feature_size=512, history_length=history_length)
            )
        self.encoders = nn.ModuleList(encoder_modules)

        # feed forward
        self.fc = nn.Linear(512, self.out_feature_dim)
        self.after_norm = nn.LayerNorm(self.out_feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        chunk: torch.Tensor,
        history_layer_1: torch.Tensor,
        history_layer_2: torch.Tensor,
        history_layer_3: torch.Tensor,
        history_layer_4: torch.Tensor,
        history_layer_5: torch.Tensor,
        history_layer_6: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Arguments:
            chunk: torch.Tensor (batch_size, chunk_size, input_feature_size)
                新しく入力されたチャンク
            history_layer_1: torch.Tensor (batch_size, seq_len, 512)
                レイヤー1のチャンクのヒストリー
            history_layer_2: torch.Tensor (batch_size, seq_len, 512)
                レイヤー2のチャンクのヒストリー
            history_layer_3: torch.Tensor (batch_size, seq_len, 512)
                レイヤー3のチャンクのヒストリー
            history_layer_4: torch.Tensor (batch_size, seq_len, 512)
                レイヤー4のチャンクのヒストリー
            history_layer_5: torch.Tensor (batch_size, seq_len, 512)
                レイヤー5のチャンクのヒストリー
            history_layer_6: torch.Tensor (batch_size, seq_len, 512)
                レイヤー6のチャンクのヒストリー
        Returns:
            (chunk, history_layer_1, history_layer_2, history_layer_3, history_layer_4, history_layer_5, history_layer_6): Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

            chunk: torch.Tensor (batch_size, chunk_size, decoder_feature_size)
                出力の特徴量
            history_layer_1: torch.Tensor (batch_size, seq_len, 512)
                レイヤー1のチャンクのヒストリー
            history_layer_2: torch.Tensor (batch_size, seq_len, 512)
                レイヤー2のチャンクのヒストリー
            history_layer_3: torch.Tensor (batch_size, seq_len, 512)
                レイヤー3のチャンクのヒストリー
            history_layer_4: torch.Tensor (batch_size, seq_len, 512)
                レイヤー4のチャンクのヒストリー
            history_layer_5: torch.Tensor (batch_size, seq_len, 512)
                レイヤー5のチャンクのヒストリー
            history_layer_6: torch.Tensor (batch_size, seq_len, 512)
                レイヤー6のチャンクのヒストリー
        """
        # チャンクのembedding
        chunk = self.chunk_embed(chunk)

        # レイヤーごとの処理
        chunk, history_layer_1 = self.encoders[0](chunk, history_layer_1)
        chunk, history_layer_2 = self.encoders[1](chunk, history_layer_2)
        chunk, history_layer_3 = self.encoders[2](chunk, history_layer_3)
        chunk, history_layer_4 = self.encoders[3](chunk, history_layer_4)
        chunk, history_layer_5 = self.encoders[4](chunk, history_layer_5)
        chunk, history_layer_6 = self.encoders[5](chunk, history_layer_6)

        # Normalization
        chunk = self.after_norm(self.fc(chunk))

        # Sigmoid
        chunk = self.sigmoid(chunk)

        return (
            chunk,
            history_layer_1,
            history_layer_2,
            history_layer_3,
            history_layer_4,
            history_layer_5,
            history_layer_6,
        )
