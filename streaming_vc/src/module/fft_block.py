import torch
import torch.nn as nn
from src.module.causal_conv1d import CausalConv1d
from src.module.mask import chunk_mask
from src.module.multi_head_attention import MultiHeadAttention
from src.module.positional_encoding import PositionalEncoding

CHUNK_SIZE = 6


class FFTBlockST(nn.Module):
    """
    Streaming対応のFastSpeechのFFTBlock
    """

    def __init__(self, input_feature_size: int, decoder_feature_size: int):
        """
        Arguments:
            input_feature_size: int
                入力のオーディオ特徴量の次元数
            decoder_feature_size: int
                出力の特徴量の次元数
        """
        super(FFTBlockST, self).__init__()
        self.out_feature_dim = decoder_feature_size

        # 入力のembedding
        self.embed = nn.Sequential(
            nn.Linear(input_feature_size, self.out_feature_dim),
            nn.LayerNorm(self.out_feature_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            PositionalEncoding(self.out_feature_dim),
        )

        # attention
        self.attention = MultiHeadAttention(
            4,
            decoder_feature_size,
            decoder_feature_size,
            decoder_feature_size,
            decoder_feature_size,
        )
        self.norm1 = nn.LayerNorm(decoder_feature_size)
        self.dropout1 = nn.Dropout(0.1)

        # conv1d
        self.conv = CausalConv1d(decoder_feature_size, decoder_feature_size, 3)
        self.norm2 = nn.LayerNorm(decoder_feature_size)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            input_features: Tensor (batch, seq_length, input_feature_size)
                入力のオーディオ特徴量
        Returns:
            xs: Tensor (batch, seq_length, decoder_feature_size)
                出力の特徴量
        """
        seq_length = input_features.shape[1]

        # チャンクの先読みを封じるマスクを作る
        mask_chunk = chunk_mask(seq_length, CHUNK_SIZE).to(
            input_features.device, torch.bool
        )

        # 入力のembedding
        xs = self.embed(input_features)

        # self attentionの計算
        residual = xs
        xs = self.norm1(xs)
        xs = residual + self.dropout1(self.attention(xs, xs, xs, mask_chunk))

        # feed forwardの計算
        residual = xs
        xs = self.norm2(xs)
        xs = residual + self.dropout2(self.conv(xs.transpose(1, 2)).transpose(1, 2))

        return xs


class FFTBlock(nn.Module):
    """
    FastSpeechのFFTBlock
    """

    def __init__(self, input_feature_size: int, decoder_feature_size: int):
        """
        Arguments:
            input_feature_size: int
                入力のオーディオ特徴量の次元数
            decoder_feature_size: int
                出力の特徴量の次元数
        """
        super(FFTBlock, self).__init__()
        self.out_feature_dim = decoder_feature_size

        # 入力のembedding
        self.embed = nn.Sequential(
            nn.Linear(input_feature_size, self.out_feature_dim),
            nn.LayerNorm(self.out_feature_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            PositionalEncoding(self.out_feature_dim),
        )

        # attention
        self.attention = MultiHeadAttention(
            4,
            decoder_feature_size,
            decoder_feature_size,
            decoder_feature_size,
            decoder_feature_size,
        )
        self.norm1 = nn.LayerNorm(decoder_feature_size)
        self.dropout1 = nn.Dropout(0.1)

        # conv1d
        self.conv = nn.Conv1d(decoder_feature_size, decoder_feature_size, 3, padding=1)
        self.norm2 = nn.LayerNorm(decoder_feature_size)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            input_features: Tensor (batch, seq_length, input_feature_size)
                入力のオーディオ特徴量
        Returns:
            xs: Tensor (batch, seq_length, decoder_feature_size)
                出力の特徴量
        """
        seq_length = input_features.shape[1]

        mask = torch.ones((1, seq_length, seq_length)).to(
            input_features.device, torch.bool
        )

        # 入力のembedding
        xs = self.embed(input_features)

        # self attentionの計算
        residual = xs
        xs = self.norm1(xs)
        xs = residual + self.dropout1(self.attention(xs, xs, xs, mask))

        # feed forwardの計算
        residual = xs
        xs = self.norm2(xs)
        xs = residual + self.dropout2(self.conv(xs.transpose(1, 2)).transpose(1, 2))

        return xs
