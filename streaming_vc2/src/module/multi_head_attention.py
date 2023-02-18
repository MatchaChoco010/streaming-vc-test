import math
from typing import Tuple

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    MultiHeadのAttentionレイヤー。
    """

    def __init__(
        self,
        num_heads: int,
        query_feature_size: int,
        key_feature_size: int,
        value_feature_size: int,
        output_feature_size: int,
    ):
        """
        Arguments:
            num_heads: int
                ヘッドの数
            query_feature_size: int
                queryの特徴量の次元数
            key_feature_size: int
                keyの特徴量の次元数
            value_feature_size: int
                valueの特徴量の次元数
            output_feature_size: int
                出力の特徴量の次元数
        """
        super(MultiHeadAttention, self).__init__()
        assert output_feature_size % num_heads == 0
        self.output_feature_size = output_feature_size
        self.d_k = output_feature_size // num_heads
        self.h = num_heads
        self.linear_query = nn.Linear(query_feature_size, output_feature_size)
        self.linear_key = nn.Linear(key_feature_size, output_feature_size)
        self.linear_value = nn.Linear(value_feature_size, output_feature_size)
        self.linear_out = nn.Linear(output_feature_size, output_feature_size)

    def forward_query_key_value(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        query, key, valueのprojectionを行い、headで分割する

        Arguments:
            query: torch.Tensor (batch_size, query_length, query_feature_size)
                queryの特徴量
            key: torch.Tensor (batch_size, key_value_length, key_feature_size)
                keyの特徴量
            value: torch.Tensor (batch_size, key_value_length, value_feature_size)
                valueの特徴量
        Returns:
            (tuple): (query, key, value)

            query: torch.Tensor (batch_size, num_heads, query_length, d_k)
                queryの特徴量
            key: torch.Tensor (batch_size, num_heads, key_value_length, d_k)
                keyの特徴量
            value: torch.Tensor (batch_size, num_heads, key_value_length, d_k)
                valueの特徴量
        """
        batch_size = query.shape[0]
        # query_length = query.shape[1]
        # key_length = key.shape[1]
        # value_length = value.shape[1]

        query = self.linear_query(query).view(batch_size, -1, self.h, self.d_k)
        key = self.linear_key(key).view(batch_size, -1, self.h, self.d_k)
        value = self.linear_value(value).view(batch_size, -1, self.h, self.d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # assert query.shape == (batch_size, self.h, query_length, self.d_k)
        # assert key.shape == (batch_size, self.h, key_length, self.d_k)
        # assert value.shape == (batch_size, self.h, value_length, self.d_k)

        return query, key, value

    def forward_attention(
        self, value: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Attentionを計算する

        Arguments:
            value: torch.Tensor (batch_size, num_heads, key_value_length, d_k)
                valueの特徴量
            scores: torch.Tensor (batch_size, num_heads, query_length, key_value_length)
                Attentionのスコア
            mask: torch.Tensor (batch_size, query_length, key_value_length)
                Attentionのマスク
        Returns:
            xs: torch.Tensor (batch_size, query_length, output_feature_size)
        """
        batch_size = value.shape[0]
        # query_length = scores.shape[2]

        # (batch_size, 1, query_length, key_value_length)
        mask = mask.unsqueeze(1).eq(0)
        min_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask, min_value)
        attention = nn.Softmax(dim=-1)(scores).masked_fill(mask, 0.0)

        xs = (
            torch.matmul(attention, value)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.h * self.d_k)
        )
        # assert xs.shape == (batch_size, query_length, self.output_feature_size)

        return self.linear_out(xs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Arguments:
            query: torch.Tensor (batch_size, query_length, query_feature_size)
                queryの特徴量
            key: torch.Tensor (batch_size, key_value_length, key_feature_size)
                keyの特徴量
            value: torch.Tensor (batch_size, key_value_length, value_feature_size)
                valueの特徴量
            mask: torch.Tensor (batch_size, key_value_length)
                Attentionのマスク
        Returns:
            xs: torch.Tensor (batch_size, query_length, output_feature_size)
        """
        # batch_size = query.shape[0]
        # query_length = query.shape[1]
        # key_length = key.shape[1]
        # value_length = value.shape[1]
        # assert key_length == value_length

        # query, key, valueのprojectionを行い、headで分割する
        query, key, value = self.forward_query_key_value(query, key, value)

        # Attentionのスコアを計算する
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        # assert scores.shape == (batch_size, self.h, query_length, key_length)

        # Attentionを計算する
        xs = self.forward_attention(value, scores, mask)
        # assert xs.shape == (batch_size, query_length, self.output_feature_size)

        return xs
