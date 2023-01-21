from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.module.mask import length_mask, seq_mask
from src.module.multi_head_attention import MultiHeadAttention
from src.module.positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    """
    ASR Transformerのデコーダー。
    """

    def __init__(self, decoder_feature_size: int, vocab_size: int):
        """
        Arguments:
            decoder_feature_size: int
                デコーダーの特徴量の次元数
            vocab_size: int
                出力の語彙数
        """
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size

        # character embedding
        self.embed = nn.Linear(vocab_size, 1024)

        # positional encoding
        self.pe = PositionalEncoding(1024)

        # self attention
        self.self_attention = MultiHeadAttention(
            4,
            1024,
            1024,
            1024,
            1024,
        )
        self.norm1 = nn.LayerNorm(1024)

        # encoder-decoder attention
        self.encoder_decoder_attention = MultiHeadAttention(
            4,
            1024,
            decoder_feature_size,
            decoder_feature_size,
            1024,
        )
        self.norm2 = nn.LayerNorm(1024)

        # feed forward
        self.fc1 = nn.Linear(1024, 2048)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(2048, 1024)

        # character probability
        self.character_probability = nn.Linear(1024, vocab_size)

        # history
        self.history: torch.Tensor | None = None

    def forward(
        self,
        input_xs: torch.Tensor,
        input_lengths: torch.Tensor,
        teacher: torch.Tensor,
        teacher_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Arguments:
            input_xs: torch.Tensor (batch_size, input_length, decoder_feature_size)
                入力の特徴量
            input_lengths: torch.Tensor (batch_size)
                入力の各バッチの長さ
            teacher: torch.Tensor (batch_size, teacher_length, vocab_size)
                教師データ
            teacher_lengths: torch.Tensor (batch_size)
                教師データの各バッチの長さ
        """
        batch_size = input_xs.shape[0]

        # 教師データの先頭に<sos>を追加し教師データを右にシフトする
        sos_char = torch.zeros((batch_size, 1, self.vocab_size)).to(teacher.device)
        sos_char[:, :, 1] = 1  # <sos>
        teacher = torch.cat([sos_char, teacher[:, :-1]], dim=1)

        # character embedding
        ys = self.embed(teacher)

        # positional encoding
        ys = self.pe(ys)

        # 教師データのマスクを作る
        l_mask = length_mask(teacher_lengths).to(teacher.device)
        s_mask = seq_mask(teacher.size(1)).to(teacher.device)
        mask = l_mask & s_mask

        # self attentionを計算する
        residual = ys
        ys = self.norm1(ys)
        ys = residual + self.dropout(self.self_attention(ys, ys, ys, mask))

        # inputデータの長さによるマスクを作る
        mask = length_mask(input_lengths, input_xs.shape[1]).to(input_xs.device)

        # encoder-decoder attentionを計算する
        residual = ys
        ys = self.norm2(ys)
        ys = residual + self.dropout(
            self.encoder_decoder_attention(ys, input_xs, input_xs, mask)
        )

        # feed forwardを計算する
        residual = ys
        ys = self.relu(self.fc1(ys))
        ys = self.dropout(ys)
        ys = residual + self.fc2(ys)

        # character probability
        return self.character_probability(ys)

    def _forward_one_step(
        self, input_x: torch.Tensor, input_lengths: torch.Tensor, history: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            input_x: torch.Tensor (batch_size, input_length, decoder_feature_size)
                入力の特徴量
            input_lengths: torch.Tensor (batch_size)
                入力の各バッチの長さ
            history: torch.Tensor (batch_size, history_length, vocab_size)
                前回までの出力
        Returns
            (outputs, history): Tuple[torch.Tensor, torch.Tensor]

            outputs: torch.Tensor (batch_size, 1, vocab_size)
                出力の語彙の確率分布
            history: torch.Tensor (batch_size, history_length + 1, vocab_size)
                前回までの出力と今回の出力の結合
        """
        batch_size = input_x.shape[0]
        history_length = history.shape[1]

        # character embedding
        ys = self.embed(history)

        # positional encoding
        ys = self.pe(ys)

        # 履歴の長さの全部通すマスクを作る
        mask = torch.ones((batch_size, 1, history_length)).to(input_x.device)

        # self attention
        residual = ys
        ys = self.norm1(ys)
        ys = residual + self.dropout(self.self_attention(ys, ys, ys, mask))

        # 入力のマスクを作る
        mask = length_mask(input_lengths, input_x.shape[1]).to(input_x.device)

        # encoder-decoder attentionを計算する
        residual = ys
        ys = self.norm2(ys)
        ys = residual + self.dropout(
            self.encoder_decoder_attention(ys, input_x, input_x, mask)
        )

        # feed forwardを計算する
        residual = ys
        ys = self.relu(self.fc1(ys))
        ys = self.dropout(ys)
        ys = residual + self.fc2(ys)

        # character probabilityを計算する
        ys = self.character_probability(ys)
        char = ys[:, -1:]

        # historyを計算する
        history_char = F.one_hot(char.argmax(-1), self.vocab_size)
        history = torch.cat([history, history_char], dim=1)

        return char, history

    def forward_test(
        self, input_xs: torch.Tensor, input_lengths: torch.Tensor, max_length: int
    ) -> torch.Tensor:
        """
        Arguments:
            input_xs: torch.Tensor (batch_size, input_length, decoder_feature_size)
                入力の特徴量
            input_lengths: torch.Tensor (batch_size)
                入力の各バッチの長さ
            max_length: int
                出力の最大長
        Returns:
            outputs: torch.Tensor (batch_size, max_length, vocab_size)
                出力の語彙の確率分布
        """
        batch_size = input_xs.shape[0]

        # historyの初期化
        self.history = torch.zeros((batch_size, 1, self.vocab_size)).to(input_xs.device)
        self.history[:, :, 1] = 1  # <sos>

        # max_length回ステップを繰り返す
        chars = []
        for _ in range(max_length):
            char, history = self._forward_one_step(
                input_xs, input_lengths, self.history
            )
            chars.append(char)
            self.history = history

        # 出力した文字列を結合する
        outputs = torch.cat(chars, dim=1)
        outputs = outputs.contiguous()

        return outputs
