from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.model.feature_extractor import FeatureExtractor

# 64msのチャンクサイズ
CHUNK_SIZE: int = 24000 * 64 // 1000


class ASR(nn.Module):
    """
    CTCとAttentionのhybridな音声認識モデル。
    """

    def __init__(self, vocab_size: int):
        """
        Arguments:
            vocab_size: int
                出力の単語の種類数
        """
        super(ASR, self).__init__()

        # パラメータ
        self.input_feature_size = 240
        self.decoder_feature_size = 32
        self.vocab_size = vocab_size

        # モジュール
        self.feature_extractor = FeatureExtractor()
        self.encoder = Encoder(self.input_feature_size, self.decoder_feature_size)
        self.decoder = Decoder(self.decoder_feature_size, vocab_size)
        self.ctc_layers = nn.Linear(self.decoder_feature_size, vocab_size)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        teacher: torch.Tensor,
        teacher_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Arguments:
            audio: torch.Tensor (batch_size, max(audio_len))
                入力のオーディオ
            audio_lengths: torch.Tensor (batch_size)
                入力の各バッチのオーディオの長さ
            teacher: torch.Tensor (batch_size, max(teacher_len), vocab_size)
                教師ラベル
            teacher_lengths: torch.Tensor (batch_size)
                教師ラベルの各バッチの長さ
        Returns:
            (encode_len, ctc_output, att_output):
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

            encode_len: torch.Tensor (batch_size)
                各バッチのエンコード後の長さ
            ctc_output: torch.Tensor (batch_size, max_audio_len, vocab_size)
                CTCの出力したテキストの確率分布
            att_output: torch.Tensor (batch_size, max_teacher_len, vocab_size)
                Attentionの出力したテキストの確率分布
        """
        batch_size = audio.shape[0]

        # feature extract
        # 64msのチャンクに区切って食わせる
        audio = audio.split(CHUNK_SIZE, dim=1)
        audio_features = []
        audio_len = torch.zeros(batch_size, dtype=torch.long).to(audio_lengths.device)
        for i in range(len(audio)):
            audio_item = audio[i]
            audio_item = F.pad(
                audio_item, (0, CHUNK_SIZE - audio_item.shape[1]), "constant", 0
            )
            item_lengths = (audio_lengths - i * CHUNK_SIZE).clamp(min=0, max=CHUNK_SIZE)
            audio_feature_item, item_lengths = self.feature_extractor(
                audio_item, item_lengths
            )
            audio_features.append(audio_feature_item)
            audio_len += item_lengths
        audio_feature = torch.cat(audio_features, dim=1)

        # encode
        encode_feature, encode_len = self.encoder(audio_feature, audio_len)

        # ctc
        ctc_output = self.ctc_layers(encode_feature)

        # att
        att_output = self.decoder(encode_feature, encode_len, teacher, teacher_lengths)

        return encode_len, ctc_output, att_output

    def forward_test(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor, decode_step: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Arguments:
            audio_feature: torch.Tensor (batch_size, max(audio_len))
                入力のオーディオ
            audio_lengths: torch.Tensor (batch_size)
                入力の各バッチのオーディオの長さ
            decode_step: int
                decodeを走らせるステップ数
        Returns:
            (tuple): (encode_len, ctc_output, att_output)

            encode_len: torch.Tensor (batch_size)
                各バッチのエンコード後の長さ
            ctc_output: torch.Tensor (batch_size, max_audio_len, vocab_size)
                CTCの出力したテキストの確率分布
            att_output: torch.Tensor (batch_size, max_teacher_len, vocab_size)
                Attentionの出力したテキストの確率分布
        """
        batch_size = audio.shape[0]

        # feature extract
        # 64msのチャンクに区切って食わせる
        audio = audio.split(CHUNK_SIZE, dim=1)
        audio_features = []
        audio_len = torch.zeros(batch_size, dtype=torch.long).to(audio_lengths.device)
        for i in range(len(audio)):
            audio_item = audio[i]
            audio_item = F.pad(
                audio_item, (0, CHUNK_SIZE - audio_item.shape[1]), "constant", 0
            )
            item_lengths = (audio_lengths - i * CHUNK_SIZE).clamp(min=0, max=CHUNK_SIZE)
            audio_feature_item, item_lengths = self.feature_extractor(
                audio_item, item_lengths
            )
            audio_features.append(audio_feature_item)
            audio_len += item_lengths
        audio_feature = torch.cat(audio_features, dim=1)

        # encode
        encode_feature, encode_len = self.encoder(audio_feature, audio_len)

        # ctc
        ctc_output = self.ctc_layers(encode_feature)

        # att
        att_output = self.decoder.forward_test(encode_feature, encode_len, decode_step)

        return encode_len, ctc_output, att_output
