import math
from typing import List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


def init_bert_params(module):
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    min_mask: int,
) -> np.ndarray:

    batch_size, all_size = shape
    mask = np.full((batch_size, all_size), False, dtype=bool)

    all_num_mask = int(
        mask_prob * all_size / float(mask_length) + np.random.rand()  # 確率的な丸め込み
    )
    all_num_mask = max(min_mask, all_num_mask)

    mask_indices_list = []
    for i in range(batch_size):
        # padding_maskがある場合は、このバッチのpadding_mask分だけ取り除く
        if padding_mask is not None:
            size: int = all_size - padding_mask[i].long().sum().item()  # type: ignore
            num_mask = int(
                mask_prob * size / float(mask_length) + np.random.rand()  # 確率的な丸め込み
            )
            num_mask = max(min_mask, num_mask)
        else:
            size = all_size
            num_mask = all_num_mask

        if size - mask_length <= num_mask:
            mask_length = size - num_mask - 1

        # range(size - mask_length)からnum_mask個選ぶ
        mask_indices = np.random.choice(
            range(size - mask_length), num_mask, replace=False
        )

        # mask_length分の連続したindexを作る
        mask_indices = np.asarray(
            [
                mask_indices[j] + offset
                for j in range(len(mask_indices))
                for offset in range(mask_length)
            ]
        )

        mask_indices_list.append(np.unique(mask_indices[mask_indices < size]))

    for i, mask_idc in enumerate(mask_indices_list):
        mask[i, mask_idc] = True

    return mask


class WavLM(nn.Module):
    def __init__(self):
        super(WavLM, self).__init__()

        feature_enc_layers = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractor(conv_layers=feature_enc_layers)

        self.post_extract_proj = nn.Linear(self.embed, 1024)

        self.mask_prob = 0.65
        self.mask_length = 10

        self.mask_emb = nn.Parameter(torch.FloatTensor(1024).uniform_())

        self.encoder = TransformerEncoder()
        self.layer_norm = LayerNorm(self.embed)

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape

        mask_indices = torch.from_numpy(
            compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                min_mask=2,
            )
        ).to(x.device)
        x[mask_indices] = self.mask_emb

        return x, mask_indices

    def forward_padding_mask(
        self, features: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
    ):
        features = self.feature_extractor(source)

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        features = self.post_extract_proj(features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
        else:
            x, mask_indices = features, None

        x = self.encoder(x, padding_mask)

        return x, mask_indices

    def forward(self, source: torch.Tensor):
        return self.extract_features(source)


class ConvFeatureExtractor(nn.Module):
    def __init__(self, conv_layers: List[Tuple[int, int, int]]):
        super(ConvFeatureExtractor, self).__init__()

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_group_norm=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=False)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            if is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(
                    make_conv(),
                    nn.GELU(),
                )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, (dim, k, stride) in enumerate(conv_layers):
            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_group_norm=i == 0,
                )
            )
            in_d = dim

    def forward(self, x, mask=None):
        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()

        self.dropout = 0.1
        self.embedding_dim = 1024

        self.pos_conv_ = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=128,
            padding=64,
            groups=16,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (128 * 1024))
        nn.init.normal_(self.pos_conv_.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv_.bias, 0)

        self.pos_conv_ = nn.utils.weight_norm(self.pos_conv_, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv_, SamePad(128), nn.GELU())

        self.relative_position_embedding = False

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=3072,
                    num_attention_heads=16,
                    dropout=self.dropout,
                    attention_dropout=0.1,
                    activation_dropout=0.0,
                    # layer_norm_first=False,
                    layer_norm_first=True,
                )
                for i in range(12)
            ]
        )

        self.layer_norm = LayerNorm(self.embedding_dim)

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        x = self.layer_norm(x)

        x = x.transpose(0, 1)

        for layer in self.layers:
            x, z = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_weights=False,
                self_attn_mask=None,
            )

        x = x.transpose(0, 1)

        return x


class TransformerSentenceEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim=1024,
        ffn_embedding_dim=3072,
        num_attention_heads=8,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        layer_norm_first=False,
    ):

        super(TransformerSentenceEncoderLayer, self).__init__()

        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.activation_fn = nn.GELU()
        self.self_attn = nn.MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = LayerNorm(self.embedding_dim)

        self.alpha = (2 * 12) ** 0.25

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
    ):
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x * self.alpha
            x = self.final_layer_norm(x)

        return x, attn
