from functools import partial
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from src.data.libri_dataset import LibriDataset
from src.data.reazon_dataset import ReazonDataset
from src.module.text_encoder import TextEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def collect_audio_batch(
    batch: List[Dataset],
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    データのバッチをまとめる関数

    Arguments:
        batch: List[Dataset]
            データのバッチ
        vocab_size: int
            語彙のサイズ
    Returns:
        (audio, audio_lengths, texts):
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

        audio: torch.Tensor (batch_size, max(audio_len))
            音声の特徴量
        audio_lengths: torch.Tensor (batch_size)
            各バッチの音声特徴量の長さ
        texts: torch.Tensor (batch_size, text_len, vocab_size)
            テキストのone-hot表現
    """
    audio_list, audio_len, text_list = [], [], []
    with torch.no_grad():
        for b in batch:
            audio, _ = torchaudio.load(str(b[0]))
            audio = audio.squeeze(0)
            # audio = b[0].squeeze(0)
            audio_list.append(audio)
            audio_len.append(audio.shape[0])
            text_list.append(F.one_hot(torch.LongTensor(b[1]), num_classes=vocab_size))

    audio_len, audio_list, text_list = zip(
        *[
            (audio_len, audio, txt)
            for audio_len, audio, txt in sorted(
                zip(audio_len, audio_list, text_list),
                reverse=True,
                key=lambda x: x[0],
            )
        ]
    )

    max_audio_len = max(audio_len)
    audio_items = []
    for audio in audio_list:
        audio_items.append(
            F.pad(audio, (0, max_audio_len - audio.shape[0]), "constant", 0)
        )
    audio = torch.stack(audio_items, dim=0)
    texts = pad_sequence(text_list, batch_first=True)
    audio_lengths = torch.LongTensor(audio_len)

    return audio, audio_lengths, texts


def load_data(
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, int, TextEncoder]:
    """
    DataLoaderを作成する関数

    Arguments:
        batch_size: int
            バッチサイズ
    Returns:
        (train_set, dev_set, vocab_size, text_encoder):
            Tuple[DataLoader, DataLoader, int, TextEncoder]

        train_set: DataLoader
            学習用のデータセットのローダー
        dev_set: DataLoader
            検証用のデータセットのローダー
        vocab_size: int
            語彙のサイズ
        text_encoder: TextEncoder
            テキストのエンコーダー
    """
    text_encoder = TextEncoder()

    # dev_set = ReazonDataset(text_encoder, train=False)
    # train_set = ReazonDataset(text_encoder, train=True)
    dev_set = LibriDataset(["dev-clean"], text_encoder)
    train_set = LibriDataset(
        ["test-clean", "train-clean-100", "train-clean-360"], text_encoder
    )

    collect_data_fn = partial(
        collect_audio_batch,
        vocab_size=text_encoder.vocab_size,
    )

    dev_set = DataLoader(
        dev_set,
        batch_size=batch_size,
        drop_last=False,
        collate_fn=collect_data_fn,
        pin_memory=True,
    )
    train_set = DataLoader(
        train_set,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=collect_data_fn,
        pin_memory=True,
        shuffle=True,
    )

    return train_set, dev_set, text_encoder.vocab_size, text_encoder
