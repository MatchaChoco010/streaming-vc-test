import pickle
import random
from functools import partial
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from src.data.daps_dataset import DapsDataset
from torch.utils.data import DataLoader, Dataset


def collect_audio_batch(batch: List[Dataset]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    データのバッチをまとめる関数

    Arguments:
        batch: List[Dataset]
            データのバッチ
    Returns:
        (audio, mel):
            Tuple[torch.Tensor, torch.Tensor]

        audio: torch.Tensor (batch_size, max_audio_len)
            音声の特徴量
        mel: torch.Tensor (batch_size, max_mel_len, mel_feature_size)
            各バッチの音声特徴量の長さ
    """
    audio_list, mel_list = [], []
    with torch.no_grad():
        for b in batch:
            audio_filename, mel_filename = b

            audio, _ = torchaudio.load(audio_filename)
            mel = pickle.load(open(mel_filename, "rb"))

            if audio.shape[1] < 48000:
                audio = F.pad(audio, (0, 0, 48000 - audio.shape[1]), "constant")
                mel = F.pad(mel, (0, (48000 - audio.shape[1]) // 256, 0), "constsant")
            else:
                random_start = random.randint(0, (audio.shape[1] - 48000) // 48000)
                audio = audio[:, random_start : random_start + 48000]
                mel = mel[:, random_start // 256 : random_start // 256 + 24000 // 256]

            audio_list.append(audio)
            mel_list.append(mel)

    audio = torch.stack(audio_list, dim=0)
    mel = torch.stack(mel_list, dim=0)

    return audio, mel


def load_dataset(
    batch_size: int,
) -> DataLoader:
    """
    DataLoaderを作成する関数

    Arguments:
        batch_size: int
            バッチサイズ
    Returns:
        (data_loader, validation_loader): Tuple[DataLoader, DataLoader]

        data_loader: DataLoader
            学習用のデータセットのローダー
        validation_loader: DataLoader
            学習用のデータセットのローダー
    """
    collect_data_fn = partial(collect_audio_batch)
    collect_validation_fn = partial(collect_audio_batch)

    data_loader = DataLoader(
        DapsDataset(train=True),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collect_data_fn,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        DapsDataset(train=False),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collect_validation_fn,
        pin_memory=True,
    )

    return data_loader, validation_loader
