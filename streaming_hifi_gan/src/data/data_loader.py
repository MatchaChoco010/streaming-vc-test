import math
import random
from functools import partial
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from src.data.vctk_dataset import VCTKDataset
from src.module.log_melspectrogram import log_melspectrogram
from torch.utils.data import DataLoader, Dataset

SEGMENT_SIZE = 8192


def collect_audio_batch(batch: List[Dataset[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    データのバッチをまとめる関数

    Arguments:
        batch: List[Dataset]
            データのバッチ
    Returns:
        (audio, mel):
            Tuple[torch.Tensor, torch.Tensor]

        audio: torch.Tensor (batch_size, segments)
            音声の特徴量
        mel: torch.Tensor (batch_size, segments / 256, mel_feature_size)
            各バッチの音声特徴量の長さ
    """
    with torch.no_grad():
        audio_list, mel_list = [], []

        for audio_filename in batch:
            audio, sr = torchaudio.load(audio_filename)

            # SEGMENT_SIZEの2倍のサンプル数で適当に切り出す
            cut_size = SEGMENT_SIZE * 2
            audio_start = random.randint(0, max(audio.shape[1] - cut_size, 0))
            audio = audio[:, audio_start : audio_start + cut_size]

            # スピードを0.98倍から1.02倍までの範囲でランダムに引き伸ばす
            speed = random.uniform(0.98, 1.02)
            audio, _ = torchaudio.sox_effects.apply_effects_tensor(
                audio,
                sr,
                [
                    ["speed", f"{speed}"],
                    ["rate", "24000"],
                ],
            )

            # SEGMENT_SIZEで切り出す
            if audio.shape[1] < SEGMENT_SIZE:
                audio = F.pad(audio, (0, SEGMENT_SIZE - audio.shape[1]), "constant")
            else:
                audio_start = random.randint(0, audio.shape[1] - SEGMENT_SIZE)
                audio = audio[:, audio_start : audio_start + SEGMENT_SIZE]

            mel = torchaudio.transforms.MelSpectrogram(
                n_fft=1024,
                n_mels=80,
                sample_rate=24000,
                hop_length=256,
                win_length=1024,
            )(audio)[:, :, : SEGMENT_SIZE // 256]
            mel = log_melspectrogram(mel)

            audio_list.append(audio)
            mel_list.append(mel)

        audio = torch.stack(audio_list, dim=0)
        mel = torch.stack(mel_list, dim=0).squeeze()

    return audio, mel


def load_dataset(
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
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
        VCTKDataset(train=True),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collect_data_fn,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        VCTKDataset(train=False),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collect_validation_fn,
        pin_memory=True,
    )

    return data_loader, validation_loader
