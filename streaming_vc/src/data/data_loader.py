import pathlib
import pickle
import random
from functools import partial
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from src.module.log_melspectrogram import log_melspectrogram
from torch.utils.data import DataLoader, Dataset

SEGMENT_SIZE = 8192


class VCDataset(Dataset):
    """
    VC訓練用のデータセットを扱うクラス
    """

    def __init__(self, dataset_dir: str):
        """
        Arguments:
            dataset_dir: str
                データセットの入っているティでクトリ
        """
        self.file_list = [
            str(item) for item in pathlib.Path(dataset_dir).rglob("*.flac")
        ]

    def __getitem__(self, index) -> str:
        return self.file_list[index]

    def __len__(self) -> int:
        return len(self.file_list)


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
            audio, _ = torchaudio.load(audio_filename)

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

        audio = torch.stack(audio_list, dim=0).squeeze(1)
        mel = torch.stack(mel_list, dim=0).squeeze()

    return audio, mel


def load_dataset(
    dataset_dir: str,
    batch_size: int,
) -> DataLoader:
    """
    DataLoaderを作成する関数

    Arguments:
        dataset_dir: str
            データセットのディレクトリ
        batch_size: int
            バッチサイズ
    Returns:
        data_loader: DataLoader

        data_loader: DataLoader
            学習用のデータセットのローダー
    """
    collect_data_fn = partial(collect_audio_batch)

    data_loader = DataLoader(
        VCDataset(dataset_dir),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collect_data_fn,
        pin_memory=True,
    )

    return data_loader
