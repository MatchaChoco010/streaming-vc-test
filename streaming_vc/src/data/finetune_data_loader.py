import pathlib
import random
from typing import Tuple, List

import torch
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset
from src.module.log_melspectrogram import log_melspectrogram
from torch.utils.data import DataLoader, IterableDataset, Dataset

SEGMENT_SIZE = 6 * 256 * 64


class VCDataset(IterableDataset):
    """
    VC訓練用のデータセットを扱うクラス
    """

    def __init__(self, dataset_dir: str):
        """
        Arguments:
            dataset_dir: str
                データセットの入っているディレクトリ
        """
        self.file_list = [
            str(item) for item in pathlib.Path(dataset_dir).rglob("*.wav")
        ]

    def __iter__(self):
        """
        Returns:
            Generator[(audio, mel), None, None]:
                Generator[Tuple[torch.Tensor, torch.Tensor], None, None]

            audio: torch.Tensor (batch_size, segments)
                音声の特徴量
            mel: torch.Tensor (batch_size, segments / 256, mel_feature_size)
                各バッチの音声特徴量の長さ
        """
        for item in self.file_list:
            audio, _ = torchaudio.load(item)
            yield audio


class ShuffleDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass


def collect_audio_batch(batch):
    audio_len = []
    for b in batch:
        audio_len.append(b.shape[1])

    max_audio_len = max(audio_len)
    audio_items = []
    for audio in batch:
        audio = audio.squeeze(0)
        audio_items.append(
            F.pad(audio, (0, max_audio_len - audio.shape[0]), "constant", 0)
        )
    audio = torch.stack(audio_items, dim=0)

    return audio


def load_data(
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
        (data_loader, ts_data_loader, fs_data_loader):
            Tuple[DataLoader, DataLoader, DataLoader]

        data_loader: DataLoader
            学習用のデータセットのローダー
    """
    data_loader = DataLoader(
        ShuffleDataset(VCDataset(dataset_dir), 256),
        batch_size=batch_size,
        drop_last=False,
        pin_memory=True,
        collate_fn=collect_audio_batch,
    )

    return data_loader
