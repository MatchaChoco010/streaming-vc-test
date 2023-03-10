import pathlib
import random
from typing import Tuple

import datasets
import torch
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

MAX_AUDIO_LENGTH = 24 * 24000


class VCGanRealDataset(IterableDataset):
    """
    VCのGANの訓練用のRealのデータセットを扱うクラス
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
            Generator[audio, None, None]:
                Generator[torch.Tensor, None, None]

            audio: torch.Tensor (batch_size, segments)
                音声の特徴量
        """
        for item in self.file_list:
            audio, _ = torchaudio.load(item)
            yield audio[:, :MAX_AUDIO_LENGTH]


class VCGanFakeDataset(IterableDataset):
    """
    VCのGANの訓練用のFakeのデータセットを扱うクラス
    """

    def __init__(self):
        self.dataset = load_dataset(  # type: ignore
            "reazon-research/reazonspeech",
            "medium",
            num_proc=12,
        )["train"]

    def __iter__(self):
        for data in self.dataset:
            audio = torch.from_numpy(data["audio"]["array"]).to(dtype=torch.float32)
            audio = torchaudio.transforms.Resample(
                data["audio"]["sampling_rate"], 24000
            )(audio).unsqueeze(0)

            yield audio[:, :MAX_AUDIO_LENGTH]


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
) -> Tuple[DataLoader, DataLoader]:
    real_data_loader = DataLoader(
        ShuffleDataset(VCGanRealDataset(dataset_dir), 256),
        batch_size=max(batch_size, 1),
        drop_last=False,
        pin_memory=True,
        collate_fn=collect_audio_batch,
    )
    fake_data_loader = DataLoader(
        VCGanFakeDataset(),
        batch_size=max(batch_size, 1),
        drop_last=False,
        pin_memory=True,
        collate_fn=collect_audio_batch,
    )

    return real_data_loader, fake_data_loader
