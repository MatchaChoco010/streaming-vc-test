import pathlib
import random
from typing import Tuple

import datasets
import torch
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset
from src.module.log_melspectrogram import log_melspectrogram
from torch.utils.data import DataLoader, IterableDataset

MAX_AUDIO_LENGTH = 6 * 256 * 32


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

            start = random.randint(0, max(0, audio.shape[1] - MAX_AUDIO_LENGTH))
            clip_audio = audio[:, start : start + MAX_AUDIO_LENGTH]

            if clip_audio.shape[1] < MAX_AUDIO_LENGTH:
                clip_audio = F.pad(
                    clip_audio, (0, MAX_AUDIO_LENGTH - clip_audio.shape[1]), "constant"
                )

            mel = torchaudio.transforms.MelSpectrogram(
                n_fft=1024,
                n_mels=80,
                sample_rate=24000,
                hop_length=256,
                win_length=1024,
            )(clip_audio)[:, :, : MAX_AUDIO_LENGTH // 256]
            mel = log_melspectrogram(mel).squeeze(0)

            yield clip_audio, mel


class VCGanFakeDataset(IterableDataset):
    """
    VCのGANの訓練用のFakeのデータセットを扱うクラス
    """

    def __init__(self):
        self.dataset = load_dataset(  # type: ignore
            "reazon-research/reazonspeech", "small" # , streaming=True
        )["train"]

    def __iter__(self):
        for data in self.dataset:
            audio = torch.from_numpy(data["audio"]["array"]).to(dtype=torch.float32)
            audio = torchaudio.transforms.Resample(
                data["audio"]["sampling_rate"], 24000
            )(audio).unsqueeze(0)

            start = random.randint(0, max(0, audio.shape[1] - MAX_AUDIO_LENGTH))
            clip_audio = audio[:, start : start + MAX_AUDIO_LENGTH]

            if clip_audio.shape[1] < MAX_AUDIO_LENGTH:
                clip_audio = F.pad(
                    clip_audio, (0, MAX_AUDIO_LENGTH - clip_audio.shape[1]), "constant"
                )

            yield clip_audio


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


def load_data(
    dataset_dir: str,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    real_data_loader = DataLoader(
        ShuffleDataset(VCGanRealDataset(dataset_dir), 256),
        batch_size=max(batch_size, 1),
        drop_last=False,
        pin_memory=True,
    )
    fake_data_loader = DataLoader(
        VCGanFakeDataset(),
        batch_size=max(batch_size, 1),
        drop_last=False,
        pin_memory=True,
    )

    return real_data_loader, fake_data_loader
