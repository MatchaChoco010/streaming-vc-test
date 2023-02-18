import pathlib
import random
from typing import Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, IterableDataset
from src.module.log_melspectrogram import log_melspectrogram

AUDIO_LENGTH = int(24000 * 1.0)
MEL_LENGTH = int(24000 * 1.0 / 256.0)


class TargetDataset(IterableDataset):
    def __init__(self, target_data_dir: str):
        self.file_list = [
            str(item) for item in pathlib.Path(target_data_dir).rglob("*.wav")
        ]
        self.mel = torchaudio.transforms.MelSpectrogram(
            n_fft=1024,
            n_mels=80,
            sample_rate=24000,
            hop_length=256,
            win_length=1024,
        )

    def __iter__(self):
        for item in self.file_list:
            audio, _ = torchaudio.load(item)

            start = random.randint(0, max(0, audio.shape[1] - AUDIO_LENGTH * 2))
            clip_audio = audio[:, start : start + AUDIO_LENGTH * 2]
            if clip_audio.shape[1] < AUDIO_LENGTH * 2:
                clip_audio = F.pad(
                    clip_audio,
                    (0, AUDIO_LENGTH * 2 - clip_audio.shape[1]),
                    "constant",
                )

            clip_x_mel = self.mel(clip_audio)
            clip_x_mel = log_melspectrogram(clip_x_mel).squeeze(0)

            start = random.randint(0, max(0, clip_x_mel.shape[1] - MEL_LENGTH))
            clip_x_mel = clip_x_mel[:, start : start + MEL_LENGTH]
            if clip_x_mel.shape[1] < MEL_LENGTH:
                clip_x_mel = F.pad(
                    clip_audio,
                    (0, MEL_LENGTH - clip_x_mel.shape[1]),
                    "constant",
                )

            yield clip_x_mel


class SourceDataset(IterableDataset):
    def __init__(self, source_data_dir):
        self.file_list = [
            str(item) for item in pathlib.Path(source_data_dir).rglob("*.wav")
        ]

    def __iter__(self):
        for item in self.file_list:
            audio, _ = torchaudio.load(item)

            start = random.randint(0, max(0, audio.shape[1] - AUDIO_LENGTH))
            clip_audio = audio[:, start : start + AUDIO_LENGTH]
            if clip_audio.shape[1] < AUDIO_LENGTH:
                clip_audio = F.pad(
                    clip_audio,
                    (0, AUDIO_LENGTH - clip_audio.shape[1]),
                    "constant",
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
    source_data_dir: str,
    target_data_dir: str,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    source_data_loader = DataLoader(
        ShuffleDataset(SourceDataset(source_data_dir), 32),
        batch_size=max(batch_size, 1),
        drop_last=False,
        pin_memory=True,
    )
    target_data_loader = DataLoader(
        ShuffleDataset(TargetDataset(target_data_dir), 32),
        batch_size=max(batch_size, 1),
        drop_last=False,
        pin_memory=True,
    )

    return source_data_loader, target_data_loader
