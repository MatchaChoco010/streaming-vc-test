import pathlib
import random

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, IterableDataset
from src.model.resize_feature_extractor import FeatureExtractor

AUDIO_LENGTH = 256 * 128


class VoiceDataset(IterableDataset):
    def __init__(self, voice_data_dir, min_scale: int, max_scale: int):
        self.file_list = [
            str(item) for item in pathlib.Path(voice_data_dir).rglob("*.wav")
        ]
        self.feature_extractor = FeatureExtractor()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __iter__(self):
        for item in self.file_list:
            audio, _ = torchaudio.load(item)

            for scale in range(self.min_scale, self.max_scale + 1):

                start = random.randint(0, max(0, audio.shape[1] - AUDIO_LENGTH))
                clip_audio = audio[:, start : start + AUDIO_LENGTH]
                if clip_audio.shape[1] < AUDIO_LENGTH:
                    clip_audio = F.pad(
                        clip_audio,
                        (0, AUDIO_LENGTH - clip_audio.shape[1]),
                        "constant",
                    )

                feat = self.feature_extractor(clip_audio, scale).squeeze(0)

                yield clip_audio, feat


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
    voice_data_dir: str,
    batch_size: int,
    min_scale: int,
    max_scale: int,
) -> DataLoader:
    voice_data_loader = DataLoader(
        ShuffleDataset(VoiceDataset(voice_data_dir, min_scale, max_scale), 8192),
        batch_size=batch_size,
        drop_last=False,
        pin_memory=True,
    )

    return voice_data_loader
