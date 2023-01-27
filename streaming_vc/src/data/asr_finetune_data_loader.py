import pathlib
import random
from typing import Tuple, List

import datasets
import torch
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset, Dataset

MAX_AUDIO_LENGTH = 6 * 256 * 128


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

            yield clip_audio


class VCGanFakeDataset(IterableDataset):
    """
    VCのGANの訓練用のFakeのデータセットを扱うクラス
    """

    def __init__(self):
        self.dataset = load_dataset(  # type: ignore
            "reazon-research/reazonspeech", "small"  # , streaming=True
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


class ReazonDataset(IterableDataset):
    """
    ReazonSpeechのデータセットを扱うクラス
    """

    def __init__(self):
        self.dataset = load_dataset(  # type: ignore
            "reazon-research/reazonspeech", "small"  # , streaming=True
        )["train"]

    def __iter__(self):
        for data in self.dataset:
            audio = torch.from_numpy(data["audio"]["array"]).to(dtype=torch.float32)
            audio = torchaudio.transforms.Resample(
                data["audio"]["sampling_rate"], 24000
            )(audio[: MAX_AUDIO_LENGTH * 2])[:MAX_AUDIO_LENGTH]
            audio = audio.unsqueeze(0)

            if audio.shape[1] < MAX_AUDIO_LENGTH:
                audio = F.pad(audio, (0, MAX_AUDIO_LENGTH - audio.shape[1]), "constant")

            audio = audio.squeeze(0)
            yield audio


def collect_audio_batch(batch: List[Dataset]) -> Tuple[torch.Tensor, torch.Tensor]:
    audio_list, audio_len = [], []
    with torch.no_grad():
        for b in batch:
            audio = b.squeeze(0)  # type: ignore
            audio_list.append(audio)
            audio_len.append(audio.shape[0])

    max_audio_len = max(max(audio_len), 256 * 64)
    audio_items = []
    for audio in audio_list:
        audio_items.append(
            F.pad(audio, (0, max_audio_len - audio.shape[0]), "constant", 0)
        )
    audio = torch.stack(audio_items, dim=0)
    audio_lengths = torch.LongTensor(audio_len)

    return audio, audio_lengths


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
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    DataLoaderを作成する関数

    Arguments:
        dataset_dir: str
            データセットのディレクトリ
        batch_size: int
            バッチサイズ
    Returns:
        (ts_data_loader, fs_data_loader, spk_rm_data_loader):
            Tuple[DataLoader, DataLoader, DataLoader]

        ts_data_loader: DataLoader
            GAN学習用の正データセットのローダー
        fs_data_loader: DataLoader
            GAN学習用の偽データセットのローダー
        spk_rm_data_loader: DataLoader
            話者情報取り除きのための偽データセットのローダー
    """
    ts_data_loader = DataLoader(
        ShuffleDataset(VCGanRealDataset(dataset_dir), 256),
        batch_size=max(batch_size, 1),
        drop_last=False,
        pin_memory=True,
    )
    fs_data_loader = DataLoader(
        VCGanFakeDataset(),
        batch_size=max(batch_size, 1),
        drop_last=False,
        pin_memory=True,
    )
    spk_rm_data_loader = DataLoader(
        ReazonDataset(),
        batch_size=max(batch_size // 2, 1),
        drop_last=False,
        collate_fn=collect_audio_batch,
        pin_memory=True,
    )

    return ts_data_loader, fs_data_loader, spk_rm_data_loader
