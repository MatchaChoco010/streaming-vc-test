import random
from functools import partial
from typing import Tuple
from transformers import AutoModel
from datasets import load_dataset

import torch
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset

from src.module.apply_kmeans import ApplyKmeans

MAX_AUDIO_LENGTH = 16000 * 5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wavlm = AutoModel.from_pretrained("microsoft/wavlm-large").to(device)


class ReazonDataset(IterableDataset):
    """
    ReazonSpeechのデータセットを扱うクラス
    """

    def __init__(self, train: bool):
        self.train = train
        if self.train:
            self.dataset = load_dataset(  # type: ignore
                # "reazon-research/reazonspeech", "medium", streaming=True
                "reazon-research/reazonspeech", "small"
            )["train"]
        else:
            self.dataset = load_dataset(  # type: ignore
                "reazon-research/reazonspeech", "small"
            )["train"]

    def __iter__(self):
        for data in self.dataset:
            audio = torch.from_numpy(data["audio"]["array"]).to(dtype=torch.float32)
            audio = torchaudio.transforms.Resample(
                data["audio"]["sampling_rate"], 16000
            )(audio)

            start = random.randint(0, max(0, audio.shape[0] - MAX_AUDIO_LENGTH))
            clip_audio = audio[start : start + MAX_AUDIO_LENGTH]

            yield clip_audio


def collect_audio_batch(
    batch: Dataset,
    apply_kmeans,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    データのバッチをまとめる関数

    Arguments:
        batch: List[Dataset]
            データのバッチ
    Returns:
        (audio, padding_mask, target_list):
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

        audio: torch.Tensor (batch_size, max(audio_len))
            音声データ
        padding_mask: torch.Tensor (batch_size, max(audio_len))
            paddingのマスク
        target: torch.Tensor (batch_size)
            WavLMに通したターゲット
    """
    audio_list, audio_len, target_list = [], [], []
    with torch.no_grad():
        for b in batch:  # type: ignore
            audio = b
            audio_list.append(audio)
            audio_len.append(audio.shape[0])

            with torch.no_grad():
                audio = audio.unsqueeze(0).to(device)
                outputs = wavlm(input_values=audio, output_hidden_states=True)
                feature = outputs.hidden_states[10]
                target = apply_kmeans(feature)
                target = target.squeeze(0).cpu()

            target_list.append(target)

    max_audio_len = max(audio_len)
    audio_items = []
    for a in audio_list:
        audio_items.append(F.pad(a, (0, max_audio_len - a.shape[0]), "constant", 0))
    audio = torch.stack(audio_items, dim=0)

    padding_mask = torch.ones_like(audio)
    for i, al in enumerate(audio_len):
        padding_mask[i, :al] = 0

    target = pad_sequence(target_list, batch_first=True)

    return audio, padding_mask, target


def load_data(
    batch_size: int,
    km_path: str,
) -> Tuple[DataLoader, DataLoader]:
    apply_kmeans = ApplyKmeans(km_path)
    collect_data_fn = partial(
        collect_audio_batch,
        apply_kmeans=apply_kmeans,
    )

    train_set = DataLoader(
        ReazonDataset(train=True),
        batch_size=batch_size,
        collate_fn=collect_data_fn,
        pin_memory=True,
    )
    test_set = DataLoader(
        ReazonDataset(train=False),
        batch_size=batch_size,
        collate_fn=collect_data_fn,
        pin_memory=True,
    )

    return train_set, test_set
