import pykakasi
import torch
import torchaudio
from datasets import load_dataset
from src.module.text_encoder import TextEncoder
from torch.utils.data import Dataset, IterableDataset

MAX_AUDIO_LENGTH = 24000 * 30
MAX_TEXT_LENGTH = 600


class ReazonDataset(IterableDataset):
    """
    ReazonSpeechのデータセットを扱うクラス
    """

    def __init__(self, text_encoder: TextEncoder, train: bool):
        self.train = train
        self.dataset = load_dataset(
            "reazon-research/reazonspeech", "all", streaming=True
        )[
            "train"
        ]  # type: ignore
        self.kks = pykakasi.kakasi()
        self.text_encoder = text_encoder

    def __iter__(self):
        count = 0
        for data in self.dataset:
            if self.train and count < 2400:
                # skip first 2400 samples
                count += 1
            elif not self.train and count >= 2400:
                # skip after 2400 samples
                return

            audio = torch.from_numpy(data["audio"]["array"]).to(dtype=torch.float32)
            audio = torchaudio.transforms.Resample(
                data["audio"]["sampling_rate"], 24000
            )(audio)[:MAX_AUDIO_LENGTH]
            result = self.kks.convert(data["transcription"])
            text = "".join([item["hepburn"] for item in result]).upper()
            encoded_text = self.text_encoder.encode(text[:MAX_TEXT_LENGTH])

            yield audio, encoded_text
            count += 1
