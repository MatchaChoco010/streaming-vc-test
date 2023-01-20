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
        if train:
            self.dataset = load_dataset("reazon-research/reazonspeech", streaming=True, name="small")["train"].skip(600)  # type: ignore
        else:
            self.dataset = load_dataset("reazon-research/reazonspeech", streaming=True, name="small")["train"].take(600)  # type: ignore
        self.kks = pykakasi.kakasi()
        self.text_encoder = text_encoder

    def __iter__(self):
        for data in self.dataset:
            audio = torch.from_numpy(data["audio"]["array"]).to(dtype=torch.float32)
            audio = torchaudio.transforms.Resample(
                data["audio"]["sampling_rate"], 24000
            )(audio)[:MAX_AUDIO_LENGTH]
            result = self.kks.convert(data["transcription"])
            text = "".join([item["kana"] for item in result])
            encoded_text = self.text_encoder.encode(text[:MAX_TEXT_LENGTH])

            yield audio, encoded_text
