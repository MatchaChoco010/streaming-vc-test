import MeCab
import datasets
import ipadic
import pykakasi
import torch
import torchaudio
from datasets import load_dataset
from src.module.text_encoder import TextEncoder
from torch.utils.data import IterableDataset

MAX_AUDIO_LENGTH = 24000 * 30
MAX_TEXT_LENGTH = 600


class ReazonDataset(IterableDataset):
    """
    ReazonSpeechのデータセットを扱うクラス
    """

    def __init__(self, text_encoder: TextEncoder, train: bool):
        self.train = train
        if self.train:
            self.dataset = load_dataset(  # type: ignore
                "reazon-research/reazonspeech", "all", streaming=True
            )["train"].skip(2400)
        else:
            self.dataset = load_dataset(  # type: ignore
                "reazon-research/reazonspeech", "small", streaming=True
            )["train"].take(2400)
        self.kks = pykakasi.kakasi()
        self.text_encoder = text_encoder
        self.T = MeCab.Tagger(f"{ipadic.MECAB_ARGS} -Oyomi")

    def __iter__(self):
        for data in self.dataset:
            audio = torch.from_numpy(data["audio"]["array"]).to(dtype=torch.float32)
            audio = torchaudio.transforms.Resample(
                data["audio"]["sampling_rate"], 24000
            )(audio)[:MAX_AUDIO_LENGTH]
            result = self.kks.convert(self.T.parse(data["transcription"]))
            text = "".join([item["kunrei"] for item in result]).upper()
            encoded_text = self.text_encoder.encode(text[:MAX_TEXT_LENGTH])

            yield audio, encoded_text
