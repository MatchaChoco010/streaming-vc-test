import pathlib
from typing import List, Tuple

from src.module.text_encoder import TextEncoder
from torch.utils.data import Dataset

READ_FILE_THREADS = 4


def read_text(file_path: str) -> str:
    """
    音声ファイルに対応するテキストを読み込む関数

    Arguments:
        file_path: str
            音声ファイルのパス
    Returns:
        text: str
            音声ファイルに対応するテキスト
    """
    with open(
        file_path.replace("resampled", "text").replace(".flac", ".txt"), "r"
    ) as f:
        return f.read()


class LibriDataset(Dataset):
    """
    LibriSpeechのデータセットを扱うクラス
    """

    def __init__(self, split: List[str], text_encoder: TextEncoder):
        """
        Arguments:
            split: List[str]
                読み込むデータセットの種類
            text_encoder: TextEncoder
                テキストをエンコードするクラス
        """
        self.path = "dataset/resampled/LibriSpeech/"
        file_list = []
        for s in split:
            split_list = list((pathlib.Path(self.path) / s).rglob("*.flac"))
            file_list += split_list

        text = [read_text(str(f)) for f in file_list]
        text = [text_encoder.encode(txt) for txt in text]

        # データセットを長さの降順にソート
        self.file_list, self.text = zip(
            *[
                (f_name, txt)
                for f_name, txt in sorted(
                    zip(file_list, text), reverse=True, key=lambda x: len(x[1])
                )
            ]
        )

    def __getitem__(self, index) -> Tuple[str, List[int]]:
        return self.file_list[index], self.text[index]

    def __len__(self) -> int:
        return len(self.file_list)
