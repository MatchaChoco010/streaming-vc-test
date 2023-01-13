import pathlib
from typing import List, Tuple

from torch.utils.data import Dataset


class LibriDataset(Dataset):
    """
    LibriSpeechのデータセットを扱うクラス
    """

    def __init__(self, train: bool = True):
        """
        Arguments:
            train: bool
                訓練用かどうか
        """
        self.path = "dataset/resampled/LibriSpeech/"
        self.file_list = []
        if train:
            for s in ["test-clean", "train-clean-100", "train-clean-360"]:
                split_list = list((pathlib.Path(self.path) / s).rglob("*.flac"))
                self.file_list += split_list
        else:
            split_list = list((pathlib.Path(self.path) / "dev-clean").rglob("*.flac"))
            self.file_list += split_list

    def __getitem__(self, index) -> str:
        return str(self.file_list[index])

    def __len__(self) -> int:
        return len(self.file_list)
