import pathlib
from typing import List, Tuple

from torch.utils.data import Dataset

READ_FILE_THREADS = 4
VALIDATION_FILENAMES = [
    "f1_script1_clean.flac",
    "f2_script2_clean.flac",
    "f3_script3_clean.flac",
    "f4_script4_clean.flac",
    "f5_script5_clean.flac",
    "m1_script1_clean.flac",
    "m2_script2_clean.flac",
    "m3_script3_clean.flac",
    "m4_script4_clean.flac",
    "m5_script5_clean.flac",
]


class DapsDataset(Dataset):
    """
    DAPSのデータセットを扱うクラス
    """

    def __init__(self, train=True):
        self.path = "dataset/resampled/"

        file_list = list(pathlib.Path(self.path).rglob("*.flac"))
        self.file_list = []
        for f in file_list:
            if train and f.name in VALIDATION_FILENAMES:
                continue
            elif not train and not (f.name in VALIDATION_FILENAMES):
                continue
            self.file_list.append(f)

    def __getitem__(self, index) -> Tuple[str, str]:
        mel = (
            str(self.file_list[index])
            .replace("resampled", "mel")
            .replace(".flac", ".mel")
        )
        return str(self.file_list[index]), mel

    def __len__(self) -> int:
        return len(self.file_list)
