import random

import torch
from src.data.vctk_dataset import VCTKDataset
from torch.utils.data import DataLoader
from typing import Tuple



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


def load_dataset(
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    DataLoaderを作成する関数

    Arguments:
        batch_size: int
            バッチサイズ
    Returns:
        (data_loader, validation_loader): Tuple[DataLoader, DataLoader]

        data_loader: DataLoader
            学習用のデータセットのローダー
        validation_loader: DataLoader
            学習用のデータセットのローダー
    """
    data_loader = DataLoader(
        ShuffleDataset(VCTKDataset(train=True), 2048),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        VCTKDataset(train=False),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    return data_loader, validation_loader
