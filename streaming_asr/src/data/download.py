import os
import pathlib
from shutil import unpack_archive
from urllib.request import urlretrieve


def download():
    """
    データセットをダウンロードする
    """
    print("Downloading dataset...", end="\r")

    # LibriSpeechのデータセット
    datasets = [
        "http://www.openslr.org/resources/12/dev-clean.tar.gz",
        "http://www.openslr.org/resources/12/test-clean.tar.gz",
        "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
    ]
    output_dir = "dataset/download/"
    os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:
        output_filename = dataset.split("/")[-1]
        output_path = os.path.join(output_dir, output_filename)
        urlretrieve(
            dataset,
            output_path,
            lambda block_num, block_size, total_size: print(
                f"downloading {output_filename}",
                f"{round(block_num * block_size / total_size *100, 2)}%",
                end="\r",
            ),
        )
        print("")


def extract():
    """
    ダウンロードしたデータセットを解凍する
    """
    print("Extracting dataset...", end="\r")

    # LibriSpeechのデータセット
    output_dir = pathlib.Path("dataset/extracted/")
    os.makedirs(output_dir, exist_ok=True)
    for dataset in pathlib.Path("dataset/download/").rglob("*.tar.gz"):
        print(f"extracting {dataset.name}")
        unpack_archive(dataset, output_dir)
