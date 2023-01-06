import os
import pathlib
from shutil import unpack_archive
from urllib.request import urlretrieve


def download():
    """
    データセットをダウンロードする
    """
    print("Downloading dataset...", end="\r")

    # DAPSのデータセット
    url = "https://zenodo.org/record/4660670/files/daps.tar.gz?download=1"
    output_dir = "dataset/download/"
    os.makedirs(output_dir, exist_ok=True)

    output_filename = "daps.tar.gz"
    output_path = os.path.join(output_dir, output_filename)
    urlretrieve(
        url,
        output_path,
        lambda block_num, block_size, total_size: print(
            f"downloading {output_filename}",
            f"{round(block_num * block_size / total_size *100, 2):>6.2f}%",
            end="\r",
        ),
    )
    print("")


def extract():
    """
    ダウンロードしたデータセットを解凍する
    """
    print("Extracting dataset...", end="\r")

    # DAPSのデータセット
    output_dir = pathlib.Path("dataset/extracted/")
    os.makedirs(output_dir, exist_ok=True)
    for dataset in pathlib.Path("dataset/download/").rglob("*.tar.gz"):
        print(f"extracting {dataset.name}")
        unpack_archive(dataset, output_dir)
