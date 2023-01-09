import os
import pathlib
import sys

import torchaudio


def resample(dataset_dir: str, output_dir: str):
    """
    音声ファイルを24kHzにリサンプリングする
    """
    print("Resampling dataset...")


    output_dir = pathlib.Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for filename in pathlib.Path(dataset_dir).rglob("*.wav"):
        sys.stdout.write("\033[1A")  # Line Up
        sys.stdout.write("\033[K")  # Clear line
        print(f"audio resampling {filename}")

        y, sr = torchaudio.load(str(filename))
        y = torchaudio.transforms.Resample(sr, 24000)(y)

        torchaudio.save(
            str(output_dir / filename.name).replace(".wav", ".flac"),
            y,
            24000,
            format="flac"
        )
