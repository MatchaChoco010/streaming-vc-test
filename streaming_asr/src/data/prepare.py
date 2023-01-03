import os
import pathlib
import sys

import torchaudio


def resample():
    """
    音声ファイルを24kHzにリサンプリングする
    """
    print("Resampling dataset...")

    # LibriSpeechのオーディオファイル
    basepath = pathlib.Path("dataset/extracted")
    output_dir = pathlib.Path("dataset/resampled")
    for filename in pathlib.Path("dataset/extracted/LibriSpeech/").rglob("*.flac"):
        sys.stdout.write("\033[1A")  # Line Up
        sys.stdout.write("\033[K")  # Clear line
        print(f"audio resampling {filename}")

        relative_path = filename.relative_to(basepath)

        y, sr = torchaudio.load(str(filename))
        y = torchaudio.transforms.Resample(sr, 24000)(y)

        os.makedirs(output_dir / relative_path.parent, exist_ok=True)
        torchaudio.save(str(output_dir / relative_path), y, 24000, format="flac")


def copy_text():
    """
    データセットのテキストファイルをリネームして保存する
    """
    print("Copying text files...")

    # LibriSpeechのテキストファイル
    basepath = pathlib.Path("dataset/extracted")
    outpath = pathlib.Path("dataset/text")
    for filename in pathlib.Path("dataset/extracted/LibriSpeech/").rglob("*.trans.txt"):
        sys.stdout.write("\033[1A")  # Line Up
        sys.stdout.write("\033[K")  # Clear line
        print(f"text copying {filename}")

        relative_path = filename.relative_to(basepath)
        parent_path = relative_path.parent

        with open(filename, "r") as f:
            for line in [s.strip() for s in f.readlines()]:
                if line == "":
                    continue
                name, text = line.split(" ", 1)

                os.makedirs(outpath / parent_path, exist_ok=True)
                with open(outpath / parent_path / f"{name}.txt", "w") as f:
                    f.write(text)
