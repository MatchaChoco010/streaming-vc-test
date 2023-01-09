import os
import pathlib
import sys

import torchaudio
from pydub import AudioSegment, silence


def resample(dataset_dir: str, out_dir: str):
    """
    音声ファイルを24kHzにリサンプリングする
    """
    print("Resampling dataset...")

    output_dir = pathlib.Path(out_dir)
    os.makedirs(output_dir, exist_ok=True)
    for filename in pathlib.Path(dataset_dir).rglob("*.wav"):
        sys.stdout.write("\033[1A")  # Line Up
        sys.stdout.write("\033[K")  # Clear line
        print(f"audio resampling {filename}")

        y, sr = torchaudio.load(str(filename))
        y = torchaudio.transforms.Resample(sr, 24000)(y)

        torchaudio.save(
            str(output_dir / filename.name),
            y,
            24000,
            format="wav",
        )


def remove_silence(dataset_dir: str, out_dir: str):
    """
    音声ファイルの無音区間を削除する
    """
    print("Removing silence from dataset...")

    output_dir = pathlib.Path(out_dir)
    os.makedirs(output_dir, exist_ok=True)
    for filename in pathlib.Path(dataset_dir).rglob("*.wav"):
        sys.stdout.write("\033[1A")  # Line Up
        sys.stdout.write("\033[K")  # Clear line
        print(f"audio silence removal {filename}")

        audio = AudioSegment.from_file(filename, format="wav", frame_rate=24000)
        chunks = silence.split_on_silence(
            audio, min_silence_len=1000, silence_thresh=-50
        )
        for i, chunk in enumerate(chunks):
            chunk.export(
                str(output_dir / filename.name).replace(".wav", f"-{i}.wav"),
                format="wav",
            )
