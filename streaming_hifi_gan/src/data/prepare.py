import os
import pathlib
import pickle
import sys

import torchaudio


def resample():
    """
    24kHzにリサンプリングする
    """
    print("Resampling audio data...")

    # DAPSのオーディオファイル
    basepath = pathlib.Path("dataset/extracted/LJSpeech-1.1/wavs")
    output_dir = pathlib.Path("dataset/resampled")
    for filename in basepath.rglob("*.wav"):
        sys.stdout.write("\033[1A")  # Line Up
        sys.stdout.write("\033[K")  # Clear line
        print(f"audio processing {filename}")

        relative_path = filename.relative_to(basepath)

        try:
            y, sr = torchaudio.load(str(filename))
            y = torchaudio.transforms.Resample(sr, 24000)(y)
            os.makedirs(output_dir / relative_path.parent, exist_ok=True)
            torchaudio.save(
                str(output_dir / relative_path).replace("wav", "flac"),
                y,
                24000,
                format="flac",
            )
        except RuntimeError:
            pass
