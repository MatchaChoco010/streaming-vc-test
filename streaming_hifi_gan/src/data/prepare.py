import os
import pathlib
import sys

import torchaudio
from pydub import AudioSegment, silence


def resample():
    """
    24kHzにリサンプリングする
    """
    print("Resampling audio data...")

    # VCTKのオーディオファイル
    basepath = pathlib.Path("dataset/extracted/VCTK-Corpus/wav48")
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
                str(output_dir / relative_path),
                y,
                24000,
                format="wav",
            )
        except RuntimeError:
            pass


def remove_silence():
    """
    音声ファイルの無音区間を削除する
    """
    print("Removing silence from dataset...")

    output_dir = pathlib.Path("dataset/silence-removed")
    os.makedirs(output_dir, exist_ok=True)
    for filename in pathlib.Path("dataset/resampled").rglob("*.wav"):
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
