import os
import pathlib
import pickle
import sys

import torchaudio


def resample():
    """
    48kHzにリサンプリングする
    """
    print("Resampling audio data...")

    # DAPSのオーディオファイル
    basepath = pathlib.Path("dataset/extracted/daps/clean")
    output_dir = pathlib.Path("dataset/resampled")
    for filename in basepath.rglob("*.wav"):
        sys.stdout.write("\033[1A")  # Line Up
        sys.stdout.write("\033[K")  # Clear line
        print(f"audio processing {filename}")

        relative_path = filename.relative_to(basepath)

        try:
            y, sr = torchaudio.load(str(filename))
            y = torchaudio.transforms.Resample(sr, 48000)(y)
            os.makedirs(output_dir / relative_path.parent, exist_ok=True)
            torchaudio.save(
                str(output_dir / relative_path).replace("wav", "flac"),
                y,
                48000,
                format="flac",
            )
        except RuntimeError:
            pass


def calc_mel():
    """
    メルスペクトログラムを計算する
    """
    print("Calculate melspectrogram...")

    basepath = pathlib.Path("dataset/resampled")
    output_dir = pathlib.Path("dataset/mel")
    for filename in basepath.rglob("*.flac"):
        sys.stdout.write("\033[1A")
        sys.stdout.write("\033[K")
        print(f"audio processing {filename}")

        relative_path = filename.relative_to(basepath)

        y, sr = torchaudio.load(str(filename))

        # 24000Hzでhop_lengthは256でmelspectrogramを計算する
        y = torchaudio.transforms.Resample(sr, 24000)(y).squeeze(0)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, win_length=1024, hop_length=256, n_mels=80
        )(y)

        os.makedirs(output_dir / relative_path.parent, exist_ok=True)

        with open(output_dir / relative_path.with_suffix(".mel"), "wb") as f:
            pickle.dump(mel, f)
