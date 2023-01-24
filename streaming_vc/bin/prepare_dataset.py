import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import src.data.prepare as prepare

if __name__ == "__main__":
    prepare.resample("dataset/extracted", "dataset/resampled")
    prepare.remove_silence("dataset/resampled", "dataset/silence-removed")
