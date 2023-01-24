import argparse
import os
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import src.data.prepare as prepare_dataset


def prepare(dataset_dir: str, output_dir: str):
    prepare_dataset.resample(dataset_dir, "tmp")
    prepare_dataset.remove_silence("tmp", output_dir)
    shutil.rmtree("tmp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="resample dataset")
    parser.add_argument("--voice_data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    prepare(args.dataset_dir, args.output_dir)
