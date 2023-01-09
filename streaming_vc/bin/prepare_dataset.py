import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import src.data.prepare as prepare_dataset

def prepare(dataset_dir: str, output_dir: str):
    prepare_dataset.resample(dataset_dir, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'resample dataset')
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    prepare(args.dataset_dir, args.output_dir)
