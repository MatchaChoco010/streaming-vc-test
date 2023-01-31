import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.util.extract_onnx import extract

# ckpt_path = "output/ckpt/exp-20230104-083821/ckpt-00220000.pt"
parser = argparse.ArgumentParser(description = 'export onnx model')
parser.add_argument("--ckpt", required=True)
args = parser.parse_args()
ckpt_path = args.ckpt

extract(ckpt_path)
