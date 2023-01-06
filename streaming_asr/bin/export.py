import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.util.extract_onnx import extract

ckpt_path = "output/ckpt/exp-20230104-083821/ckpt-00220000.pt"
extract(ckpt_path)
