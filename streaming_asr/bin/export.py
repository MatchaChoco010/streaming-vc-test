import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.util.extract_onnx import extract

ckpt_path = "output/ckpt/exp-20230103-200705/best-att.pt"
extract(ckpt_path)
