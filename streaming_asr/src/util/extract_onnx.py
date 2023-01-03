import os

import onnx
import torch
from onnxsim import simplify
from src.model.asr import ASR
from src.module.text_encoder import TextEncoder


def extract(ckpt_path: str):
    text_encoder = TextEncoder()
    model = ASR(text_encoder.vocab_size)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model"])
    os.makedirs("output/onnx", exist_ok=True)

    feature_extractor = model.feature_extractor
    encoder = model.encoder

    feature_extractor.melspec.set_mode("DFT", "store")
    torch.onnx.export(
        feature_extractor,
        (torch.zeros(1, 24000 * 64 // 1000), torch.ones(1) * 24000 * 64 // 1000),
        "output/onnx/feature_extractor.orig.onnx",
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
    )
    torch.onnx.export(
        encoder,
        (torch.ones(1, 6, 240), torch.ones(1) * 6),
        "output/onnx/encoder.orig.onnx",
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        dynamic_axes={
            "input": {1: "seq_len"},
            "output": {1: "seq_len"},
        },
    )

    fe = onnx.load("output/onnx/feature_extractor.orig.onnx")
    fe, _ = simplify(fe)
    onnx.save(fe, "output/onnx/feature_extractor.onnx")

    enc = onnx.load("output/onnx/encoder.orig.onnx")
    # enc, _ = simplify(enc)
    onnx.save(enc, "output/onnx/encoder.onnx")
