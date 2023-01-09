import os

import onnx
import onnx_graphsurgeon as gs
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
        (
            torch.zeros(1, 24000 * 64 // 1000).to(dtype=torch.float),
            torch.ones(1).to(dtype=torch.int64) * 24000 * 64 // 1000,
        ),
        "output/onnx/feature_extractor.orig.onnx",
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        input_names=["input", "input_length"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={
            "input": {0: "batch_size", 1: "seq_len"},
            "input_length": {0: "batch_size"},
            "output": {0: "batchsize", 1: "seq_len"},
        },
    )
    torch.onnx.export(
        encoder,
        (
            torch.ones(1, 8, 240).to(dtype=torch.float),
            torch.ones(1).to(dtype=torch.int64) * 8,
        ),
        "output/onnx/encoder.orig.onnx",
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        input_names=["input", "input_length"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={
            "input": {0: "batch_size", 1: "seq_len"},
            "input_length": {0: "batch_size"},
            "output": {0: "batch_size", 1: "seq_len"},
        },
    )

    fe = onnx.load("output/onnx/feature_extractor.orig.onnx")
    fe = gs.import_onnx(fe)
    fe.cleanup()
    fe = gs.export_onnx(fe, do_type_check=True)
    onnx.save(fe, "output/onnx/feature_extractor.onnx")

    enc = onnx.load("output/onnx/encoder.orig.onnx")
    enc = gs.import_onnx(enc)
    enc.cleanup()
    enc = gs.export_onnx(enc, do_type_check=True)
    onnx.save(enc, "output/onnx/encoder.onnx")

    for _ in range(5):
        fe = onnx.load("output/onnx/feature_extractor.onnx")
        enc = onnx.load("output/onnx/encoder.onnx")

        fe, _ = simplify(fe)
        enc, _ = simplify(enc, skip_constant_folding=True)

        onnx.save(fe, "output/onnx/feature_extractor.onnx")
        onnx.save(enc, "output/onnx/encoder.onnx")
