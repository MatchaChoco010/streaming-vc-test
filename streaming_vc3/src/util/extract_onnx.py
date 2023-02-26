import os

import onnx
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn
from onnxsim import simplify
from src.model.asr_feature_extractor import FeatureExtractor
from src.model.asr_encoder import Encoder
from src.model.bottleneck import Bottleneck
from src.model.hifi_gan_generator import Generator
from src.model.residual_coupling_block import ResidualCouplingBlock


class Flow(nn.Module):
    def __init__(self):
        super(Flow, self).__init__()
        self.flow = ResidualCouplingBlock()

    def forward(self, z):
        return self.flow.reverse(z)


def extract(ckpt_path: str):
    ckpt = torch.load(ckpt_path)

    encoder = Encoder(240, 128)
    bottleneck = Bottleneck()
    vocoder = Generator()
    flow = Flow()

    encoder.load_state_dict(ckpt["asr_encoder"])
    bottleneck.load_state_dict(ckpt["bottleneck"])
    vocoder.load_state_dict(ckpt["vocoder"])
    flow.flow.load_state_dict(ckpt["flow"])

    encoder.eval()
    bottleneck.eval()
    vocoder.eval()
    flow.eval()

    os.makedirs("output/onnx", exist_ok=True)

    feature_extractor = FeatureExtractor()
    feature_extractor.melspec.set_mode("DFT", "store")
    feature_extractor.eval()

    torch.onnx.export(
        feature_extractor,
        (torch.zeros(1, 24000 * 64 // 1000).to(dtype=torch.float),),
        "output/onnx/feature_extractor.orig.onnx",
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={
            "input": {0: "batch_size", 1: "seq_len"},
            "output": {0: "batchsize", 1: "seq_len"},
        },
    )
    torch.onnx.export(
        encoder,
        (torch.ones(1, 8, 240).to(dtype=torch.float),),
        "output/onnx/encoder.orig.onnx",
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={
            "input": {0: "batch_size", 1: "seq_len"},
            "output": {0: "batch_size", 1: "seq_len"},
        },
    )
    torch.onnx.export(
        bottleneck,
        (torch.ones(1, 8, 128).to(dtype=torch.float),),
        "output/onnx/bottleneck.orig.onnx",
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output", "mu", "log_sigma"],
        opset_version=11,
        dynamic_axes={
            "input": {0: "batch_size", 1: "seq_len"},
            "output": {0: "batch_size", 2: "seq_len"},
            "mu": {0: "batch_size", 2: "seq_len"},
            "log_sigma": {0: "batch_size", 2: "seq_len"},
        },
    )
    torch.onnx.export(
        flow,
        (torch.ones(1, 256, 6).to(dtype=torch.float),),
        "output/onnx/flow.orig.onnx",
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={
            "input": {0: "batch_size", 2: "seq_len"},
            "output": {0: "batch_size", 2: "seq_len"},
        },
    )
    torch.onnx.export(
        vocoder,
        (torch.ones(1, 256, 256 * 6).to(dtype=torch.float),),
        "output/onnx/vocoder.orig.onnx",
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={
            "input": {0: "batch_size", 2: "seq_len"},
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

    bnc = onnx.load("output/onnx/bottleneck.orig.onnx")
    bnc = gs.import_onnx(bnc)
    bnc.cleanup()
    bnc = gs.export_onnx(bnc, do_type_check=True)
    onnx.save(bnc, "output/onnx/bottleneck.onnx")

    flw = onnx.load("output/onnx/flow.orig.onnx")
    flw = gs.import_onnx(flw)
    flw.cleanup()
    flw = gs.export_onnx(flw, do_type_check=True)
    onnx.save(flw, "output/onnx/flow.onnx")

    vcd = onnx.load("output/onnx/vocoder.orig.onnx")
    vcd = gs.import_onnx(vcd)
    vcd.cleanup()
    vcd = gs.export_onnx(vcd, do_type_check=True)
    onnx.save(vcd, "output/onnx/vocoder.onnx")

    for _ in range(5):
        fe = onnx.load("output/onnx/feature_extractor.onnx")
        enc = onnx.load("output/onnx/encoder.onnx")
        bnc = onnx.load("output/onnx/bottleneck.onnx")
        flw = onnx.load("output/onnx/flow.onnx")
        vcd = onnx.load("output/onnx/vocoder.onnx")

        fe, _ = simplify(fe)
        enc, _ = simplify(enc, skip_constant_folding=True)
        bnc, _ = simplify(bnc)
        flw, _ = simplify(flw)
        vcd, _ = simplify(vcd)

        onnx.save(fe, "output/onnx/feature_extractor.onnx")
        onnx.save(enc, "output/onnx/encoder.onnx")
        onnx.save(bnc, "output/onnx/bottleneck.onnx")
        onnx.save(flw, "output/onnx/flow.onnx")
        onnx.save(vcd, "output/onnx/vocoder.onnx")
