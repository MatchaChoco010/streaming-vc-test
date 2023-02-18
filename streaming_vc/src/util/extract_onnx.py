import os

import onnx
import onnx_graphsurgeon as gs
import torch
from onnxsim import simplify
from src.model.asr_model import ASRModel
from src.model.spk_rm import SpeakerRemoval
from src.model.mel_gen import MelGenerate
from src.model.hifi_gan_generator import Generator


def extract(ckpt_path: str):
    ckpt = torch.load(ckpt_path)

    asr_model = ASRModel(vocab_size=32)
    spk_rm = SpeakerRemoval()
    mel_gen = MelGenerate()
    generator = Generator()

    asr_model.load_state_dict(ckpt["asr_model"])
    spk_rm.load_state_dict(ckpt["spk_rm"])
    mel_gen.load_state_dict(ckpt["mel_gen"])
    generator.load_state_dict(ckpt["generator"])

    asr_model.eval()
    spk_rm.eval()
    mel_gen.eval()
    generator.eval()

    os.makedirs("output/onnx", exist_ok=True)

    feature_extractor = asr_model.feature_extractor
    feature_extractor.melspec.set_mode("DFT", "store")
    feature_extractor.eval()

    encoder = asr_model.encoder

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
        spk_rm,
        (torch.ones(1, 8, 128).to(dtype=torch.float),),
        "output/onnx/spk_rm.orig.onnx",
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
        mel_gen,
        (torch.ones(1, 8, 128).to(dtype=torch.float),),
        "output/onnx/mel_gen.orig.onnx",
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={
            "input": {0: "batch_size", 1: "seq_len"},
            "output": {0: "batch_size", 2: "seq_len"},
        },
    )
    torch.onnx.export(
        generator,
        (torch.ones(1, 80, 256 * 6).to(dtype=torch.float),),
        "output/onnx/hifi_gan.orig.onnx",
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

    srm = onnx.load("output/onnx/spk_rm.orig.onnx")
    srm = gs.import_onnx(srm)
    srm.cleanup()
    srm = gs.export_onnx(srm, do_type_check=True)
    onnx.save(srm, "output/onnx/spk_rm.onnx")

    mgn = onnx.load("output/onnx/mel_gen.orig.onnx")
    mgn = gs.import_onnx(mgn)
    mgn.cleanup()
    mgn = gs.export_onnx(mgn, do_type_check=True)
    onnx.save(mgn, "output/onnx/mel_gen.onnx")

    hgn = onnx.load("output/onnx/hifi_gan.orig.onnx")
    hgn = gs.import_onnx(hgn)
    hgn.cleanup()
    hgn = gs.export_onnx(hgn, do_type_check=True)
    onnx.save(hgn, "output/onnx/hifi_gan.onnx")

    for _ in range(5):
        fe = onnx.load("output/onnx/feature_extractor.onnx")
        enc = onnx.load("output/onnx/encoder.onnx")
        srm = onnx.load("output/onnx/spk_rm.onnx")
        mgn = onnx.load("output/onnx/mel_gen.onnx")
        hgn = onnx.load("output/onnx/hifi_gan.onnx")

        fe, _ = simplify(fe)
        enc, _ = simplify(enc, skip_constant_folding=True)
        srm, _ = simplify(srm)
        mgn, _ = simplify(mgn)
        hgn, _ = simplify(hgn)

        onnx.save(fe, "output/onnx/feature_extractor.onnx")
        onnx.save(enc, "output/onnx/encoder.onnx")
        onnx.save(srm, "output/onnx/spk_rm.onnx")
        onnx.save(mgn, "output/onnx/mel_gen.onnx")
        onnx.save(hgn, "output/onnx/hifi_gan.onnx")
