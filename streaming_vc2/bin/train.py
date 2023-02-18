import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.trainer.trainer import Trainer

# default parameters
batch_size: int = 8
max_step: int = 10000001
progress_step: int = 10
valid_step: int = 1000
exp_name: str | None = None


def train(
    source_data_dir: str,
    target_data_dir: str,
    testdata_dir: str,
    asr_ckpt_path: str,
    vocoder_ckpt_path: str,
    batch_size: int,
    max_step: int,
    progress_step: int,
    valid_step: int,
    exp_name: str | None,
    ckpt_dir: str,
    log_dir: str,
):
    trainer = Trainer(
        source_data_dir=source_data_dir,
        target_data_dir=target_data_dir,
        testdata_dir=testdata_dir,
        asr_ckpt_path=asr_ckpt_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        batch_size=batch_size,
        max_step=max_step,
        progress_step=progress_step,
        valid_step=valid_step,
        exp_name=exp_name,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir,
    )
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument("--source_data_dir", required=True)
    parser.add_argument("--target_data_dir", required=True)
    parser.add_argument("--testdata_dir", required=True)
    parser.add_argument("--asr_ckpt_path", required=True)
    parser.add_argument("--vocoder_ckpt_path", required=True)
    parser.add_argument("--batch_size", default=batch_size)
    parser.add_argument("--max_step", default=max_step)
    parser.add_argument("--progress_step", default=progress_step)
    parser.add_argument("--valid_step", default=valid_step)
    parser.add_argument("--exp_name", default=exp_name)
    parser.add_argument("--ckpt_dir", default="output/vc/ckpt")
    parser.add_argument("--log_dir", default="output/vc/log")
    args = parser.parse_args()
    train(
        source_data_dir=args.source_data_dir,
        target_data_dir=args.target_data_dir,
        testdata_dir=args.testdata_dir,
        asr_ckpt_path=args.asr_ckpt_path,
        vocoder_ckpt_path=args.vocoder_ckpt_path,
        batch_size=args.batch_size,
        max_step=args.max_step,
        progress_step=args.progress_step,
        valid_step=args.valid_step,
        exp_name=args.exp_name,
        ckpt_dir=args.ckpt_dir,
        log_dir=args.log_dir,
    )
