import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.trainer.finetune import Finetune

# default parameters
batch_size: int = 4
max_step: int = 10000001
progress_step: int = 10
valid_step: int = 500
exp_name: str | None = None


def finetune(
    dataset_dir: str,
    testdata_dir: str,
    vc_ckpt_path: str,
    vocoder_ckpt_path: str,
    batch_size: int,
    max_step: int,
    progress_step: int,
    valid_step: int,
    exp_name: str | None,
):
    ft = Finetune(
        dataset_dir=dataset_dir,
        testdata_dir=testdata_dir,
        vc_ckpt_path=vc_ckpt_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        batch_size=batch_size,
        max_step=max_step,
        progress_step=progress_step,
        valid_step=valid_step,
        exp_name=exp_name,
    )
    ft.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument("--voice_data_dir", required=True)
    parser.add_argument("--testdata_dir", required=True)
    parser.add_argument("--vc_ckpt_path", required=True)
    parser.add_argument("--vocoder_ckpt_path", required=True)
    parser.add_argument("--batch_size", default=batch_size)
    parser.add_argument("--max_step", default=max_step)
    parser.add_argument("--progress_step", default=progress_step)
    parser.add_argument("--valid_step", default=valid_step)
    parser.add_argument("--exp_name", default=exp_name)
    args = parser.parse_args()
    finetune(
        dataset_dir=args.voice_data_dir,
        testdata_dir=args.testdata_dir,
        vc_ckpt_path=args.vc_ckpt_path,
        vocoder_ckpt_path=args.vocoder_ckpt_path,
        batch_size=args.batch_size,
        max_step=args.max_step,
        progress_step=args.progress_step,
        valid_step=args.valid_step,
        exp_name=args.exp_name,
    )
