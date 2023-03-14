import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.trainer.trainer import Trainer

# default parameters
batch_size: int = 4
max_step: int = 10000001
progress_step: int = 10
valid_step: int = 1000
exp_name: str | None = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument("--km_path", required=True)
    parser.add_argument("--batch_size", default=batch_size)
    parser.add_argument("--max_step", default=max_step)
    parser.add_argument("--progress_step", default=progress_step)
    parser.add_argument("--valid_step", default=valid_step)
    parser.add_argument("--exp_name", default=exp_name)
    parser.add_argument("--ckpt_dir", default="output/ckpt")
    parser.add_argument("--log_dir", default="output/log")
    args = parser.parse_args()

    trainer = Trainer(
        km_path=args.km_path,
        batch_size=args.batch_size,
        max_step=args.max_step,
        progress_step=args.progress_step,
        valid_step=args.valid_step,
        exp_name=args.exp_name,
        ckpt_dir=args.ckpt_dir,
        log_dir=args.log_dir,
    )
    trainer.run()
