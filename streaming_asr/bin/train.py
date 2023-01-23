import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.trainer.trainer import Trainer

batch_size: int = 4
ctc_weight: float = 0.5
accumulation_steps: int = 4
max_step: int = 10000001
progress_step: int = 10
valid_step: int = 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--ckpt_dir", default="output/ckpt")
    parser.add_argument("--log_dir", default="output/log")
    args = parser.parse_args()
    trainer = Trainer(
        batch_size=batch_size,
        ctc_weight=ctc_weight,
        accumulation_steps=accumulation_steps,
        max_step=max_step,
        progress_step=progress_step,
        valid_step=valid_step,
        exp_name=args.exp_name,
        ckpt_dir=args.ckpt_dir,
        log_dir=args.log_dir,
    )
    trainer.run()
