import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.trainer.trainer import Trainer

batch_size: int = 16
max_step: int = 10000001
progress_step: int = 5
valid_step: int = 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--output_dir", default="output")
    args = parser.parse_args()
    trainer = Trainer(
        batch_size=batch_size,
        max_step=max_step,
        progress_step=progress_step,
        valid_step=valid_step,
        exp_name=args.exp_name,
        output_dir=args.output_dir,
    )
    trainer.run()
