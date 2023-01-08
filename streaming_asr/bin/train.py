import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.trainer.trainer import Trainer

batch_size: int = 16
ctc_weight: float = 0.5
accumulation_steps: int = 4
max_step: int = 10000001
progress_step: int = 10
valid_step: int = 5000
exp_name: str | None = None

if __name__ == "__main__":
    trainer = Trainer(
        batch_size=batch_size,
        ctc_weight=ctc_weight,
        accumulation_steps=accumulation_steps,
        max_step=max_step,
        progress_step=progress_step,
        valid_step=valid_step,
        exp_name=exp_name,
    )
    trainer.run()
