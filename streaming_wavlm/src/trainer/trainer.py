import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.model.wavlm import WavLM
from src.data.data_loader import load_data

torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(
        self,
        km_path: str,
        batch_size: int = 4,
        max_step: int = 100000001,
        progress_step: int = 10,
        valid_step: int = 5000,
        exp_name: str | None = None,
        ckpt_dir: str = "output/ckpt",
        log_dir: str = "output/log",
    ):
        self.batch_size = batch_size
        self.max_step = max_step
        self.progress_step = progress_step
        self.valid_step = valid_step

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_name = (
            exp_name
            if exp_name is not None
            else f"exp-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        self.ckpt_dir = os.path.join(ckpt_dir, self.exp_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.log_dir = os.path.join(log_dir, self.exp_name)
        self.log = SummaryWriter(self.log_dir)

        self.step = 0
        self.start_time = datetime.now()

        self.wavlm = WavLM().to(self.device)
        self.fc = nn.Linear(1024, 1024).to(self.device)

        self.train_data_loader, self.test_data_loader = load_data(batch_size, km_path)

        self.optimizer = torch.optim.Adam(
            list(self.wavlm.parameters()) + list(self.fc.parameters()),
            lr=0.002,
        )

        if exp_name is not None:
            self.load_ckpt()

        self.log.add_text(
            "train/params",
            f"lr: {0.002}  \n",
            0,
        )

    def load_ckpt(self):
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt-latest.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.wavlm.load_state_dict(ckpt["wavlm"])
        self.fc.load_state_dict(ckpt["fc"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]

        print(f"Load checkpoint from {ckpt_path}")

    def save_ckpt(self):
        save_dict = {
            "wavlm": self.wavlm.state_dict(),
            "fc": self.fc.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
        }

        ckpt_path = os.path.join(self.ckpt_dir, f"ckpt-latest.pt")
        torch.save(save_dict, ckpt_path)

    def get_time(self) -> str:
        """
        スタート時からの経過時刻をフォーマットした文字列を返す
        """
        total_sec = (datetime.now() - self.start_time).total_seconds()
        days = total_sec // (3600 * 24)
        remain = total_sec - (days * 3600 * 24)
        hours = remain // 3600
        remain = remain - (hours * 3600)
        minutes = remain // 60
        seconds = remain - (minutes * 60)
        return f"{int(days):2}days {int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    def run(self):
        """
        トレーニングのメソッド
        """
        print("\n\n")
        print(f"Experiment name: {self.exp_name}")
        print("\n\n")

        losses = []

        while self.step < self.max_step:
            for audio, padding_mask, target in self.train_data_loader:

                audio = audio.to(self.device)
                padding_mask = padding_mask.to(self.device)
                target = target.to(self.device)

                feat, mask = self.wavlm.extract_features(audio, padding_mask, mask=True)
                log_prob = self.fc(feat)

                loss_keepdim = F.cross_entropy(
                    log_prob.transpose(1, 2), target, reduction="none"
                )

                loss = (loss_keepdim * mask).sum() / mask.sum()
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # ロギング
                if self.step % self.progress_step == 0:
                    ## console
                    current_time = self.get_time()
                    loss_item = sum(losses) / len(losses)
                    print(
                        f"[{current_time}][Step: {self.step}] loss: {loss_item}",
                    )
                    ## tensorboard
                    self.log.add_scalar("train/loss", loss_item, self.step)

                    # reset losses
                    losses = []

                # バリデーションの実行
                if self.step % self.valid_step == 0:
                    self.validate()
                    self.save_ckpt()

                # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                torch.cuda.empty_cache()

                # End of step
                self.step += 1
                if self.step > self.max_step:
                    break
        self.log.close()

    def validate(self):
        self.wavlm.eval()

        losses = []

        with torch.no_grad():
            for audio, padding_mask, target in self.test_data_loader:

                audio = audio.to(self.device)
                padding_mask = padding_mask.to(self.device)
                target = target.to(self.device)

                feat, mask = self.wavlm.extract_features(audio, padding_mask, mask=True)
                log_prob = self.fc(feat)

                loss_keepdim = F.cross_entropy(
                    log_prob.transpose(1, 2), target, reduction="none"
                )

                loss = (loss_keepdim * mask).sum() / mask.sum()
                losses.append(loss.item())

            self.log.add_scalar("valid/loss", sum(losses) / len(losses), self.step)

        self.wavlm.train()
