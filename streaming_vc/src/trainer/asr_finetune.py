import os
import copy
from datetime import datetime

import torch
import torch.nn.functional as F
from src.data.asr_finetune_data_loader import load_data
from src.model.asr_model import ASRModel
from src.model.discriminator import Discriminator
from src.trainer.cosine_annealing_warm_up_restarts import CosineAnnealingWarmupRestarts
from torch import optim
from torch.utils.tensorboard import SummaryWriter

SEGMENT_SIZE = 6 * 256 * 16


class Trainer:
    """
    asrの特徴量から話者情報を取り除くトレーニングをするクラス。
    """

    def __init__(
        self,
        dataset_dir: str,
        asr_ckpt_path: str,
        batch_size: int = 4,
        max_step: int = 10000001,
        progress_step: int = 10,
        valid_step: int = 500,
        exp_name: str | None = None,
        ckpt_dir: str = "output/asr-fintune/ckpt",
        log_dir: str = "output/asr-fintune/log",
    ):
        """
        Arguments:
            exp_name: str
                再開する実験名。Noneの場合は新しい実験が生成される。
        """
        self.dataset_dir = dataset_dir
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

        (
            self.real_loader,
            self.fake_loader,
            self.spk_rm_loader,
        ) = load_data(self.dataset_dir, self.batch_size)

        self.asr_model = ASRModel(vocab_size=32).to(self.device)
        asr_ckpt = torch.load(asr_ckpt_path, map_location=self.device)
        self.asr_model.load_state_dict(asr_ckpt["model"])

        self.frozen_encoder = copy.deepcopy(self.asr_model.encoder).to(self.device)

        self.discriminator = Discriminator().to(self.device)

        self.optimizer_asr_d = optim.AdamW(self.discriminator.parameters(), lr=0.001)
        self.optimizer_asr_g = optim.AdamW(
            self.asr_model.encoder.parameters(), lr=0.0001
        )

        self.scheduler_d = CosineAnnealingWarmupRestarts(
            self.optimizer_asr_d,
            first_cycle_steps=1000,
            warmup_steps=0,
            max_lr=0.005,
            min_lr=0.0001,
        )

        if exp_name is not None:
            self.load_ckpt()

    def load_ckpt(self):
        """
        ckptから復元する
        """
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt-latest.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.asr_model.load_state_dict(ckpt["asr_model"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.optimizer_asr_d.load_state_dict(ckpt["optimizer_asr_d"])
        self.optimizer_asr_g.load_state_dict(ckpt["optimizer_asr_g"])
        self.scheduler_d.load_state_dict(ckpt["scheduler_d"])
        self.step = ckpt["step"]
        print(f"Load checkpoint from {ckpt_path}")

    def save_ckpt(self):
        """
        ckptを保存する
        """
        ckpt_path = os.path.join(self.ckpt_dir, f"ckpt-{self.step:0>8}.pt")
        save_dict = {
            "asr_model": self.asr_model.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "optimizer_asr_d": self.optimizer_asr_d.state_dict(),
            "optimizer_asr_g": self.optimizer_asr_g.state_dict(),
            "scheduler_d": self.scheduler_d.state_dict(),
            "step": self.step,
        }
        torch.save(save_dict, ckpt_path)

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
        print("\n")
        print(f"Experiment name: {self.exp_name}")
        print("\n")

        self.start_time = datetime.now()

        spk_d_losses = []
        spk_g_losses = []
        spk_g_misleading_losses = []
        spk_g_text_losses = []

        def cycle(iterable):
            iterator = iter(iterable)
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    iterator = iter(iterable)

        r_data_loader = cycle(self.real_loader)
        f_data_loader = cycle(self.fake_loader)

        while self.step < self.max_step:
            for spk_rm_audio, _ in self.spk_rm_loader:
                self.scheduler_d.step()

                # speaker removal discriminator
                r_audio = next(r_data_loader)
                f_audio = next(f_data_loader)

                r_audio = torch.autograd.Variable(r_audio.to(device=self.device))
                f_audio = torch.autograd.Variable(f_audio.to(device=self.device))

                r_feat = self.asr_model.feature_extractor(r_audio.squeeze(1))
                r_feature = self.asr_model.encoder(r_feat)
                r_result = self.discriminator(r_feature)
                r_loss_d = F.binary_cross_entropy(r_result, torch.ones_like(r_result))

                f_feat = self.asr_model.feature_extractor(f_audio.squeeze(1))
                f_feature = self.asr_model.encoder(f_feat)
                f_result = self.discriminator(f_feature)
                f_loss_d = F.binary_cross_entropy(f_result, torch.zeros_like(f_result))

                spk_d_loss = r_loss_d + f_loss_d
                spk_d_losses.append(spk_d_loss.item())

                self.optimizer_asr_d.zero_grad()
                spk_d_loss.backward()
                self.optimizer_asr_d.step()

                # speaker removal encoder
                spk_rm_audio = torch.autograd.Variable(
                    spk_rm_audio.to(device=self.device)
                )

                spk_rm_feat = self.asr_model.feature_extractor(spk_rm_audio.squeeze(1))

                spk_rm_feature_hat = self.asr_model.encoder(spk_rm_feat)
                spk_rm_text_hat = self.asr_model.ctc_layers(spk_rm_feature_hat)
                # spk_rm_text_hat = spk_rm_text_hat.log_softmax(dim=2)

                spk_rm_result = self.discriminator(spk_rm_feature_hat)
                mislead_loss = F.binary_cross_entropy(
                    spk_rm_result, torch.ones_like(spk_rm_result)
                )
                spk_g_misleading_losses.append(mislead_loss.item())

                spk_rm_feature = self.frozen_encoder(spk_rm_feat)
                spk_rm_text = self.asr_model.ctc_layers(spk_rm_feature)

                # text_loss = F.cross_entropy(spk_rm_text_hat, spk_rm_text.argmax(dim=1))
                text_loss = F.huber_loss(spk_rm_text_hat, spk_rm_text)
                spk_g_text_losses.append(text_loss.item())

                spk_g_loss = mislead_loss + text_loss
                spk_g_losses.append(spk_g_loss.item())

                self.optimizer_asr_g.zero_grad()
                spk_g_loss.backward()
                self.optimizer_asr_g.step()

                # ロギング
                if self.step % self.progress_step == 0:
                    ## console
                    current_time = self.get_time()
                    print(
                        f"[{current_time}][Step: {self.step}] spk d loss: {sum(spk_d_losses) / len(spk_d_losses)}, spk g loss: {sum(spk_g_losses) / len(spk_g_losses)}",
                    )
                    ## tensorboard
                    self.log.add_scalar(
                        "train/spk_d_loss",
                        sum(spk_d_losses) / len(spk_d_losses),
                        self.step,
                    )
                    self.log.add_scalar(
                        "train/spk_g_loss",
                        sum(spk_g_losses) / len(spk_g_losses),
                        self.step,
                    )
                    self.log.add_scalar(
                        "train/spk_g_oss[misleading]",
                        sum(spk_g_misleading_losses) / len(spk_g_misleading_losses),
                        self.step,
                    )
                    self.log.add_scalar(
                        "train/spk_g_loss[text]",
                        sum(spk_g_text_losses) / len(spk_g_text_losses),
                        self.step,
                    )
                    # clear loss buffer
                    spk_d_losses = []
                    spk_g_losses = []
                    spk_g_misleading_losses = []
                    spk_g_text_losses = []

                # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                torch.cuda.empty_cache()

                # バリデーションの実行
                if self.step % self.valid_step == 0:
                    self.validate()

                # End of step
                self.step += 1
                if self.step > self.max_step:
                    break

        self.log.close()

    def validate(self):
        self.asr_model.eval()

        # バリデーションもなにかやりたいが……

        # save ckpt
        self.save_ckpt()

        # Resume training
        self.asr_model.train()
