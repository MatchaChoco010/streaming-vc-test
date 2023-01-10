import itertools
import os
import pathlib
from datetime import datetime

import torch
import torch.nn.functional as F
import torchaudio
from src.data.data_loader import load_dataset
from src.model.generator import Generator
from src.model.multi_period_discriminator import MultiPeriodDiscriminator
from src.model.multi_scale_discriminator import MultiScaleDiscriminator
from src.module.log_melspectrogram import log_melspectrogram
from src.trainer.loss import discriminator_loss, feature_loss, generator_loss
from torch import optim
from torch.utils.tensorboard import SummaryWriter

SEGMENT_SIZE = 8192


class Trainer:
    """
    HiFi-GANをトレーニングするクラス。
    """

    def __init__(
        self,
        batch_size: int = 16,
        max_step: int = 10000001,
        progress_step: int = 5,
        valid_step: int = 1000,
        exp_name: str | None = None,
    ):
        """
        Arguments:
            exp_name: str
                再開する実験名。Noneの場合は新しい実験が生成される。
        """
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

        self.ckpt_dir = os.path.join("output/ckpt", self.exp_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.log_dir = os.path.join("output/log", self.exp_name)
        self.log = SummaryWriter(self.log_dir)

        self.step = 0
        self.n_epochs = 0
        self.start_time = datetime.now()

        self.data_loader, self.validation_loader = load_dataset(batch_size)

        self.generator = Generator().to(self.device)
        self.mpd = MultiPeriodDiscriminator().to(self.device)
        self.msd = MultiScaleDiscriminator().to(self.device)

        self.optimizer_g = optim.AdamW(
            self.generator.parameters(), lr=0.0002, betas=(0.8, 0.99)
        )
        self.optimizer_d = optim.AdamW(
            itertools.chain(self.mpd.parameters(), self.msd.parameters()),
            lr=0.0002,
            betas=(0.8, 0.99),
        )

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer_g, gamma=0.999
        )
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer_d, gamma=0.999
        )

        self.best_val_error = 100.0

        if exp_name is not None:
            self.load_ckpt()

    def load_ckpt(self):
        """
        ckptから復元する
        """
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt-latest.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.generator.load_state_dict(ckpt["generator"])
        self.mpd.load_state_dict(ckpt["mpd"])
        self.msd.load_state_dict(ckpt["msd"])
        self.optimizer_g.load_state_dict(ckpt["optimizer_g"])
        self.optimizer_d.load_state_dict(ckpt["optimizer_d"])
        self.scheduler_g.load_state_dict(ckpt["scheduler_g"])
        self.scheduler_d.load_state_dict(ckpt["scheduler_d"])
        self.step = ckpt["step"]
        self.n_epochs = ckpt["n_epochs"]
        self.best_val_error = ckpt["best_val_error"]
        print(f"Load checkpoint from {ckpt_path}")

    def save_ckpt(self, best: bool = False):
        """
        ckptを保存する

        Arguments:
            best: bool
                ベストスコアかどうか
        """
        if best:
            ckpt_path = os.path.join(self.ckpt_dir, f"ckpt-best.pt")
            save_dict = {
                "generator": self.generator.state_dict(),
                "mpd": self.mpd.state_dict(),
                "msd": self.msd.state_dict(),
                "optimizer_g": self.optimizer_g.state_dict(),
                "optimizer_d": self.optimizer_d.state_dict(),
                "scheduler_g": self.scheduler_g.state_dict(),
                "scheduler_d": self.scheduler_d.state_dict(),
                "step": self.step,
                "n_epochs": self.n_epochs,
                "best_val_error": self.best_val_error,
            }
            torch.save(save_dict, ckpt_path)
        else:
            ckpt_path = os.path.join(self.ckpt_dir, f"ckpt-{self.step:0>8}.pt")
            save_dict = {
                "generator": self.generator.state_dict(),
                "mpd": self.mpd.state_dict(),
                "msd": self.msd.state_dict(),
                "optimizer_g": self.optimizer_g.state_dict(),
                "optimizer_d": self.optimizer_d.state_dict(),
                "scheduler_g": self.scheduler_g.state_dict(),
                "scheduler_d": self.scheduler_d.state_dict(),
                "step": self.step,
                "n_epochs": self.n_epochs,
                "best_val_error": self.best_val_error,
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
        print(f"Experiment name: {self.exp_name}\n")
        print("Parameters:")
        for k in self.generator.state_dict().keys():
            print(f"\t{k}")
        print("\n")

        self.start_time = datetime.now()

        while self.step < self.max_step:
            for audio, mel in self.data_loader:
                audio = torch.autograd.Variable(audio.to(device=self.device))
                mel = torch.autograd.Variable(mel.to(device=self.device))

                audio_g_hat = self.generator(mel)
                mel_g_hat = torchaudio.transforms.MelSpectrogram(
                    n_fft=1024,
                    n_mels=80,
                    sample_rate=24000,
                    hop_length=256,
                    win_length=1024,
                ).to(device=self.device)(audio_g_hat.squeeze(1))[
                    :, :, : SEGMENT_SIZE // 256
                ]
                mel_g_hat = log_melspectrogram(mel_g_hat)

                self.optimizer_d.zero_grad()

                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = self.mpd(audio, audio_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                    y_df_hat_r, y_df_hat_g
                )

                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(audio, audio_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                    y_ds_hat_r, y_ds_hat_g
                )

                loss_disc_all = loss_disc_s + loss_disc_f

                loss_disc_all.backward()
                self.optimizer_d.step()

                # Generator
                self.optimizer_g.zero_grad()

                # L1 Mel-Spectrogram Loss
                loss_mel = F.l1_loss(mel, mel_g_hat) * 45

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(
                    audio, audio_g_hat
                )
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(
                    audio, audio_g_hat
                )
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_all = (
                    loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
                )

                loss_gen_all.backward()
                self.optimizer_g.step()

                # ロギング
                if self.step % self.progress_step == 0:
                    ## console
                    current_time = self.get_time()
                    print(
                        f"[{current_time}][Epochs: {self.n_epochs}, Step: {self.step}] d_loss: {loss_disc_all.item()}, g_loss: {loss_gen_all.item()}",
                    )
                    ## mel error
                    mel_error = F.l1_loss(mel, mel_g_hat).item()
                    ## tensorboard
                    self.log.add_scalar("train/loss/d", loss_disc_all.item(), self.step)
                    self.log.add_scalar("train/loss/g", loss_gen_all.item(), self.step)
                    self.log.add_scalar("train/mel_error", mel_error, self.step)

                # バリデーションの実行
                if self.step % self.valid_step == 0:
                    self.validate()

                # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                torch.cuda.empty_cache()

                # End of step
                self.step += 1
                if self.step > self.max_step:
                    break

            self.n_epochs += 1
            self.scheduler_g.step()
            self.scheduler_d.step()
        self.log.close()

    def validate(self):
        self.generator.eval()
        torch.cuda.empty_cache()

        val_err_tot = 0.0
        with torch.no_grad():
            for audio, mel in self.validation_loader:
                audio_g_hat = self.generator(mel.to(device=self.device))[
                    :, :, :SEGMENT_SIZE
                ]
                mel = mel.to(device=self.device)
                mel_g_hat = torchaudio.transforms.MelSpectrogram(
                    n_fft=1024,
                    n_mels=80,
                    sample_rate=24000,
                    hop_length=256,
                    win_length=1024,
                ).to(device=self.device)(audio_g_hat.squeeze(1))[
                    :, :, : SEGMENT_SIZE // 256
                ]
                mel_g_hat = log_melspectrogram(mel_g_hat)
                val_err_tot += F.l1_loss(mel, mel_g_hat).item()

            val_err = val_err_tot / len(self.validation_loader)
            self.log.add_scalar("validation/mel_spec_error", val_err, self.step)

        self.generator.train()

        # 現在のckptをlatestとして保存
        self.save_ckpt()

        # それぞれの指標で良くなっていたら保存
        if val_err < self.best_val_error:
            self.best_val_error = val_err
            self.save_ckpt(best=True)

        # テストデータで試す
        with torch.no_grad():
            mel_history_size = 4

            for filepath in pathlib.Path("test_data").rglob("*.wav"):
                y, sr = torchaudio.load(str(filepath))
                y = torchaudio.transforms.Resample(sr, 24000)(y).to(device=self.device)
                mel = torchaudio.transforms.MelSpectrogram(
                    n_fft=1024,
                    n_mels=80,
                    sample_rate=24000,
                    hop_length=256,
                    win_length=1024,
                ).to(device=self.device)(y)
                mel = log_melspectrogram(mel)
                audio_items = []

                # melを6sampleずつずらしながらhistory_sizeも加えて食わせることで
                # streamingで生成する
                for i in range(0, mel.shape[2] // 6):
                    mel_item = mel[:, :, max(0, i * 6 - mel_history_size) : (i + 1) * 6]
                    audio_item = self.generator(mel_item).squeeze(0).squeeze(0)
                    audio_item = audio_item[-6 * 256 :]
                    audio_items.append(audio_item)
                audio = torch.cat(audio_items)
                audio = audio.unsqueeze(0)

                self.log.add_audio(
                    f"generated/audio/{filepath.name}", audio[0], self.step, 24000
                )

        # Resume training
        self.generator.train()
