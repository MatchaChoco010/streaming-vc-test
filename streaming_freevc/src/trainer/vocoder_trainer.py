import os
import pathlib
from datetime import datetime

import torch
import torch.nn.functional as F
import torchaudio

from src.data.vocoder_data_loader import load_data

from src.model.posterior_encoder import PosteriorEncoder

# from src.model.multi_period_discriminator import MultiPeriodDiscriminator
# from src.model.multi_scale_discriminator import MultiScaleDiscriminator
from src.model.vdecoder import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)
from src.module.log_melspectrogram import log_melspectrogram
from src.module.f0_utils import compute_f0, normalize_f0
from src.trainer.loss import discriminator_loss, feature_loss, generator_loss
from src.trainer.reg_loss import ResidualLoss
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel


class Trainer:
    """
    VCをトレーニングするクラス。
    """

    def __init__(
        self,
        voice_data_dir: str,
        testdata_dir: str,
        batch_size: int = 4,
        max_step: int = 10000001,
        progress_step: int = 10,
        valid_step: int = 5000,
        exp_name: str | None = None,
        ckpt_dir: str = "output/vocoder/ckpt",
        log_dir: str = "output/vocoder/log",
    ):
        """
        Arguments:
            exp_name: str
                再開する実験名。Noneの場合は新しい実験が生成される。
        """
        self.voice_data_dir = voice_data_dir
        self.testdata_dir = testdata_dir
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

        self.spec = torchaudio.transforms.MelSpectrogram(
            n_fft=1280,
            n_mels=80,
            sample_rate=16000,
            hop_length=320,
            win_length=1280,
        ).to(self.device)

        self.vocoder = Generator().to(self.device)
        self.posterior_encoder = PosteriorEncoder().to(self.device)
        self.mpd = MultiPeriodDiscriminator().to(self.device)
        self.msd = MultiScaleDiscriminator().to(self.device)
        self.reg_loss = ResidualLoss(
            sample_rate=16000,
            fft_size=2048,
            hop_size=320,
            f0_floor=50,
            f0_ceil=1100,
            n_mels=80,
            fmin=0,
            fmax=None,
            power=False,
            elim_0th=True,
        ).to(self.device)

        self.data_loader = load_data(voice_data_dir, batch_size)

        self.optimizer_g = optim.AdamW(
            list(self.vocoder.parameters()),
            lr=0.00025,
            betas=(0.5, 0.9),
            eps=1e-9,
        )
        self.optimizer_d = optim.AdamW(
            list(self.mpd.parameters()) + list(self.msd.parameters()),
            lr=0.00025,
            betas=(0.5, 0.9),
            eps=1e-9,
        )

        if exp_name is not None:
            self.load_ckpt()

        self.log.add_text(
            "train/params",
            f"g_lr: {0.00025}  \n"
            + f"d_lr: {0.00025}  \n"
            + f"sr-range: {40}-{100}  \n"
            + f"sr-enabled: {True}  \n"
            + f"bottleneck: {32}, {256}",
            0,
        )

    def load_ckpt(self):
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt-latest.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.vocoder.load_state_dict(ckpt["vocoder"])
        self.posterior_encoder.load_state_dict(ckpt["posterior_encoder"])
        self.mpd.load_state_dict(ckpt["mpd"])
        self.msd.load_state_dict(ckpt["msd"])
        self.optimizer_g.load_state_dict(ckpt["optimizer_g"])
        self.optimizer_d.load_state_dict(ckpt["optimizer_d"])
        self.step = ckpt["step"]

        print(f"Load checkpoint from {ckpt_path}")

    def save_ckpt(self):
        save_dict = {
            "vocoder": self.vocoder.state_dict(),
            "posterior_encoder": self.posterior_encoder.state_dict(),
            "mpd": self.mpd.state_dict(),
            "msd": self.msd.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "step": self.step,
        }

        ckpt_path = os.path.join(self.ckpt_dir, f"ckpt-latest.pt")
        torch.save(save_dict, ckpt_path)
        # ckpt_path = os.path.join(self.ckpt_dir, f"ckpt-{self.step}.pt")
        # torch.save(save_dict, ckpt_path)

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

        d_losses = []
        g_losses = []

        d_s_losses = []
        d_f_losses = []
        gen_s_losses = []
        gen_f_losses = []
        mel_losses = []
        reg_losses = []
        mel_errors = []

        def kl_loss(mu_1, log_sigma_1, mu_2, log_sigma_2):
            kl = log_sigma_2 - log_sigma_1 - 0.5
            kl += 0.5 * ((mu_1 - mu_2) ** 2) * torch.exp(-2.0 * log_sigma_2)
            return kl.mean()

        while self.step < self.max_step:

            # for audio, aug_audio, fake_audio in self.data_loader:
            for audio in self.data_loader:
                audio = audio.to(self.device)

                audio_f0 = compute_f0(audio)

                spec = self.spec(audio)[:, :, :-1]
                z, mu_2, log_sigma_2 = self.posterior_encoder(spec)

                audio_hat = self.vocoder(z, audio_f0)

                mel = log_melspectrogram(self.spec(audio.unsqueeze(1))[:, :, :-1])
                mel_hat = log_melspectrogram(self.spec(audio_hat)[:, :, :-1])

                # discrimator step
                self.optimizer_d.zero_grad()

                ## MPD
                y_df_hat_r, y_df_hat_g, _, _ = self.mpd(
                    audio.unsqueeze(1), audio_hat.detach()
                )
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                    y_df_hat_r, y_df_hat_g
                )

                ## MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(
                    audio.unsqueeze(1), audio_hat.detach()
                )
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                    y_ds_hat_r, y_ds_hat_g
                )

                loss_disc_all = loss_disc_s + loss_disc_f

                loss_disc_all.backward()
                self.optimizer_d.step()

                # generator step
                self.optimizer_g.zero_grad()

                ## L1 Mel-Spectrogram Loss
                loss_mel = F.l1_loss(mel, mel_hat) * 45

                ## GAN Loss
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(
                    audio.unsqueeze(1), audio_hat
                )
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(
                    audio.unsqueeze(1), audio_hat
                )
                # loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                # loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

                ## reg loss
                loss_reg = self.reg_loss(audio_hat, audio, audio_f0) * 2

                loss_gen_all = loss_gen_s + loss_gen_f + loss_mel + loss_reg

                loss_gen_all.backward()
                torch.nn.utils.clip_grad_value_(
                    list(self.vocoder.parameters())
                    + list(self.posterior_encoder.parameters()),
                    10000.0,
                )
                self.optimizer_g.step()

                # losses
                d_losses.append(loss_disc_all.item())
                g_losses.append(loss_gen_all.item())
                d_s_losses.append(loss_disc_s.item())
                d_f_losses.append(loss_disc_f.item())
                gen_s_losses.append(loss_gen_s.item())
                gen_f_losses.append(loss_gen_f.item())
                mel_losses.append(loss_mel.item())
                reg_losses.append(loss_reg.item())

                # ロギング
                if self.step % self.progress_step == 0:
                    ## console
                    current_time = self.get_time()
                    d_loss = sum(d_losses) / len(d_losses)
                    g_loss = sum(g_losses) / len(g_losses)
                    print(
                        f"[{current_time}][Step: {self.step}] d_loss: {d_loss}, g_loss: {g_loss}",
                    )
                    ## mel error
                    mel_error = F.l1_loss(mel, mel_hat).item()
                    mel_errors.append(mel_error)
                    ## tensorboard
                    self.log.add_scalar("train/loss/d", d_loss, self.step)
                    self.log.add_scalar("train/loss/g", g_loss, self.step)
                    self.log.add_scalar(
                        "train/mel_error", sum(mel_errors) / len(mel_errors), self.step
                    )

                    self.log.add_scalar(
                        "train/loss/d/d_s", sum(d_s_losses) / len(d_s_losses), self.step
                    )
                    self.log.add_scalar(
                        "train/loss/d/d_f", sum(d_f_losses) / len(d_f_losses), self.step
                    )
                    self.log.add_scalar(
                        "train/loss/g/gen_s",
                        sum(gen_s_losses) / len(gen_s_losses),
                        self.step,
                    )
                    self.log.add_scalar(
                        "train/loss/g/gen_f",
                        sum(gen_f_losses) / len(gen_f_losses),
                        self.step,
                    )
                    self.log.add_scalar(
                        "train/loss/g/mel", sum(mel_losses) / len(mel_losses), self.step
                    )
                    self.log.add_scalar(
                        "train/loss/g/reg", sum(reg_losses) / len(reg_losses), self.step
                    )
                    # reset losses
                    d_losses = []
                    g_losses = []
                    mel_errors = []

                    d_s_losses = []
                    d_f_losses = []
                    gen_s_losses = []
                    gen_f_losses = []
                    mel_losses = []
                    reg_losses = []
                    mel_errors = []

                # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                del audio, spec, z
                del mu_2, log_sigma_2, audio_hat, mel, mel_hat
                del y_ds_hat_g, y_df_hat_g
                del loss_disc_f, losses_disc_f_r, losses_disc_f_g
                del loss_disc_s, losses_disc_s_r, losses_disc_s_g
                del loss_disc_all
                del loss_mel, fmap_f_r, fmap_f_g, fmap_s_r, fmap_s_g
                # del loss_fm_f, loss_fm_s
                del loss_gen_f, losses_gen_f, loss_gen_s, losses_gen_s
                del loss_gen_all
                torch.cuda.empty_cache()

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
        self.vocoder.eval()

        if self.step == 0:
            for filepath in pathlib.Path(self.testdata_dir).rglob("*.wav"):
                y, sr = torchaudio.load(str(filepath))
                y = torchaudio.transforms.Resample(sr, 16000)(y).to(device=self.device)
                self.log.add_audio(f"before/audio/{filepath.name}", y, self.step, 16000)

        # PosteriorEncoder-Decoderを試す
        with torch.no_grad():
            for filepath in pathlib.Path(self.testdata_dir).rglob("*.wav"):
                current_time = self.get_time()
                print(
                    f"[{current_time}][Step: {self.step}] Start convert test file : {filepath.name[:24]}"
                )

                y, sr = torchaudio.load(str(filepath))
                y = torchaudio.transforms.Resample(sr, 16000)(y).squeeze(0)
                y = y.to(device=self.device)

                y_f0 = compute_f0(y.unsqueeze(0))

                spec = self.spec(y.unsqueeze(0))[:, :, :-1]
                z, mu_2, log_sigma_2 = self.posterior_encoder(spec)
                audio_hat = self.vocoder(z, y_f0)

                self.log.add_audio(
                    f"posterior-vocoder/audio/1x/{filepath.name}",
                    audio_hat[-1],
                    self.step,
                    16000,
                )

                # f0を0.5倍にして再構成
                y_f0 = compute_f0(y.unsqueeze(0)) * 0.5

                spec = self.spec(y.unsqueeze(0))[:, :, :-1]
                z, mu_2, log_sigma_2 = self.posterior_encoder(spec)
                audio_hat = self.vocoder(z, y_f0)

                self.log.add_audio(
                    f"posterior-vocoder/audio/0.5x/{filepath.name}",
                    audio_hat[-1],
                    self.step,
                    16000,
                )

                # f0を1.5倍にして再構成
                y_f0 = compute_f0(y.unsqueeze(0)) * 1.5

                spec = self.spec(y.unsqueeze(0))[:, :, :-1]
                z, mu_2, log_sigma_2 = self.posterior_encoder(spec)
                audio_hat = self.vocoder(z, y_f0)

                self.log.add_audio(
                    f"posterior-vocoder/audio/1.5x/{filepath.name}",
                    audio_hat[-1],
                    self.step,
                    16000,
                )

                # f0を2倍にして再構成
                y_f0 = compute_f0(y.unsqueeze(0)) * 2

                spec = self.spec(y.unsqueeze(0))[:, :, :-1]
                z, mu_2, log_sigma_2 = self.posterior_encoder(spec)
                audio_hat = self.vocoder(z, y_f0)

                self.log.add_audio(
                    f"posterior-vocoder/audio/2x/{filepath.name}",
                    audio_hat[-1],
                    self.step,
                    16000,
                )

        # Resume training
        self.vocoder.train()
