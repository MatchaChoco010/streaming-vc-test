import os
import math
import sys
import pathlib
from datetime import datetime

import torch
import torch.nn.functional as F
import torchaudio

from src.data.data_loader import load_data
from src.model.asr_model import ASRModel
from src.model.hifi_gan_generator import Generator
from src.model.flow import FlowModel
from src.model.bottleneck import Bottleneck
from src.model.random_resize_feature_extractor import FeatureExtractor
from torch import optim
from torch.utils.tensorboard import SummaryWriter

AUDIO_LENGTH = int(24000 * 3.0)
MEL_LENGTH = AUDIO_LENGTH // 256


class Trainer:
    """
    VCをトレーニングするクラス。
    """

    def __init__(
        self,
        voice_data_dir: str,
        testdata_dir: str,
        asr_ckpt_path: str,
        vocoder_ckpt_path: str,
        batch_size: int = 4,
        max_step: int = 10000001,
        progress_step: int = 10,
        valid_step: int = 5000,
        exp_name: str | None = None,
        ckpt_dir: str = "output/vc/ckpt",
        log_dir: str = "output/vc/log",
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

        self.asr_model = ASRModel(vocab_size=32).to(self.device).eval()
        asr_ckpt = torch.load(asr_ckpt_path, map_location=self.device)
        self.asr_model.load_state_dict(asr_ckpt["model"])

        self.random_feature_extractor = FeatureExtractor(0.8, 1.2).to(self.device)

        self.mel = torchaudio.transforms.MelSpectrogram(
            n_fft=1024,
            n_mels=80,
            sample_rate=24000,
            hop_length=256,
            win_length=1024,
        ).to(self.device)
        self.flow = FlowModel(4, 80).to(self.device)
        self.bottleneck = Bottleneck().to(self.device)

        self.vocoder = Generator().to(self.device).eval()
        vocoder_ckpt = torch.load(vocoder_ckpt_path, map_location=self.device)
        self.vocoder.load_state_dict(vocoder_ckpt["generator"])

        self.data_loader = load_data(voice_data_dir, batch_size)

        self.optim = optim.AdamW(
            list(self.flow.parameters()) + list(self.bottleneck.parameters()), lr=0.0001
        )

        if exp_name is not None:
            self.load_ckpt()

    def load_ckpt(self):
        """
        ckptから復元する
        """
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt-latest.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.asr_model.encoder.load_state_dict(ckpt["asr_encoder"])
        self.flow.load_state_dict(ckpt["flow"])
        self.bottleneck.load_state_dict(ckpt["bottleneck"])
        self.optim.load_state_dict(ckpt["optim"])
        self.step = ckpt["step"]

        print(f"Load checkpoint from {ckpt_path}")

    def save_ckpt(self):
        save_dict = {
            "asr_encoder": self.asr_model.encoder.state_dict(),
            "flow": self.flow.state_dict(),
            "bottleneck": self.bottleneck.state_dict(),
            "optim": self.optim.state_dict(),
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

        kl_losses = []

        def kl_loss(mu_1, log_sigma_1, mu_2, log_sigma_2):
            kl = log_sigma_1 - log_sigma_2 - 0.5
            kl += 0.5 * ((mu_1 - mu_2) ** 2) * torch.exp(-2.0 * log_sigma_1)
            return kl.mean()

        while self.step < self.max_step:

            for audio in self.data_loader:
                audio = audio.to(self.device)
                audio = audio.squeeze(1)

                mel = self.mel(audio)
                mel = mel[:, :, :MEL_LENGTH]
                xs, log_det_jacobian = self.flow(mel)
                mu_1, log_sigma_1 = xs.chunk(2, dim=1)

                feat = self.random_feature_extractor(audio)
                feature = self.asr_model.encoder(feat)
                mu_2, log_sigma_2 = self.bottleneck(feature)

                loss_kl = kl_loss(mu_1, log_sigma_1, mu_2, log_sigma_2)
                kl_losses.append(loss_kl.item())

                self.optim.zero_grad()
                loss_kl.backward()
                self.optim.step()

                # ロギング
                if self.step % self.progress_step == 0:
                    # calculate mean of losses and cers in progress steps
                    avg_kl_loss = sum(kl_losses) / len(kl_losses)
                    ## console
                    current_time = self.get_time()
                    print(f"[{current_time}][Step: {self.step}] loss_kl: {avg_kl_loss}")
                    ## tensorboard
                    self.log.add_scalar("train/loss_kl", avg_kl_loss, self.step)
                    # reset losses
                    kl_losses = []

                # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
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
        self.flow.eval()
        self.bottleneck.eval()

        if self.step == 0:
            for filepath in pathlib.Path(self.testdata_dir).rglob("*.wav"):
                y, sr = torchaudio.load(str(filepath))
                y = torchaudio.transforms.Resample(sr, 24000)(y).to(device=self.device)
                self.log.add_audio(f"before/audio/{filepath.name}", y, self.step, 24000)

        # テストデータで試す
        with torch.no_grad():
            asr_history_size = 6 * 256
            history_size = 6 * 64
            vocoder_history_size = 16

            for filepath in pathlib.Path(self.testdata_dir).rglob("*.wav"):
                current_time = self.get_time()
                print(
                    f"[{current_time}][Step: {self.step}] Start convert test file : {filepath.name[:24]}"
                )

                y, sr = torchaudio.load(str(filepath))
                y = torchaudio.transforms.Resample(sr, 24000)(y).squeeze(0)
                y = y.to(device=self.device)

                # historyを初期化
                feat_history = torch.zeros((1, history_size, 240)).to(self.device)
                feature_history = torch.zeros((1, history_size, 128)).to(self.device)
                mel_hat_history = torch.zeros((1, 80, vocoder_history_size)).to(
                    self.device
                )

                # melを64msずつずらしながら食わせることでstreamingで生成する
                audio_items = []
                for i in range(0, y.shape[0], 256 * 6):
                    audio = y[i : i + 256 * 6]
                    audio = F.pad(audio, (0, 256 * 6 - audio.shape[0]))
                    audio = audio.unsqueeze(0)

                    feat = self.asr_model.feature_extractor(audio)

                    feat_history = torch.cat([feat_history, feat], dim=1)[
                        :, -asr_history_size:, :
                    ]

                    feature = self.asr_model.encoder(feat_history)[:, -6:, :]

                    feature_history = torch.cat([feature_history, feature], dim=1)[
                        :, -history_size:, :
                    ]

                    mu, sigma = self.bottleneck(feature_history)
                    # z = mu + torch.exp(sigma) * torch.rand_like(mu)
                    z = torch.cat([mu, sigma], dim=1)
                    mel_hat = self.flow.reverse(z)[:, :, -6:]

                    mel_hat_history = torch.cat([mel_hat_history, mel_hat], dim=2)[
                        :, :, -vocoder_history_size:
                    ]

                    audio_hat = self.vocoder(mel_hat_history)[:, :, -256 * 6 :]
                    audio_items.append(audio_hat)

                    # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                    torch.cuda.empty_cache()

                audio = torch.cat(audio_items, dim=-1)

                current_time = self.get_time()
                print(
                    f"[{current_time}][Step: {self.step}] Finish convert test file: {filepath.name[:24]}"
                )

                self.log.add_audio(
                    f"generated/audio/{filepath.name}", audio[-1], self.step, 24000
                )

        # Resume training
        self.flow.train()
        self.bottleneck.train()
