import os
import math
import sys
import pathlib
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from src.data.vc_data_loader import load_data
from src.model.asr_model import ASRModel
from src.model.hifi_gan_generator import Generator
from src.model.mel_gen import MelGenerator, MelReverse
from src.model.discriminator import DiscriminatorMel, DiscriminatorFeat
from torch import optim
from torch.utils.tensorboard import SummaryWriter


AUDIO_LENGTH = int(24000 * 1.0)
MEL_LENGTH = int(24000 * 1.0 / 256.0)


class Trainer:
    """
    VCをトレーニングするクラス。
    """

    def __init__(
        self,
        source_data_dir: str,
        target_data_dir: str,
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
        self.source_data_dir = source_data_dir
        self.target_data_dir = target_data_dir
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

        self.asr_model = ASRModel(vocab_size=32).to(self.device)
        asr_ckpt = torch.load(asr_ckpt_path, map_location=self.device)
        self.asr_model.load_state_dict(asr_ckpt["model"])

        self.mel_gen = MelGenerator().to(self.device)
        self.mel_rev = MelReverse().to(self.device)

        self.d_target_mel = DiscriminatorMel().to(self.device)
        self.d_source_feat = DiscriminatorFeat().to(self.device)

        self.vocoder = Generator().to(self.device).eval()
        vocoder_ckpt = torch.load(vocoder_ckpt_path, map_location=self.device)
        self.vocoder.load_state_dict(vocoder_ckpt["generator"])

        (
            self.source_audio_loader,
            self.target_mel_loader,
        ) = load_data(source_data_dir, target_data_dir, batch_size)

        self.optim_target_d = optim.Adam(self.d_target_mel.parameters(), lr=0.0001)
        self.optim_source_d = optim.Adam(self.d_source_feat.parameters(), lr=0.0001)
        self.optim_target_g = optim.Adam(self.mel_gen.parameters(), lr=0.0002)
        self.optim_source_g = optim.Adam(self.mel_rev.parameters(), lr=0.0002)
        self.optim_cycle = optim.Adam(
            list(self.mel_gen.parameters()) + list(self.mel_rev.parameters()),
            lr=0.0001,
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
        self.mel_gen.load_state_dict(ckpt["mel_gen"])
        self.mel_rev.load_state_dict(ckpt["mel_rev"])
        self.d_target_mel.load_state_dict(ckpt["d_target_mel"])
        self.d_source_feat.load_state_dict(ckpt["d_source_feat"])
        self.optim_target_d.load_state_dict(ckpt["optim_target_d"])
        self.optim_source_d.load_state_dict(ckpt["optim_source_d"])
        self.optim_target_g.load_state_dict(ckpt["optim_target_g"])
        self.optim_source_g.load_state_dict(ckpt["optim_source_g"])
        self.optim_cycle.load_state_dict(ckpt["optim_cycle"])

        print(f"Load checkpoint from {ckpt_path}")

    def save_ckpt(self):
        save_dict = {
            "asr_encoder": self.asr_model.encoder.state_dict(),
            "mel_gen": self.mel_gen.state_dict(),
            "mel_rev": self.mel_rev.state_dict(),
            "d_target_mel": self.d_target_mel.state_dict(),
            "d_source_feat": self.d_source_feat.state_dict(),
            "optim_target_d": self.optim_target_d.state_dict(),
            "optim_source_d": self.optim_source_d.state_dict(),
            "optim_target_g": self.optim_target_g.state_dict(),
            "optim_source_g": self.optim_source_g.state_dict(),
            "optim_cycle": self.optim_cycle.state_dict(),
            "step": self.step,
        }
        # torch.save(save_dict, ckpt_path)

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
        print("\n\n")

        def cycle(iterable):
            iterator = iter(iterable)
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    iterator = iter(iterable)

        source_audio_loader = cycle(self.source_audio_loader)
        target_mel_loader = cycle(self.target_mel_loader)

        target_d_losses = []
        target_g_losses = []
        source_d_losses = []
        source_g_losses = []
        target_cycle_losses = []
        source_cycle_losses = []
        cycle_losses = []

        while self.step < self.max_step:

            # --------------------
            # gan loss

            ## source feat -> target mel

            ### discriminator
            x_source_audio = next(source_audio_loader).to(self.device).squeeze(1)
            x_source_feature, audio_len = self.asr_model.feature_extractor(
                x_source_audio, torch.tensor([AUDIO_LENGTH])
            )
            x_source_feat, _ = self.asr_model.encoder.forward_train(
                x_source_feature, audio_len
            )
            x_target_mel = next(target_mel_loader).to(self.device).squeeze(1)
            x_source_mel = self.mel_gen(x_source_feat)

            xs_t = self.d_target_mel(x_target_mel)
            xs_s = self.d_target_mel(x_source_mel)

            loss_target_d = F.binary_cross_entropy(
                xs_t, torch.ones_like(xs_t)
            ) + F.binary_cross_entropy(xs_s, torch.zeros_like(xs_s))
            target_d_losses.append(loss_target_d.item())

            self.optim_target_d.zero_grad()
            loss_target_d.backward()
            self.optim_target_d.step()

            ### generator
            x_source_audio = next(source_audio_loader).to(self.device).squeeze(1)
            x_source_feature, audio_len = self.asr_model.feature_extractor(
                x_source_audio, torch.tensor([AUDIO_LENGTH])
            )
            x_source_feat, _ = self.asr_model.encoder.forward_train(
                x_source_feature, audio_len
            )
            x_target_mel = next(target_mel_loader).to(self.device).squeeze(1)
            x_source_mel = self.mel_gen(x_source_feat)

            xs_t = self.d_target_mel(x_target_mel)
            xs_s = self.d_target_mel(x_source_mel)

            loss_target_g = F.binary_cross_entropy(
                xs_t, torch.ones_like(xs_t)
            ) + F.binary_cross_entropy(xs_s, torch.ones_like(xs_s))
            target_g_losses.append(loss_target_g.item())

            self.optim_target_g.zero_grad()
            loss_target_g.backward()
            self.optim_target_g.step()

            ## target mel -> source feat

            ### discriminator
            x_source_audio = next(source_audio_loader).to(self.device).squeeze(1)
            x_source_feature, audio_len = self.asr_model.feature_extractor(
                x_source_audio, torch.tensor([AUDIO_LENGTH])
            )
            x_source_feat, _ = self.asr_model.encoder.forward_train(
                x_source_feature, audio_len
            )
            x_target_mel = next(target_mel_loader).to(self.device).squeeze(1)
            x_target_feat = self.mel_rev(x_target_mel)

            xs_s = self.d_source_feat(x_source_feat)
            xs_t = self.d_source_feat(x_target_feat)

            loss_source_d = F.binary_cross_entropy(
                xs_s, torch.ones_like(xs_s)
            ) + F.binary_cross_entropy(xs_t, torch.zeros_like(xs_t))
            source_d_losses.append(loss_source_d.item())

            self.optim_source_d.zero_grad()
            loss_source_d.backward()
            self.optim_source_d.step()

            ### generator
            x_source_audio = next(source_audio_loader).to(self.device).squeeze(1)
            x_source_feature, audio_len = self.asr_model.feature_extractor(
                x_source_audio, torch.tensor([AUDIO_LENGTH])
            )
            x_source_feat, _ = self.asr_model.encoder.forward_train(
                x_source_feature, audio_len
            )
            x_target_mel = next(target_mel_loader).to(self.device).squeeze(1)
            x_target_feat = self.mel_rev(x_target_mel)

            xs_s = self.d_source_feat(x_source_feat)
            xs_t = self.d_source_feat(x_target_feat)

            loss_source_g = F.binary_cross_entropy(
                xs_s, torch.ones_like(xs_s)
            ) + F.binary_cross_entropy(xs_t, torch.ones_like(xs_t))
            source_g_losses.append(loss_source_g.item())

            self.optim_source_g.zero_grad()
            loss_source_g.backward()
            self.optim_source_g.step()

            # --------------------
            # cycle loss

            x_source_audio = next(source_audio_loader).to(self.device).squeeze(1)
            x_source_feature, audio_len = self.asr_model.feature_extractor(
                x_source_audio, torch.tensor([AUDIO_LENGTH])
            )
            x_source_feat, _ = self.asr_model.encoder.forward_train(
                x_source_feature, audio_len
            )
            x_target_mel = next(target_mel_loader).to(self.device).squeeze(1)
            x_target_feat = self.mel_rev(x_target_mel)

            ## source feat -> source mel -> source feat'
            x_source_mel = self.mel_gen(x_source_feat)
            x_source_feat_prime = self.mel_rev(x_source_mel)

            loss_cycle_feat_source = F.mse_loss(x_source_feat, x_source_feat_prime)
            source_cycle_losses.append(loss_cycle_feat_source.item())

            ## target mel -> target feat -> target mel'
            x_target_feat = self.mel_rev(x_target_mel)
            x_target_mel_prime = self.mel_gen(x_target_feat)

            loss_cycle_mel_target = F.mse_loss(x_target_mel, x_target_mel_prime)
            target_cycle_losses.append(loss_cycle_mel_target.item())

            ## total cycle loss
            loss_cycle = loss_cycle_feat_source + loss_cycle_mel_target
            cycle_losses.append(loss_cycle.item())

            self.optim_cycle.zero_grad()
            loss_cycle.backward()
            self.optim_cycle.step()

            # ロギング
            if self.step % self.progress_step == 0:
                # calculate mean of losses and cers in progress steps
                target_d_loss = sum(target_d_losses) / len(target_d_losses)
                target_g_loss = sum(target_g_losses) / len(target_g_losses)
                source_d_loss = sum(source_d_losses) / len(source_d_losses)
                source_g_loss = sum(source_g_losses) / len(source_g_losses)
                target_cycle_loss = sum(target_cycle_losses) / len(target_cycle_losses)
                source_cycle_loss = sum(source_cycle_losses) / len(source_cycle_losses)
                cycle_loss = sum(cycle_losses) / len(cycle_losses)

                ## console
                current_time = self.get_time()
                print(
                    f"[{current_time}][Step: {self.step}] target_d_loss: {target_d_loss:.3f}, target_g_loss: {target_g_loss:.3f}, source_d_loss: {source_d_loss:.3f}, source_g_loss: {source_g_loss:.3f}, target_cycle_loss: {target_cycle_loss:.3f}, source_cycle_loss: {source_cycle_loss:.3f}, cycle_loss: {cycle_loss:.3f}",
                )

                ## tensorboard
                self.log.add_scalar("train/target_d_loss", target_d_loss, self.step)
                self.log.add_scalar("train/target_g_loss", target_g_loss, self.step)
                self.log.add_scalar("train/source_d_loss", source_d_loss, self.step)
                self.log.add_scalar("train/source_g_loss", source_g_loss, self.step)
                self.log.add_scalar(
                    "train/target_cycle_loss", target_cycle_loss, self.step
                )
                self.log.add_scalar(
                    "train/source_cycle_loss", source_cycle_loss, self.step
                )
                self.log.add_scalar("train/cycle_loss", cycle_loss, self.step)

                # reset losses
                target_d_losses = []
                target_g_losses = []
                source_d_losses = []
                source_g_losses = []
                target_cycle_losses = []
                source_cycle_losses = []
                cycle_losses = []

            # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
            torch.cuda.empty_cache()

            # バリデーションの実行
            if self.step % self.valid_step == 0:
                self.validate()

            # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
            torch.cuda.empty_cache()

            # End of step
            self.step += 1
            if self.step > self.max_step:
                break
        self.log.close()

    def validate(self):
        self.asr_model.eval()
        self.mel_gen.eval()

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
                feature_history = torch.zeros((1, history_size, 128)).to(self.device)
                mel_hat_history = torch.zeros((1, 80, vocoder_history_size)).to(
                    self.device
                )

                # encoder layer history
                asr_encoder_history_layer_1 = torch.zeros((1, 6, 512)).to(self.device)
                asr_encoder_history_layer_2 = torch.zeros((1, 6, 512)).to(self.device)
                asr_encoder_history_layer_3 = torch.zeros((1, 6, 512)).to(self.device)
                asr_encoder_history_layer_4 = torch.zeros((1, 6, 512)).to(self.device)
                asr_encoder_history_layer_5 = torch.zeros((1, 6, 512)).to(self.device)
                asr_encoder_history_layer_6 = torch.zeros((1, 6, 512)).to(self.device)

                # melを64msずつずらしながら食わせることでstreamingで生成する
                audio_items = []
                for i in range(0, y.shape[0], 256 * 6):
                    audio = y[i : i + 256 * 6]
                    audio = F.pad(audio, (0, 256 * 6 - audio.shape[0]))
                    audio = audio.unsqueeze(0)

                    feat, _ = self.asr_model.feature_extractor(
                        audio, torch.tensor([256 * 6])
                    )

                    (
                        feature,
                        asr_encoder_history_layer_1,
                        asr_encoder_history_layer_2,
                        asr_encoder_history_layer_3,
                        asr_encoder_history_layer_4,
                        asr_encoder_history_layer_5,
                        asr_encoder_history_layer_6,
                    ) = self.asr_model.encoder(
                        feat,
                        asr_encoder_history_layer_1,
                        asr_encoder_history_layer_2,
                        asr_encoder_history_layer_3,
                        asr_encoder_history_layer_4,
                        asr_encoder_history_layer_5,
                        asr_encoder_history_layer_6,
                    )

                    feature_history = torch.cat([feature_history, feature], dim=1)

                    mel_hat = self.mel_gen(feature_history[:, -history_size:, :])[
                        :, :, -6:
                    ]

                    mel_hat_history = torch.cat([mel_hat_history, mel_hat], dim=2)

                    audio_hat = self.vocoder(
                        mel_hat_history[:, :, -vocoder_history_size:]
                    )[:, :, -256 * 6 :]
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

        # save ckpt
        self.save_ckpt()

        # Resume training
        self.asr_model.train()
        self.mel_gen.train()
