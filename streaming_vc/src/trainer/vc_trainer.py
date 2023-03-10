import os
import pathlib
import itertools
from datetime import datetime
import random

import datasets
import torch
import torch.nn.functional as F
import torchaudio
from src.data.vc_data_loader import load_data
from src.model.asr_model import ASRModel
from src.model.hifi_gan_generator import Generator
from src.model.mel_gen import MelGenerate
from src.model.spk_rm import SpeakerRemoval
from src.model.discriminator import DiscriminatorMel, DiscriminatorFeat
from src.module.log_melspectrogram import log_melspectrogram
from torch import optim
from torch.utils.tensorboard import SummaryWriter

MAX_XS_LENGTH = 6 * 64


class Trainer:
    """
    VCをトレーニングするクラス。
    """

    def __init__(
        self,
        dataset_dir: str,
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
        self.dataset_dir = dataset_dir
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

        self.spk_rm_lr = 0.0005
        self.d_feat_lr = 0.000001
        self.d_mel_lr = 0.000001
        self.mel_gen_lr = 0.0001
        self.spk_rm_feat_loss_scale = 8.0
        self.spk_rm_mel_loss_scale = 4.0
        self.spk_rm_text_loss_scale = 72.0
        self.mel_gen_loss_scale = 64.0

        self.log.add_text(
            "train/params",
            f"spl_rm_lr: {self.spk_rm_lr}  \n"
            + f"d_feat_lr: {self.d_feat_lr}  \n"
            + f"d_mel_lr: {self.d_mel_lr}  \n"
            + f"mel_gen_lr: {self.mel_gen_lr}  \n"
            + f"spk_rm_feat_loss_scale: {self.spk_rm_feat_loss_scale}  \n"
            + f"spk_rm_mel_loss_scale: {self.spk_rm_mel_loss_scale}  \n"
            + f"spk_rm_text_loss_scale: {self.spk_rm_text_loss_scale}  \n"
            + f"mel_gen_loss_scale: {self.mel_gen_loss_scale}",
            0,
        )

        self.step = 0
        self.start_time = datetime.now()

        (self.real_data_loader, self.fake_data_loader) = load_data(
            self.dataset_dir, self.batch_size
        )

        self.asr_model = ASRModel(vocab_size=32).to(self.device)
        asr_ckpt = torch.load(asr_ckpt_path, map_location=self.device)
        self.asr_model.load_state_dict(asr_ckpt["model"])

        self.mel_gen = MelGenerate().to(self.device)
        self.spk_rm = SpeakerRemoval().to(self.device)
        self.d_feat = DiscriminatorFeat().to(self.device)
        self.d_mel = DiscriminatorMel().to(self.device)

        self.vocoder = Generator().to(self.device).eval()
        vocoder_ckpt = torch.load(vocoder_ckpt_path, map_location=self.device)
        self.vocoder.load_state_dict(vocoder_ckpt["generator"])

        self.optimizer_spk_rm = optim.AdamW(self.spk_rm.parameters(), lr=self.spk_rm_lr)
        self.optimizer_d_feat = optim.AdamW(self.d_feat.parameters(), lr=self.d_feat_lr)
        self.optimizer_d_mel = optim.AdamW(self.d_mel.parameters(), lr=self.d_mel_lr)
        self.optimizer_mel_gen = optim.AdamW(
            itertools.chain(self.spk_rm.parameters(), self.mel_gen.parameters()),
            lr=self.mel_gen_lr,
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
        self.spk_rm.load_state_dict(ckpt["spk_rm"])
        self.mel_gen.load_state_dict(ckpt["mel_gen"])
        self.d_feat.load_state_dict(ckpt["d_feat"])
        self.d_mel.load_state_dict(ckpt["d_mel"])
        self.optimizer_spk_rm.load_state_dict(ckpt["optimizer_spk_rm"])
        self.optimizer_mel_gen.load_state_dict(ckpt["optimizer_mel_gen"])
        self.optimizer_d_feat.load_state_dict(ckpt["optimizer_d_feat"])
        self.optimizer_d_mel.load_state_dict(ckpt["optimizer_d_mel"])
        self.step = ckpt["step"]
        print(f"Load checkpoint from {ckpt_path}")

    def save_ckpt(self):
        """
        ckptを保存する
        """
        # ckpt_path = os.path.join(self.ckpt_dir, f"ckpt-{self.step:0>8}.pt")
        save_dict = {
            "asr_model": self.asr_model.state_dict(),
            "spk_rm": self.spk_rm.state_dict(),
            "mel_gen": self.mel_gen.state_dict(),
            "d_feat": self.d_feat.state_dict(),
            "d_mel": self.d_mel.state_dict(),
            "optimizer_spk_rm": self.optimizer_spk_rm.state_dict(),
            "optimizer_mel_gen": self.optimizer_mel_gen.state_dict(),
            "optimizer_d_feat": self.optimizer_d_feat.state_dict(),
            "optimizer_d_mel": self.optimizer_d_mel.state_dict(),
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
        print("\n")
        print(f"Experiment name: {self.exp_name}")
        print("\n")

        self.start_time = datetime.now()

        def cycle(iterable):
            iterator = iter(iterable)
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    iterator = iter(iterable)

        r_data_loader = cycle(self.real_data_loader)
        f_data_loader = cycle(self.fake_data_loader)

        d_feat_many_losses = []
        d_feat_target_losses = []
        d_feat_all_losses = []

        d_mel_many_losses = []
        d_mel_target_losses = []
        d_mel_all_losses = []

        spk_rm_feat_losses = []
        spk_rm_mel_losses = []
        spk_rm_text_losses = []
        spk_rm_all_losses = []

        mel_gen_losses = []

        mel_errors = []

        def get_random_range(xs):
            start = random.randint(0, max(0, xs.shape[1] - MAX_XS_LENGTH))
            clip_xs = xs[:, start : start + MAX_XS_LENGTH]
            if clip_xs.shape[1] < MAX_XS_LENGTH:
                clip_xs = F.pad(
                    clip_xs, (0, 0, 0, MAX_XS_LENGTH - clip_xs.shape[1]), "constant"
                ).to(self.device)
            return clip_xs

        while self.step < self.max_step:
            x_many = next(f_data_loader).to(self.device).squeeze(1)
            x_target = next(r_data_loader).to(self.device).squeeze(1)

            if self.step % 4 == 0:
                # d_featの学習
                xs = self.asr_model.feature_extractor(x_many)
                xs = self.asr_model.encoder(xs)
                clip_xs = get_random_range(xs)
                xs = self.spk_rm(clip_xs)
                xs = self.d_feat(xs)
                d_feat_many_loss = F.binary_cross_entropy(xs, torch.zeros_like(xs))
                d_feat_many_losses.append(d_feat_many_loss.item())

                xs = self.asr_model.feature_extractor(x_target)
                xs = self.asr_model.encoder(xs)
                clip_xs = get_random_range(xs)
                xs = self.spk_rm(clip_xs)
                xs = self.d_feat(xs)
                d_feat_target_loss = F.binary_cross_entropy(xs, torch.ones_like(xs))
                d_feat_target_losses.append(d_feat_target_loss.item())

                d_feat_all_loss = d_feat_many_loss + d_feat_target_loss
                d_feat_all_losses.append(d_feat_all_loss.item())

                self.optimizer_d_feat.zero_grad()
                d_feat_all_loss.backward()
                self.optimizer_d_feat.step()

            if self.step % 2 == 0:
                # d_melの学習
                xs = self.asr_model.feature_extractor(x_many)
                xs = self.asr_model.encoder(xs)
                clip_xs = get_random_range(xs)
                xs = self.spk_rm(clip_xs)
                xs = self.mel_gen(xs)
                xs = self.d_mel(xs)
                d_mel_many_loss = F.binary_cross_entropy(xs, torch.zeros_like(xs))
                d_mel_many_losses.append(d_mel_many_loss.item())

                xs = self.asr_model.feature_extractor(x_target)
                xs = self.asr_model.encoder(xs)
                clip_xs = get_random_range(xs)
                xs = self.spk_rm(clip_xs)
                xs = self.mel_gen(xs)
                xs = self.d_mel(xs)
                d_mel_target_loss = F.binary_cross_entropy(xs, torch.ones_like(xs))
                d_mel_target_losses.append(d_mel_target_loss.item())

                d_mel_all_loss = d_mel_many_loss + d_mel_target_loss
                d_mel_all_losses.append(d_mel_all_loss.item())

                self.optimizer_d_mel.zero_grad()
                d_mel_all_loss.backward()
                self.optimizer_d_mel.step()

            # spk_rmの学習
            xs = self.asr_model.feature_extractor(x_many)
            xs = self.asr_model.encoder(xs)
            clip_xs = get_random_range(xs)
            xs = self.spk_rm(clip_xs)
            xs = self.d_feat(xs)
            spk_rm_feat_loss = (
                F.binary_cross_entropy(xs, torch.ones_like(xs))
                * self.spk_rm_feat_loss_scale
            )
            spk_rm_feat_losses.append(spk_rm_feat_loss.item())

            xs = self.asr_model.feature_extractor(x_many)
            xs = self.asr_model.encoder(xs)
            clip_xs = get_random_range(xs)
            xs = self.spk_rm(clip_xs)
            xs = self.mel_gen(xs)
            xs = self.d_mel(xs)
            spk_rm_mel_loss = (
                F.binary_cross_entropy(xs, torch.ones_like(xs))
                * self.spk_rm_mel_loss_scale
            )
            spk_rm_mel_losses.append(spk_rm_mel_loss.item())

            xs = self.asr_model.feature_extractor(x_many)
            xs = self.asr_model.encoder(xs)
            clip_xs = get_random_range(xs)
            text_wo_spk_rm = F.log_softmax(self.asr_model.ctc_layers(clip_xs), dim=-1)
            xs = self.spk_rm(clip_xs)
            text_w_spk_rm = F.log_softmax(self.asr_model.ctc_layers(xs), dim=-1)
            spk_rm_text_loss = (
                F.mse_loss(text_wo_spk_rm, text_w_spk_rm) * self.spk_rm_text_loss_scale
            )
            spk_rm_text_losses.append(spk_rm_text_loss.item())

            spk_rm_all_loss = spk_rm_feat_loss + spk_rm_mel_loss + spk_rm_text_loss
            spk_rm_all_losses.append(spk_rm_all_loss.item())

            self.optimizer_spk_rm.zero_grad()
            spk_rm_all_loss.backward()
            self.optimizer_spk_rm.step()

            # mel_genの学習
            start = random.randint(0, max(0, x_target.shape[1] - MAX_XS_LENGTH * 256))
            clip_x_target = x_target[:, start : start + MAX_XS_LENGTH * 256]
            if clip_x_target.shape[1] < MAX_XS_LENGTH * 256:
                clip_x_target = F.pad(
                    clip_x_target,
                    (0, MAX_XS_LENGTH * 256 - clip_x_target.shape[1]),
                    "constant",
                )

            clip_x_target_mel = torchaudio.transforms.MelSpectrogram(
                n_fft=1024,
                n_mels=80,
                sample_rate=24000,
                hop_length=256,
                win_length=1024,
            ).to(self.device)(clip_x_target)
            clip_x_target_mel = log_melspectrogram(clip_x_target_mel).squeeze(0)[
                :, :, :MAX_XS_LENGTH
            ]

            xs = self.asr_model.feature_extractor(clip_x_target)
            xs = self.asr_model.encoder(xs)
            xs = self.spk_rm(xs)
            target_mel_hat = self.mel_gen(xs)

            mel_gen_loss = (
                F.mse_loss(target_mel_hat, clip_x_target_mel) * self.mel_gen_loss_scale
            )
            mel_gen_losses.append(mel_gen_loss.item())

            self.optimizer_mel_gen.zero_grad()
            mel_gen_loss.backward()
            self.optimizer_mel_gen.step()

            ## mel error
            mel_error = F.l1_loss(clip_x_target_mel, target_mel_hat).item()
            mel_errors.append(mel_error)

            # ロギング
            if self.step % self.progress_step == 0:

                ## console
                current_time = self.get_time()
                print(
                    f"[{current_time}][Step: {self.step}] mel gen loss: {sum(mel_gen_losses) / len(mel_gen_losses)}, mel error: {sum(mel_errors)/ len(mel_errors)}",
                )

                ## tensorboard
                self.log.add_scalar(
                    "train/d_feat_many_loss",
                    sum(d_feat_many_losses) / len(d_feat_many_losses),
                    self.step,
                )
                self.log.add_scalar(
                    "train/d_feat_target_loss",
                    sum(d_feat_target_losses) / len(d_feat_target_losses),
                    self.step,
                )
                self.log.add_scalar(
                    "train/d_feat_all_loss",
                    sum(d_feat_all_losses) / len(d_feat_all_losses),
                    self.step,
                )

                self.log.add_scalar(
                    "train/d_mel_many_loss",
                    sum(d_mel_many_losses) / len(d_mel_many_losses),
                    self.step,
                )
                self.log.add_scalar(
                    "train/d_mel_target_loss",
                    sum(d_mel_target_losses) / len(d_mel_target_losses),
                    self.step,
                )
                self.log.add_scalar(
                    "train/d_mel_all_loss",
                    sum(d_mel_all_losses) / len(d_mel_all_losses),
                    self.step,
                )

                self.log.add_scalar(
                    "train/spk_rm_feat_loss",
                    sum(spk_rm_feat_losses) / len(spk_rm_feat_losses),
                    self.step,
                )
                self.log.add_scalar(
                    "train/spk_rm_mel_loss",
                    sum(spk_rm_mel_losses) / len(spk_rm_mel_losses),
                    self.step,
                )
                self.log.add_scalar(
                    "train/spk_rm_text_loss",
                    sum(spk_rm_text_losses) / len(spk_rm_text_losses),
                    self.step,
                )
                self.log.add_scalar(
                    "train/spk_rm_all_loss",
                    sum(spk_rm_all_losses) / len(spk_rm_all_losses),
                    self.step,
                )

                self.log.add_scalar(
                    "train/mel_gen_loss",
                    sum(mel_gen_losses) / len(mel_gen_losses),
                    self.step,
                )

                self.log.add_scalar(
                    "train/mel_error", sum(mel_errors) / len(mel_errors), self.step
                )

                # clear loss buffer
                d_feat_many_losses = []
                d_feat_target_losses = []
                d_feat_all_losses = []

                d_mel_many_losses = []
                d_mel_target_losses = []
                d_mel_all_losses = []

                spk_rm_feat_losses = []
                spk_rm_mel_losses = []
                spk_rm_text_losses = []
                spk_rm_all_losses = []

                mel_gen_losses = []

                mel_errors = []

                if self.step % 100 == 0:
                    # 適当にmelをremapして画像として保存
                    self.log.add_image(
                        "mel",
                        (clip_x_target_mel[0] + 15) / 30,
                        self.step,
                        dataformats="HW",
                    )
                    self.log.add_image(
                        "mel_hat",
                        (target_mel_hat[0] + 15) / 30,
                        self.step,
                        dataformats="HW",
                    )

            # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
            torch.cuda.empty_cache()

            # バリデーションの実行
            if self.step % self.valid_step == 0:
                self.validate()
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
                feat_history = torch.zeros((1, asr_history_size, 240)).to(self.device)
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

                    feat_history = torch.cat([feat_history, feat], dim=1)

                    feature = self.spk_rm(
                        self.asr_model.encoder(feat_history[:, -asr_history_size:, :])
                    )[:, -6:, :]

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
