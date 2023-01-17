import itertools
import os
import pathlib
from datetime import datetime

import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
import torchaudio
from src.data.data_loader import load_dataset
from src.model.generator import Generator
from src.model.multi_period_discriminator import MultiPeriodDiscriminator
from src.model.multi_scale_discriminator import MultiScaleDiscriminator
from src.model.vc_model import VCModel
from src.module.log_melspectrogram import log_melspectrogram
from src.trainer.loss import discriminator_loss, feature_loss, generator_loss
from torch import optim
from torch.utils.tensorboard import SummaryWriter

SEGMENT_SIZE = 6 * 256 * 24


class Finetune:
    """
    HiFi-GANのfine tuningとvcの訓練を進めるクラス
    """

    def __init__(
        self,
        dataset_dir: str,
        testdata_dir: str,
        feature_extractor_onnx_path: str,
        encoder_onnx_path: str,
        vc_ckpt_path: str,
        vocoder_ckpt_path: str,
        batch_size: int = 4,
        max_step: int = 10000001,
        progress_step: int = 10,
        valid_step: int = 1000,
        exp_name: str | None = None,
    ):
        self.n_epochs = 0
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

        self.ckpt_dir = os.path.join("output/finetune/ckpt", self.exp_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.log_dir = os.path.join("output/finetune/log", self.exp_name)
        self.log = SummaryWriter(self.log_dir)

        self.step = 0
        self.start_time = datetime.now()

        self.best_error = 100.0

        self.train_loader = load_dataset(self.dataset_dir, self.batch_size)

        self.feature_extractor = onnxruntime.InferenceSession(
            feature_extractor_onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.encoder = onnxruntime.InferenceSession(
            encoder_onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        self.vc_model = VCModel().to(self.device)
        vc_ckpt = torch.load(vc_ckpt_path, map_location=self.device)
        self.vc_model.load_state_dict(vc_ckpt["model"])

        vocoder_ckpt = torch.load(vocoder_ckpt_path, map_location=self.device)

        self.generator = Generator().to(self.device).eval()
        self.generator.load_state_dict(vocoder_ckpt["generator"])

        self.mpd = MultiPeriodDiscriminator().to(self.device)
        self.mpd.load_state_dict(vocoder_ckpt["mpd"])

        self.msd = MultiScaleDiscriminator().to(self.device)
        self.msd.load_state_dict(vocoder_ckpt["msd"])

        self.optimizer_g = optim.AdamW(
            itertools.chain(self.vc_model.parameters(), self.generator.parameters()),
            lr=0.0002,
            betas=(0.8, 0.99),
        )
        self.optimizer_d = optim.AdamW(
            itertools.chain(self.mpd.parameters(), self.msd.parameters()),
            lr=0.0002,
            betas=(0.8, 0.99),
        )

        if exp_name is not None:
            self.load_ckpt()

    def load_ckpt(self):
        """
        ckptから復元する
        """
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt-latest.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.vc_model.load_state_dict(ckpt["vc_model"])
        self.generator.load_state_dict(ckpt["generator"])
        self.mpd.load_state_dict(ckpt["mpd"])
        self.msd.load_state_dict(ckpt["msd"])
        self.optimizer_g.load_state_dict(ckpt["optimizer_g"])
        self.optimizer_d.load_state_dict(ckpt["optimizer_d"])
        self.step = ckpt["step"]
        self.n_epochs = ckpt["n_epochs"]
        print(f"Load checkpoint from {ckpt_path}")

    def save_ckpt(self):
        """
        ckptを保存する
        """
        ckpt_path = os.path.join(self.ckpt_dir, f"ckpt-{self.step:0>8}.pt")
        latest_path = os.path.join(self.ckpt_dir, "ckpt-latest.pt")
        save_dict = {
            "vc_model": self.vc_model.state_dict(),
            "generator": self.generator.state_dict(),
            "mpd": self.mpd.state_dict(),
            "msd": self.msd.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "step": self.step,
            "n_epochs": self.n_epochs,
        }
        torch.save(save_dict, ckpt_path)
        torch.save(save_dict, latest_path)

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

    def feature_extract(self, audio: torch.Tensor) -> torch.Tensor:
        """
        音声をfeature_extractorに通してencoder用の特徴量を取得する
        """
        input_name = self.feature_extractor.get_inputs()[0].name
        input_lengths_name = self.feature_extractor.get_inputs()[1].name
        audio = audio.contiguous()
        audio_length = (
            torch.sum(torch.ones_like(audio), dim=-1).to(dtype=torch.int64).contiguous()
        )

        output_name = self.feature_extractor.get_outputs()[0].name
        output = torch.empty(
            (audio.shape[0], audio.shape[1] // 256, 240),
            device=self.device,
            dtype=torch.float32,
        ).contiguous()

        binding = self.feature_extractor.io_binding()

        binding.bind_input(
            name=input_name,
            device_type="cuda",
            device_id=0,
            element_type=np.float32,
            shape=tuple(audio.shape),
            buffer_ptr=audio.data_ptr(),
        )

        binding.bind_input(
            name=input_lengths_name,
            device_type="cuda",
            device_id=0,
            element_type=np.int64,
            shape=tuple(audio_length.shape),
            buffer_ptr=audio_length.data_ptr(),
        )

        binding.bind_output(
            name=output_name,
            device_type="cuda",
            device_id=0,
            element_type=np.float32,
            shape=tuple(output.shape),
            buffer_ptr=output.data_ptr(),
        )

        self.feature_extractor.run_with_iobinding(binding)
        return output

    def encode(self, feat: torch.Tensor) -> torch.Tensor:
        """
        encoderの特徴量をencoderに通してdecoder用の特徴量を取得する
        """
        input_name = self.encoder.get_inputs()[0].name
        input_lengths_name = self.encoder.get_inputs()[1].name
        feat = feat.contiguous()
        feat_length = (
            torch.sum(torch.ones((feat.shape[0], feat.shape[1])), dim=-1)
            .to(device=self.device, dtype=torch.int64)
            .contiguous()
        )

        output_name = self.encoder.get_outputs()[0].name
        output = torch.empty(
            (feat.shape[0], feat.shape[1], 32),
            device=self.device,
            dtype=torch.float32,
        ).contiguous()

        binding = self.encoder.io_binding()

        binding.bind_input(
            name=input_name,
            device_type="cuda",
            device_id=0,
            element_type=np.float32,
            shape=tuple(feat.shape),
            buffer_ptr=feat.data_ptr(),
        )

        binding.bind_input(
            name=input_lengths_name,
            device_type="cuda",
            device_id=0,
            element_type=np.int64,
            shape=tuple(feat_length.shape),
            buffer_ptr=feat_length.data_ptr(),
        )

        binding.bind_output(
            name=output_name,
            device_type="cuda",
            device_id=0,
            element_type=np.float32,
            shape=tuple(output.shape),
            buffer_ptr=output.data_ptr(),
        )

        self.encoder.run_with_iobinding(binding)
        return output

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

        vc_losses = []
        disc_losses = []
        gen_losses = []
        mel_errors = []

        while self.step < self.max_step:
            for audio, mel in self.train_loader:
                audio = torch.autograd.Variable(audio.to(device=self.device))
                mel = torch.autograd.Variable(mel.to(device=self.device))

                feat = self.feature_extract(audio.squeeze(1))
                feature = self.encode(feat)

                mel_hat = self.vc_model(feature)

                # calc vc loss
                loss_vc = F.mse_loss(mel_hat, mel) * 25

                # HiFi-GAN finetune
                audio_g_hat = self.generator(mel_hat)
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
                loss_mel = F.l1_loss(mel_g_hat, mel) * 45

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
                    loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + loss_vc
                )

                loss_gen_all.backward()
                self.optimizer_g.step()

                vc_losses.append(loss_vc.item())
                disc_losses.append(loss_disc_all.item())
                gen_losses.append(loss_gen_all.item())
                mel_errors.append(F.l1_loss(mel, mel_g_hat).item())

                # ロギング
                if self.step % self.progress_step == 0:
                    ## console
                    current_time = self.get_time()
                    print(
                        f"[{current_time}][Epochs: {self.n_epochs}, Step: {self.step}] vc_loss: {loss_vc.item():.4f}, d_loss: {loss_disc_all.item():.4f}, g_loss: {loss_gen_all.item():.4f}",
                    )
                    ## tensorboard
                    self.log.add_scalar(
                        "train/loss/vc", sum(vc_losses) / len(vc_losses), self.step
                    )
                    self.log.add_scalar(
                        "train/loss/d", sum(disc_losses) / len(disc_losses), self.step
                    )
                    self.log.add_scalar(
                        "train/loss/g", sum(gen_losses) / len(gen_losses), self.step
                    )
                    self.log.add_scalar(
                        "train/mel_error", sum(mel_errors) / len(mel_errors), self.step
                    )
                    ## clear loss buffer
                    vc_losses = []
                    disc_losses = []
                    gen_losses = []
                    mel_errors = []

                    if self.step % 100 == 0:
                        # 適当にmelをremapして画像として保存
                        self.log.add_image(
                            "mel", (mel[0] + 15) / 30, self.step, dataformats="HW"
                        )
                        self.log.add_image(
                            "mel_hat",
                            (mel_hat[0] + 15) / 30,
                            self.step,
                            dataformats="HW",
                        )
                        self.log.add_image(
                            "mel_g_hat",
                            (mel_g_hat[0] + 15) / 30,
                            self.step,
                            dataformats="HW",
                        )

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
        self.vc_model.eval()
        self.generator.eval()

        if self.step == 0:
            for filepath in pathlib.Path(self.testdata_dir).rglob("*.wav"):
                y, sr = torchaudio.load(str(filepath))
                y = torchaudio.transforms.Resample(sr, 24000)(y).to(device=self.device)
                self.log.add_audio(f"before/audio/{filepath.name}", y, self.step, 24000)

        # テストデータで試す
        with torch.no_grad():
            history_size = 6 * 24
            vocoder_history_size = 16

            for filepath in pathlib.Path(self.testdata_dir).rglob("*.wav"):
                current_time = self.get_time()
                print(
                    f"[{current_time}][Epochs: {self.n_epochs}, Step: {self.step}] Start convert : {filepath.name[:24]}"
                )

                y, sr = torchaudio.load(str(filepath))
                y = torchaudio.transforms.Resample(sr, 24000)(y).squeeze(0)
                y = y.to(device=self.device)

                # melを64msずつずらしながら食わせることでstreamingで生成する
                feat_history: torch.Tensor | None = None
                mel_history: torch.Tensor | None = None
                mel_hat_history: torch.Tensor | None = None
                audio_items = []
                for i in range(0, y.shape[0], 256 * 6):
                    audio = y[i : i + 256 * 6]
                    audio = F.pad(audio, (0, 256 * 6 - audio.shape[0]))
                    audio = audio.unsqueeze(0)

                    feat = self.feature_extract(audio)

                    if feat_history is None:
                        feat_history = feat
                    else:
                        feat_history = torch.cat([feat_history, feat], dim=1)

                    feature = self.encode(feat_history[:, -history_size:, :])[:, -6:, :]

                    if mel_history is None:
                        mel_history = feature
                    else:
                        mel_history = torch.cat([mel_history, feature], dim=1)

                    mel_hat = self.vc_model(mel_history[:, -history_size:, :])[
                        :, :, -6:
                    ]

                    if mel_hat_history is None:
                        mel_hat_history = mel_hat
                    else:
                        mel_hat_history = torch.cat([mel_hat_history, mel_hat], dim=2)

                    audio_hat = self.generator(
                        mel_hat_history[:, :, -vocoder_history_size:]
                    )[:, :, -256 * 6 :]
                    audio_items.append(audio_hat)

                    # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                    torch.cuda.empty_cache()

                audio = torch.cat(audio_items, dim=-1)

                current_time = self.get_time()
                print(
                    f"[{current_time}][Epochs: {self.n_epochs}, Step: {self.step}] Finish convert: {filepath.name[:24]}"
                )

                self.log.add_audio(
                    f"generated/audio/{filepath.name}", audio[-1], self.step, 24000
                )

        # save ckpt
        self.save_ckpt()

        # Resume training
        self.vc_model.train()
        self.generator.train()
