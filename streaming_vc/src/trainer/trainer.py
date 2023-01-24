import os
import pathlib
from datetime import datetime
from itertools import cycle

import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
import torchaudio
from src.data.data_loader import load_data
from src.model.generator import Generator
from src.model.vc_model import VCModel
from src.model.vc_discriminator import VCDiscriminator
from torch import optim
from torch.utils.tensorboard import SummaryWriter

SEGMENT_SIZE = 6 * 256 * 16


class Trainer:
    """
    VCをトレーニングするクラス。
    """

    def __init__(
        self,
        dataset_dir: str,
        testdata_dir: str,
        feature_extractor_onnx_path: str,
        encoder_onnx_path: str,
        vocoder_ckpt_path: str,
        batch_size: int = 4,
        max_step: int = 10000001,
        progress_step: int = 10,
        valid_step: int = 5000,
        exp_name: str | None = None,
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

        self.ckpt_dir = os.path.join("output/vc/ckpt", self.exp_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.log_dir = os.path.join("output/vc/log", self.exp_name)
        self.log = SummaryWriter(self.log_dir)

        self.step = 0
        self.start_time = datetime.now()

        self.best_error = 100.0

        (self.train_loader, self.truth_loader, self.fake_loader) = load_data(
            self.dataset_dir, self.batch_size
        )

        self.feature_extractor = onnxruntime.InferenceSession(
            feature_extractor_onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.encoder = onnxruntime.InferenceSession(
            encoder_onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        self.model = VCModel().to(self.device)
        self.discriminator = VCDiscriminator().to(self.device)

        self.vocoder = Generator().to(self.device).eval()
        vocoder_ckpt = torch.load(vocoder_ckpt_path, map_location=self.device)
        self.vocoder.load_state_dict(vocoder_ckpt["generator"])

        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0005)
        self.optimizer_d = optim.AdamW(self.discriminator.parameters(), lr=0.00001)
        self.optimizer_g = optim.AdamW(self.model.parameters(), lr=0.00005)

        if exp_name is not None:
            self.load_ckpt()

    def load_ckpt(self):
        """
        ckptから復元する
        """
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt-latest.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.optimizer_d.load_state_dict(ckpt["optimizer_d"])
        self.optimizer_g.load_state_dict(ckpt["optimizer_g"])
        self.step = ckpt["step"]
        self.best_error = ckpt["best_error"]
        print(f"Load checkpoint from {ckpt_path}")

    def save_ckpt(self):
        """
        ckptを保存する
        """
        ckpt_path = os.path.join(self.ckpt_dir, f"ckpt-{self.step:0>8}.pt")
        save_dict = {
            "model": self.model.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "step": self.step,
            "best_error": self.best_error,
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
            (feat.shape[0], feat.shape[1], 128),
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
        for k in self.model.state_dict().keys():
            print(f"\t{k}")
        print("\n")

        self.start_time = datetime.now()
        self.optimizer.zero_grad()

        losses = []
        d_losses = []
        g_losses = []

        def cycle(iterable):
            iterator = iter(iterable)
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    iterator = iter(iterable)

        t_data_loader = cycle(self.truth_loader)
        f_data_loader = cycle(self.fake_loader)

        while self.step < self.max_step:
            for audio, mel in self.train_loader:
                # MSE
                audio = torch.autograd.Variable(audio.to(device=self.device))
                mel = torch.autograd.Variable(mel.to(device=self.device))

                feat = self.feature_extract(audio.squeeze(1))
                feature = self.encode(feat)

                mel_hat = self.model(feature)

                # calc loss
                loss = F.mse_loss(mel_hat, mel)
                losses.append(loss.item())

                # optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # GAN
                ## Discriminator
                t_audio = next(t_data_loader)
                f_audio = next(f_data_loader)

                t_audio = torch.autograd.Variable(t_audio.to(device=self.device))
                f_audio = torch.autograd.Variable(f_audio.to(device=self.device))

                t_feat = self.feature_extract(t_audio.squeeze(1))
                t_feature = self.encode(t_feat)
                t_mel_hat = self.model(t_feature)
                t_result = self.discriminator(t_mel_hat)
                t_loss_d = F.mse_loss(t_result, torch.ones_like(t_result))

                f_feat = self.feature_extract(f_audio.squeeze(1))
                f_feature = self.encode(f_feat)
                f_mel_hat = self.model(f_feature)
                f_result = self.discriminator(f_mel_hat)
                f_loss_d = F.mse_loss(f_result, torch.zeros_like(f_result))

                gan_d_loss = t_loss_d + f_loss_d

                d_losses.append(gan_d_loss.item())

                self.optimizer_d.zero_grad()
                gan_d_loss.backward()
                self.optimizer_d.step()

                ## Generator
                t_audio = next(t_data_loader)
                f_audio = next(f_data_loader)

                t_audio = torch.autograd.Variable(t_audio.to(device=self.device))
                f_audio = torch.autograd.Variable(f_audio.to(device=self.device))

                t_feat = self.feature_extract(t_audio.squeeze(1))
                t_feature = self.encode(t_feat)
                t_mel_hat = self.model(t_feature)
                t_result = self.discriminator(t_mel_hat)
                t_loss_g = F.mse_loss(t_result, torch.ones_like(t_result))

                f_feat = self.feature_extract(f_audio.squeeze(1))
                f_feature = self.encode(f_feat)
                f_mel_hat = self.model(f_feature)
                f_result = self.discriminator(f_mel_hat)
                f_loss_g = F.mse_loss(f_result, torch.ones_like(f_result))

                gan_g_loss = t_loss_g + f_loss_g
                g_losses.append(gan_g_loss.item())

                self.optimizer_g.zero_grad()
                gan_g_loss.backward()
                self.optimizer_g.step()

                # ロギング
                if self.step % self.progress_step == 0:
                    ## console
                    current_time = self.get_time()
                    print(
                        f"[{current_time}][Step: {self.step}] loss: {sum(losses) / len(losses)}",
                    )
                    ## mel error
                    mel_error = F.l1_loss(mel, mel_hat).item()
                    ## tensorboard
                    self.log.add_scalar(
                        "train/mse_loss", sum(losses) / len(losses), self.step
                    )
                    self.log.add_scalar(
                        "train/d_loss", sum(d_losses) / len(d_losses), self.step
                    )
                    self.log.add_scalar(
                        "train/g_loss", sum(g_losses) / len(g_losses), self.step
                    )
                    self.log.add_scalar("train/mel_error", mel_error, self.step)
                    # clear loss buffer
                    losses = []
                    d_losses = []
                    g_losses = []

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
        self.model.eval()

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
                    f"[{current_time}][Step: {self.step}] Start convert test file : {filepath.name[:24]}"
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
                    # feature = self.encode(feat)[:, -6:, :]

                    if mel_history is None:
                        mel_history = feature
                    else:
                        mel_history = torch.cat([mel_history, feature], dim=1)

                    mel_hat = self.model(mel_history[:, -history_size:, :])[:, :, -6:]
                    # mel_hat = self.model(feature)[:, :, -6:]

                    if mel_hat_history is None:
                        mel_hat_history = mel_hat
                    else:
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
        self.model.train()
