import os
import pathlib
import copy
from datetime import datetime

import torch
import torch.nn.functional as F
import torchaudio
from src.data.vc_data_loader import load_data
from src.model.asr_model import ASRModel
from src.model.generator import Generator
from src.model.mel_gen_model import MelGenerateModel
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

        self.step = 0
        self.start_time = datetime.now()

        self.train_loader = load_data(self.dataset_dir, self.batch_size)

        self.asr_model = ASRModel(vocab_size=32).to(self.device)
        asr_ckpt = torch.load(asr_ckpt_path, map_location=self.device)
        self.asr_model.load_state_dict(asr_ckpt["asr_model"])

        self.frozen_encoder = copy.deepcopy(self.asr_model.encoder).to(self.device)

        self.mel_gen_model = MelGenerateModel().to(self.device)

        self.vocoder = Generator().to(self.device).eval()
        vocoder_ckpt = torch.load(vocoder_ckpt_path, map_location=self.device)
        self.vocoder.load_state_dict(vocoder_ckpt["generator"])

        self.optimizer_mse = optim.AdamW(self.mel_gen_model.parameters(), lr=0.0005)

        if exp_name is not None:
            self.load_ckpt()

    def load_ckpt(self):
        """
        ckptから復元する
        """
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt-latest.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.asr_model.load_state_dict(ckpt["asr_model"])
        self.mel_gen_model.load_state_dict(ckpt["mel_gen_model"])
        self.optimizer_mse.load_state_dict(ckpt["optimizer_mse"])
        self.step = ckpt["step"]
        print(f"Load checkpoint from {ckpt_path}")

    def save_ckpt(self):
        """
        ckptを保存する
        """
        # ckpt_path = os.path.join(self.ckpt_dir, f"ckpt-{self.step:0>8}.pt")
        save_dict = {
            "asr_model": self.asr_model.state_dict(),
            "mel_gen_model": self.mel_gen_model.state_dict(),
            "optimizer_mse": self.optimizer_mse.state_dict(),
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

        mel_losses = []

        while self.step < self.max_step:
            for audio, mel in self.train_loader:
                # mel MSE loss
                audio = torch.autograd.Variable(audio.to(device=self.device))
                mel = torch.autograd.Variable(mel.to(device=self.device))

                feat = self.asr_model.feature_extractor(audio.squeeze(1))
                feature = self.asr_model.encoder(feat)
                mel_hat = self.mel_gen_model(feature)

                mel_loss = F.mse_loss(mel_hat, mel)
                mel_losses.append(mel_loss.item())

                self.optimizer_mse.zero_grad()
                mel_loss.backward()
                self.optimizer_mse.step()

                # ロギング
                if self.step % self.progress_step == 0:
                    ## console
                    current_time = self.get_time()
                    print(
                        f"[{current_time}][Step: {self.step}] mel loss: {sum(mel_losses) / len(mel_losses)}",
                    )
                    ## mel error
                    mel_error = F.l1_loss(mel, mel_hat).item()
                    ## tensorboard
                    self.log.add_scalar(
                        "train/mel_loss", sum(mel_losses) / len(mel_losses), self.step
                    )
                    self.log.add_scalar("train/mel_error", mel_error, self.step)
                    # clear loss buffer
                    mel_losses = []

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
        self.asr_model.eval()
        self.mel_gen_model.eval()

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

                    feat = self.asr_model.feature_extractor(audio)

                    if feat_history is None:
                        feat_history = feat
                    else:
                        feat_history = torch.cat([feat_history, feat], dim=1)

                    feature = self.asr_model.encoder(
                        feat_history[:, -history_size:, :]
                    )[:, -6:, :]

                    if mel_history is None:
                        mel_history = feature
                    else:
                        mel_history = torch.cat([mel_history, feature], dim=1)

                    mel_hat = self.mel_gen_model(mel_history[:, -history_size:, :])[
                        :, :, -6:
                    ]

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
        self.asr_model.train()
        self.mel_gen_model.train()
