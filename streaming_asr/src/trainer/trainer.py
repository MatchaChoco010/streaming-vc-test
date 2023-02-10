import os
import math
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from jiwer import cer
from src.data.data_loader import load_data
from src.model.asr import ASR
from torch import optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    ASRをトレーニングするクラス。
    """

    def __init__(
        self,
        batch_size: int = 4,
        ctc_weight: float = 0.5,
        accumulation_steps: int = 4,
        max_step: int = 10000001,
        progress_step: int = 10,
        valid_step: int = 5000,
        exp_name: str | None = None,
        ckpt_dir: str = "output/asr/ckpt",
        log_dir: str = "output/asr/log",
    ):
        """
        Arguments:
            exp_name: str
                再開する実験名。Noneの場合は新しい実験が生成される。
        """
        self.batch_size = batch_size
        self.ctc_weight = ctc_weight
        self.accumulation_steps = accumulation_steps
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

        self.best_cer = {"att": 3.0, "ctc": 3.0}

        (
            self.train_loader,
            self.dev_loader,
            self.vocab_size,
            self.tokenizer,
        ) = load_data(self.batch_size)

        self.model = ASR(self.vocab_size).to(self.device)

        self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)

        self.optimizer = optim.Adadelta(self.model.parameters())

        if exp_name is not None:
            self.load_ckpt()

    def fetch_data(
        self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        データセットを1ステップ分取得する

        Arguments:
            data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                データのタプル
        Returns:
            (batch_size, audio, audio_len, text, text_len):
                Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

            batch_size: int
                バッチサイズ
            audio: torch.Tensor (batch, max(audio_len))
                音声特徴量
            audio_len: torch.Tensor (batch)
                音声特徴量の各バッチの長さ
            text: torch.Tensor (batch, max(text_len), vocab_size)
                エンコードされたテキストone hot表現
            text_len: torch.Tensor (batch)
                テキストの各バッチの長さ
        """
        audio_feature, audio_len, text = data
        audio_feature = Variable(audio_feature, requires_grad=False).to(self.device)
        audio_len = Variable(audio_len, requires_grad=False).to(self.device)
        assert audio_feature.shape[1] >= audio_len.max()
        text = Variable(text, requires_grad=False).to(self.device)
        text_len = Variable(
            torch.sum(text.argmax(dim=-1) != 0, dim=-1), requires_grad=False
        )
        assert text.shape[1] == text_len.max()
        batch_size = audio_feature.shape[0]
        return batch_size, audio_feature, audio_len, text, text_len

    def load_ckpt(self):
        """
        ckptから復元する
        """
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt-latest.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        self.best_cer = ckpt["best_cer"]
        print(f"Load checkpoint from {ckpt_path}")

    def save_ckpt(self, best: bool = False, best_task: str = "att"):
        """
        ckptを保存する

        Arguments:
            best: bool
                ベストスコアかどうか
            best_task: str
                ベストスコアのタスクの種類
        """
        if best:
            ckpt_path = os.path.join(self.ckpt_dir, f"best-{best_task}.pt")
            save_dict = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "best_cer": self.best_cer,
            }
            torch.save(save_dict, ckpt_path)
        else:
            ckpt_path = os.path.join(self.ckpt_dir, f"ckpt-{self.step:0>8}.pt")
            save_dict = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "best_cer": self.best_cer,
            }
            # torch.save(save_dict, ckpt_path)

            ckpt_path = os.path.join(self.ckpt_dir, f"ckpt-latest.pt")
            torch.save(save_dict, ckpt_path)

    def calc_error_rate(
        self, pred: torch.Tensor, truth: torch.Tensor, ignore_repeat=False
    ) -> float:
        """
        cerを計算する

        Arguments:
            pred: torch.Tensor
                予測結果のエンコードされたテキスト
            truth: torch.Tensor
                正解のエンコードされたテキスト
        """
        t = [self.tokenizer.decode(t) for t in truth.argmax(dim=-1).tolist()]
        p = [
            self.tokenizer.decode(t, ignore_repeat=ignore_repeat)
            for t in pred.argmax(dim=-1).tolist()
        ]
        return cer(t, p)

    def progress(
        self,
        msg: str,
        gt_text: str,
        ctc_text: str,
        att_text: str,
        overwrite: bool = True,
    ):
        """
        progressを表示する

        Arguments:
            msg: str
                メッセージ
            gt_text: str
                正解のテキスト
            ctc_text: str
                CTCの予測結果のテキスト
            att_text: str
                Attentionの予測結果のテキスト
            overwrite: bool
                前のprogressを上書きするかどうか
        """
        sys.stdout.write("\033[1A")  # Line Up
        sys.stdout.write("\033[K")  # Clear line
        sys.stdout.write("\033[1A")  # Line Up
        sys.stdout.write("\033[K")  # Clear line
        sys.stdout.write("\033[1A")  # Line Up
        sys.stdout.write("\033[K")  # Clear line
        if overwrite:
            sys.stdout.write("\033[1A")  # Line Up
            sys.stdout.write("\033[K")  # Clear line
        current_time = self.get_time()
        print(
            f"[{current_time}][Step: {self.step}] {msg}",
        )
        print(
            "[GT ]",
            gt_text[:75] + "..." if len(gt_text) > 75 else gt_text,
        )
        print(
            "[CTC]",
            ctc_text[:75] + "..." if len(ctc_text) > 75 else ctc_text,
        )
        print(
            "[ATT]",
            att_text[:75] + "..." if len(att_text) > 75 else att_text,
        )

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
        for k in self.model.state_dict().keys():
            print(f"\t{k}")
        print("\n\n\n\n")

        ctc_loss, att_loss = None, None
        self.start_time = datetime.now()
        self.optimizer.zero_grad()

        ctc_losses = []
        att_losses = []
        total_losses = []
        ctc_cers = []
        att_cers = []

        while self.step < self.max_step:
            # Gradient Accumulationを行う
            for data in self.train_loader:
                _, audio_feature, audio_len, text, text_len = self.fetch_data(data)

                encode_len, ctc_output, att_output = self.model(
                    audio_feature,
                    audio_len,
                    teacher=text,
                    teacher_lengths=text_len,
                )
                ctc_output_log_softmax = F.log_softmax(ctc_output, dim=-1).transpose(
                    0, 1
                )

                # calc loss
                ctc_loss = self.ctc_loss(
                    ctc_output_log_softmax,
                    text.argmax(dim=-1),
                    encode_len,
                    text_len,
                )
                ctc_losses.append(ctc_loss.item())

                b, t, _ = att_output.shape
                att_loss = self.seq_loss(
                    att_output.view(b * t, -1), text.argmax(dim=-1).view(-1)
                )
                att_losses.append(att_loss.item())

                total_loss = ctc_loss * self.ctc_weight + att_loss * (
                    1 - self.ctc_weight
                )
                total_losses.append(total_loss.item())

                loss = total_loss / self.accumulation_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4)

                # cerの計算
                ctc_cers.append(
                    self.calc_error_rate(ctc_output, text, ignore_repeat=True)
                )
                att_cers.append(self.calc_error_rate(att_output, text))

                # optimizer
                if self.step % self.accumulation_steps == self.accumulation_steps - 1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # ロギング
                if self.step % self.progress_step == 0:
                    # calculate mean of losses and cers in progress steps
                    ctc_loss = sum(ctc_losses) / len(ctc_losses)
                    att_loss = sum(att_losses) / len(att_losses)
                    total_loss = sum(total_losses) / len(total_losses)
                    cer_ctc = sum(ctc_cers) / len(ctc_cers)
                    cer_att = sum(att_cers) / len(att_cers)
                    ## console
                    gt_text = self.tokenizer.decode(text[0].argmax(dim=-1).tolist())
                    ctc_text = self.tokenizer.decode(
                        ctc_output[0].argmax(dim=-1).tolist(), ignore_repeat=True
                    )
                    att_text = self.tokenizer.decode(
                        att_output[0].argmax(dim=-1).tolist()
                    )
                    self.progress(
                        f"CTC Loss: {ctc_loss:.3f}, ATT Loss: {att_loss:.3f}, Total Loss: {total_loss:.3f}",
                        gt_text,
                        ctc_text,
                        att_text,
                        overwrite=False,
                    )
                    ## tensorboard
                    self.log.add_scalar("train/loss/ctc", ctc_loss, self.step)
                    self.log.add_scalar("train/loss/att", att_loss, self.step)
                    self.log.add_scalar("train/loss/total", total_loss, self.step)
                    self.log.add_scalar("train/cer/ctc", cer_ctc, self.step)
                    self.log.add_scalar("train/cer/att", cer_att, self.step)
                    # reset losses
                    ctc_losses = []
                    att_losses = []
                    total_losses = []
                    ctc_cers = []
                    att_cers = []

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
        self.model.eval()
        dev_cer: Dict[str, List[float]] = {"att": [], "ctc": []}
        losses: Dict[str, List[float]] = {"att": [], "ctc": [], "total": []}

        for i, data in enumerate(self.dev_loader):
            _, audio_feature, audio_len, text, text_len = self.fetch_data(data)

            with torch.no_grad():
                encode_len, ctc_output, att_output = self.model.forward_test(
                    audio_feature,
                    audio_len,
                    max(text_len),
                )
                ctc_output_log_softmax = F.log_softmax(ctc_output, dim=-1).transpose(
                    0, 1
                )
                # calc loss
                ctc_loss = self.ctc_loss(
                    ctc_output_log_softmax, text.argmax(dim=-1), encode_len, text_len
                )

                b, t, _ = att_output.shape
                att_loss = self.seq_loss(
                    att_output.view(b * t, -1), text.argmax(dim=-1).view(-1)
                )

                total_loss = ctc_loss * self.ctc_weight + att_loss * (
                    1 - self.ctc_weight
                )

            gt_text = self.tokenizer.decode(text[0].argmax(dim=-1).tolist())
            ctc_text = self.tokenizer.decode(
                ctc_output[0].argmax(dim=-1).tolist(), ignore_repeat=True
            )
            att_text = self.tokenizer.decode(att_output[0].argmax(dim=-1).tolist())
            self.progress(
                f"Valid step - {i+1}/{math.ceil(2400 / self.batch_size)}",
                gt_text,
                ctc_text,
                att_text,
                overwrite=i != 0,
            )

            dev_cer["ctc"].append(
                self.calc_error_rate(ctc_output, text, ignore_repeat=True)
            )
            dev_cer["att"].append(self.calc_error_rate(att_output, text))
            losses["ctc"].append(ctc_loss.item())
            losses["att"].append(att_loss.item())
            losses["total"].append(total_loss.item())

            # いくつかの例をtensorboardに表示
            if i == math.ceil(2400 / self.batch_size) // 2:
                for j in range(min(len(text), 20)):
                    if self.step == 0:
                        self.log.add_text(
                            f"gt/text-{j}",
                            self.tokenizer.decode(text[j].argmax(dim=-1).tolist()),
                            global_step=self.step,
                        )
                    self.log.add_text(
                        f"ctc/text-{j}",
                        self.tokenizer.decode(
                            ctc_output[j].argmax(dim=-1).tolist(), ignore_repeat=True
                        ),
                        global_step=self.step,
                    )
                    self.log.add_text(
                        f"att/text-{j}",
                        self.tokenizer.decode(att_output[j].argmax(dim=-1).tolist()),
                        global_step=self.step,
                    )

            # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
            torch.cuda.empty_cache()

        self.log.add_scalar(
            "valid/loss/ctc", sum(losses["ctc"]) / len(losses["ctc"]), self.step
        )
        self.log.add_scalar(
            "valid/loss/att", sum(losses["att"]) / len(losses["att"]), self.step
        )
        self.log.add_scalar(
            "valid/loss/total", sum(losses["total"]) / len(losses["total"]), self.step
        )

        # 現在のckptをlatestとして保存
        self.save_ckpt()

        # それぞれの指標で良くなっていたら保存
        for task in ["att", "ctc"]:
            cer = sum(dev_cer[task]) / len(dev_cer[task])
            if cer < self.best_cer[task]:
                self.best_cer[task] = cer
                self.save_ckpt(best=True, best_task=task)
            self.log.add_scalar(
                f"valid/cer/{task}",
                cer,
                global_step=self.step,
            )

        # Resume training
        self.model.train()
