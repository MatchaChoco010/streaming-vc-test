import json
import pathlib
import random

import torch
import torch.nn.functional as F
import torchvision.transforms
import torchvision.transforms.functional
import torchaudio
from torch.utils.data import DataLoader, IterableDataset
from src.hifigan.models import Generator

AUDIO_LENGTH = 16000 * 2


mel_basis = {}
hann_window = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = torchaudio.functional.melscale_fbanks(
            sample_rate=sampling_rate,
            n_freqs=1 + n_fft // 2,
            n_mels=num_mels,
            f_min=fmin,
            f_max=fmax,
            norm="slaney",
            mel_scale="slaney",
        ).transpose(0, 1)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (512, 512), mode="reflect")
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class VoiceDataset(IterableDataset):
    def __init__(self, voice_data_dir, min_scale: int, max_scale: int):
        self.file_list = [
            str(item) for item in pathlib.Path(voice_data_dir).rglob("*.wav")
        ]
        self.min_scale = min_scale
        self.max_scale = max_scale

        with open("src/hifigan/config.json", "r") as f:
            config = json.load(f)

        config = AttrDict(config)
        vocoder = Generator(config)
        ckpt = torch.load("src/hifigan/generator_v3")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        self.vocoder = vocoder.cuda()

    def __iter__(self):
        for item in self.file_list:
            audio, sr = torchaudio.load(item)
            audio = torchaudio.functional.resample(audio, sr, 16000)
            if audio.abs().max() >= 1.0:
                audio = audio / audio.abs().max()

            for scale in range(self.min_scale, self.max_scale + 1):

                start = random.randint(0, max(0, audio.shape[1] - AUDIO_LENGTH))
                clip_audio = audio[:, start : start + AUDIO_LENGTH]
                if clip_audio.shape[1] < AUDIO_LENGTH:
                    clip_audio = F.pad(
                        clip_audio,
                        (0, AUDIO_LENGTH - clip_audio.shape[1]),
                        "constant",
                    )

                aug_audio = torchaudio.functional.resample(
                    clip_audio.cuda(), 16000, 22050
                )
                if aug_audio.abs().max() >= 1.0:
                    aug_audio = aug_audio / aug_audio.abs().max()

                mel = mel_spectrogram(aug_audio, 1024, 80, 22050, 256, 1024, 0, 8000)

                # 縦方向にリサイズする
                height = scale
                mel = torchvision.transforms.functional.resize(
                    img=mel, size=(height, mel.shape[2])
                )
                if scale < 80:
                    mel = torchvision.transforms.Pad(
                        padding=(0, 0, 0, 80 - mel.shape[1]), padding_mode="edge"
                    )(mel)
                else:
                    mel = mel[:, :80, :]

                with torch.no_grad():
                    aug_audio = self.vocoder(mel)
                    aug_audio = torchaudio.functional.resample(aug_audio, 22050, 16000)
                    aug_audio = aug_audio.squeeze().cpu()

                yield clip_audio, aug_audio
                # yield clip_audio, torch.nn.functional.pad(
                #     clip_audio.squeeze(), (0, 256)
                # )


class ShuffleDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass


def load_data(
    voice_data_dir: str,
    batch_size: int,
    min_scale: int,
    max_scale: int,
) -> DataLoader:
    voice_data_loader = DataLoader(
        ShuffleDataset(VoiceDataset(voice_data_dir, min_scale, max_scale), 1024),
        batch_size=batch_size,
        drop_last=False,
        pin_memory=True,
    )

    return voice_data_loader
