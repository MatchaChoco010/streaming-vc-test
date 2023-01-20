import pathlib
import random

import torch
import torch.nn.functional as F
import torchaudio
from src.module.log_melspectrogram import log_melspectrogram
from torch.utils.data import IterableDataset

SEGMENT_SIZE = 8192
VALIDATION_FILENAMES = [
    "p249_083.wav",
    "p308_201.wav",
    "p261_166.wav",
    "p374_284.wav",
    "p376_163.wav",
    "p233_009.wav",
    "p264_372.wav",
    "p246_113.wav",
    "p281_282.wav",
    "p255_113.wav",
    "p261_336.wav",
    "p272_298.wav",
    "p230_354.wav",
    "p329_242.wav",
    "p273_386.wav",
    "p261_020.wav",
    "p273_419.wav",
    "p274_356.wav",
    "p247_071.wav",
    "p314_325.wav",
    "p286_405.wav",
    "p268_178.wav",
    "p269_361.wav",
    "p252_162.wav",
    "p269_377.wav",
    "p244_295.wav",
    "p238_375.wav",
    "p279_362.wav",
    "p376_205.wav",
    "p343_131.wav",
    "p264_476.wav",
    "p247_051.wav",
    "p229_386.wav",
    "p236_460.wav",
    "p314_108.wav",
    "p305_353.wav",
    "p347_067.wav",
    "p239_134.wav",
    "p361_249.wav",
    "p281_302.wav",
    "p351_123.wav",
    "p347_282.wav",
    "p279_129.wav",
    "p361_286.wav",
    "p376_175.wav",
    "p251_339.wav",
    "p247_401.wav",
    "p226_073.wav",
    "p280_118.wav",
    "p279_075.wav",
    "p340_186.wav",
    "p279_379.wav",
    "p271_300.wav",
    "p273_283.wav",
    "p248_322.wav",
    "p231_462.wav",
    "p310_110.wav",
    "p257_159.wav",
    "p231_045.wav",
    "p247_200.wav",
    "p262_235.wav",
    "p364_112.wav",
    "p351_049.wav",
    "p306_171.wav",
    "p286_236.wav",
    "p340_028.wav",
    "p248_328.wav",
    "p330_357.wav",
    "p266_136.wav",
    "p254_298.wav",
    "p286_324.wav",
    "p228_346.wav",
    "p259_345.wav",
    "p250_292.wav",
    "p248_027.wav",
    "p264_217.wav",
    "p273_291.wav",
    "p261_100.wav",
    "p336_186.wav",
    "p285_128.wav",
    "p336_252.wav",
    "p298_116.wav",
    "p312_137.wav",
    "p345_233.wav",
    "p307_151.wav",
    "p266_004.wav",
    "p364_145.wav",
    "p231_208.wav",
    "p241_227.wav",
    "p256_032.wav",
    "p311_329.wav",
    "p239_257.wav",
    "p376_278.wav",
    "p351_268.wav",
    "p234_236.wav",
    "p341_278.wav",
    "p360_054.wav",
    "p263_245.wav",
    "p227_111.wav",
    "p317_386.wav",
    "p250_266.wav",
    "p292_213.wav",
    "p256_099.wav",
    "p312_110.wav",
    "p315_228.wav",
    "p308_095.wav",
    "p268_151.wav",
    "p286_130.wav",
    "p240_339.wav",
    "p229_232.wav",
    "p351_166.wav",
    "p264_202.wav",
    "p234_321.wav",
    "p249_086.wav",
    "p318_254.wav",
    "p245_157.wav",
    "p270_194.wav",
    "p262_244.wav",
    "p273_208.wav",
    "p294_195.wav",
    "p334_355.wav",
    "p351_391.wav",
    "p279_340.wav",
    "p234_297.wav",
    "p247_350.wav",
    "p254_337.wav",
    "p362_044.wav",
    "p259_276.wav",
    "p278_334.wav",
    "p302_305.wav",
    "p336_153.wav",
    "p311_049.wav",
    "p317_295.wav",
    "p246_355.wav",
    "p293_216.wav",
    "p298_217.wav",
    "p229_352.wav",
    "p330_102.wav",
    "p283_421.wav",
    "p281_041.wav",
    "p318_236.wav",
    "p362_134.wav",
    "p345_347.wav",
    "p243_021.wav",
    "p282_013.wav",
    "p287_300.wav",
    "p343_240.wav",
    "p333_123.wav",
    "p259_153.wav",
    "p229_322.wav",
]


class VCTKDataset(IterableDataset):
    """
    VCTKのデータセットを扱うクラス
    """

    def __init__(self, train=True):
        self.path = "dataset/silence-removed/"

        file_list = list(
            pathlib.Path(self.path).rglob(
                "*.wav",
            )
        )
        self.file_list = []
        for f in file_list:
            if train and f.name in VALIDATION_FILENAMES:
                continue
            elif not train and not (f.name in VALIDATION_FILENAMES):
                continue
            self.file_list.append(f)

    def __iter__(self):
        for audio_filename in self.file_list:
            audio, sr = torchaudio.load(audio_filename)

            # 適当にずらしてSEGMENT_SIZEで刻んでいく
            audio_start = random.randint(0, SEGMENT_SIZE)
            for start in range(audio_start, audio.shape[1], SEGMENT_SIZE):
                audio = audio[:, start : start + SEGMENT_SIZE]

                if audio.shape[1] < SEGMENT_SIZE:
                    audio = F.pad(audio, (0, SEGMENT_SIZE - audio.shape[1]), "constant")

                mel = torchaudio.transforms.MelSpectrogram(
                    n_fft=1024,
                    n_mels=80,
                    sample_rate=24000,
                    hop_length=256,
                    win_length=1024,
                )(audio)[:, :, : SEGMENT_SIZE // 256]
                mel = log_melspectrogram(mel).squeeze(0)

                yield audio, mel
