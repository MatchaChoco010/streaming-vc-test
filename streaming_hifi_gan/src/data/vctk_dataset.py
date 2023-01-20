import pathlib
import random

import torch
import torch.nn.functional as F
import torchaudio
from src.module.log_melspectrogram import log_melspectrogram
from torch.utils.data import IterableDataset

SEGMENT_SIZE = 8192
VALIDATION_FILENAMES = [
    "p249_083.flac",
    "p308_201.flac",
    "p261_166.flac",
    "p374_284.flac",
    "p376_163.flac",
    "p233_009.flac",
    "p264_372.flac",
    "p246_113.flac",
    "p281_282.flac",
    "p255_113.flac",
    "p261_336.flac",
    "p272_298.flac",
    "p230_354.flac",
    "p329_242.flac",
    "p273_386.flac",
    "p261_020.flac",
    "p273_419.flac",
    "p274_356.flac",
    "p247_071.flac",
    "p314_325.flac",
    "p286_405.flac",
    "p268_178.flac",
    "p269_361.flac",
    "p252_162.flac",
    "p269_377.flac",
    "p244_295.flac",
    "p238_375.flac",
    "p279_362.flac",
    "p376_205.flac",
    "p343_131.flac",
    "p264_476.flac",
    "p247_051.flac",
    "p229_386.flac",
    "p236_460.flac",
    "p314_108.flac",
    "p305_353.flac",
    "p347_067.flac",
    "p239_134.flac",
    "p361_249.flac",
    "p281_302.flac",
    "p351_123.flac",
    "p347_282.flac",
    "p279_129.flac",
    "p361_286.flac",
    "p376_175.flac",
    "p251_339.flac",
    "p247_401.flac",
    "p226_073.flac",
    "p280_118.flac",
    "p279_075.flac",
    "p340_186.flac",
    "p279_379.flac",
    "p271_300.flac",
    "p273_283.flac",
    "p248_322.flac",
    "p231_462.flac",
    "p310_110.flac",
    "p257_159.flac",
    "p231_045.flac",
    "p247_200.flac",
    "p262_235.flac",
    "p364_112.flac",
    "p351_049.flac",
    "p306_171.flac",
    "p286_236.flac",
    "p340_028.flac",
    "p248_328.flac",
    "p330_357.flac",
    "p266_136.flac",
    "p254_298.flac",
    "p286_324.flac",
    "p228_346.flac",
    "p259_345.flac",
    "p250_292.flac",
    "p248_027.flac",
    "p264_217.flac",
    "p273_291.flac",
    "p261_100.flac",
    "p336_186.flac",
    "p285_128.flac",
    "p336_252.flac",
    "p298_116.flac",
    "p312_137.flac",
    "p345_233.flac",
    "p307_151.flac",
    "p266_004.flac",
    "p364_145.flac",
    "p231_208.flac",
    "p241_227.flac",
    "p256_032.flac",
    "p311_329.flac",
    "p239_257.flac",
    "p376_278.flac",
    "p351_268.flac",
    "p234_236.flac",
    "p341_278.flac",
    "p360_054.flac",
    "p263_245.flac",
    "p227_111.flac",
    "p317_386.flac",
    "p250_266.flac",
    "p292_213.flac",
    "p256_099.flac",
    "p312_110.flac",
    "p315_228.flac",
    "p308_095.flac",
    "p268_151.flac",
    "p286_130.flac",
    "p240_339.flac",
    "p229_232.flac",
    "p351_166.flac",
    "p264_202.flac",
    "p234_321.flac",
    "p249_086.flac",
    "p318_254.flac",
    "p245_157.flac",
    "p270_194.flac",
    "p262_244.flac",
    "p273_208.flac",
    "p294_195.flac",
    "p334_355.flac",
    "p351_391.flac",
    "p279_340.flac",
    "p234_297.flac",
    "p247_350.flac",
    "p254_337.flac",
    "p362_044.flac",
    "p259_276.flac",
    "p278_334.flac",
    "p302_305.flac",
    "p336_153.flac",
    "p311_049.flac",
    "p317_295.flac",
    "p246_355.flac",
    "p293_216.flac",
    "p298_217.flac",
    "p229_352.flac",
    "p330_102.flac",
    "p283_421.flac",
    "p281_041.flac",
    "p318_236.flac",
    "p362_134.flac",
    "p345_347.flac",
    "p243_021.flac",
    "p282_013.flac",
    "p287_300.flac",
    "p343_240.flac",
    "p333_123.flac",
    "p259_153.flac",
    "p229_322.flac",
]


class VCTKDataset(IterableDataset):
    """
    VCTKのデータセットを扱うクラス
    """

    def __init__(self, train=True):
        self.path = "dataset/silence-removed/"

        file_list = list(
            pathlib.Path(self.path).rglob(
                "*.flac",
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
