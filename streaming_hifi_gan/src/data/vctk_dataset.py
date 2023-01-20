import pathlib
import random

import torch
import torch.nn.functional as F
import torchaudio
from src.module.log_melspectrogram import log_melspectrogram
from torch.utils.data import IterableDataset

SEGMENT_SIZE = 8192
VALIDATION_FILENAMES = [
    "p249_083-0.wav",
    "p308_201-0.wav",
    "p261_166-0.wav",
    "p374_284-0.wav",
    "p376_163-0.wav",
    "p233_009-0.wav",
    "p264_372-0.wav",
    "p246_113-0.wav",
    "p281_282-0.wav",
    "p255_113-0.wav",
    "p261_336-0.wav",
    "p272_298-0.wav",
    "p230_354-0.wav",
    "p329_242-0.wav",
    "p273_386-0.wav",
    "p261_020-0.wav",
    "p273_419-0.wav",
    "p274_356-0.wav",
    "p247_071-0.wav",
    "p314_325-0.wav",
    "p286_405-0.wav",
    "p268_178-0.wav",
    "p269_361-0.wav",
    "p252_162-0.wav",
    "p269_377-0.wav",
    "p244_295-0.wav",
    "p238_375-0.wav",
    "p279_362-0.wav",
    "p376_205-0.wav",
    "p343_131-0.wav",
    "p264_476-0.wav",
    "p247_051-0.wav",
    "p229_386-0.wav",
    "p236_460-0.wav",
    "p314_108-0.wav",
    "p305_353-0.wav",
    "p347_067-0.wav",
    "p239_134-0.wav",
    "p361_249-0.wav",
    "p281_302-0.wav",
    "p351_123-0.wav",
    "p347_282-0.wav",
    "p279_129-0.wav",
    "p361_286-0.wav",
    "p376_175-0.wav",
    "p251_339-0.wav",
    "p247_401-0.wav",
    "p226_073-0.wav",
    "p280_118-0.wav",
    "p279_075-0.wav",
    "p340_186-0.wav",
    "p279_379-0.wav",
    "p271_300-0.wav",
    "p273_283-0.wav",
    "p248_322-0.wav",
    "p231_462-0.wav",
    "p310_110-0.wav",
    "p257_159-0.wav",
    "p231_045-0.wav",
    "p247_200-0.wav",
    "p262_235-0.wav",
    "p364_112-0.wav",
    "p351_049-0.wav",
    "p306_171-0.wav",
    "p286_236-0.wav",
    "p340_028-0.wav",
    "p248_328-0.wav",
    "p330_357-0.wav",
    "p266_136-0.wav",
    "p254_298-0.wav",
    "p286_324-0.wav",
    "p228_346-0.wav",
    "p259_345-0.wav",
    "p250_292-0.wav",
    "p248_027-0.wav",
    "p264_217-0.wav",
    "p273_291-0.wav",
    "p261_100-0.wav",
    "p336_186-0.wav",
    "p285_128-0.wav",
    "p336_252-0.wav",
    "p298_116-0.wav",
    "p312_137-0.wav",
    "p345_233-0.wav",
    "p307_151-0.wav",
    "p266_004-0.wav",
    "p364_145-0.wav",
    "p231_208-0.wav",
    "p241_227-0.wav",
    "p256_032-0.wav",
    "p311_329-0.wav",
    "p239_257-0.wav",
    "p376_278-0.wav",
    "p351_268-0.wav",
    "p234_236-0.wav",
    "p341_278-0.wav",
    "p360_054-0.wav",
    "p263_245-0.wav",
    "p227_111-0.wav",
    "p317_386-0.wav",
    "p250_266-0.wav",
    "p292_213-0.wav",
    "p256_099-0.wav",
    "p312_110-0.wav",
    "p315_228-0.wav",
    "p308_095-0.wav",
    "p268_151-0.wav",
    "p286_130-0.wav",
    "p240_339-0.wav",
    "p229_232-0.wav",
    "p351_166-0.wav",
    "p264_202-0.wav",
    "p234_321-0.wav",
    "p249_086-0.wav",
    "p318_254-0.wav",
    "p245_157-0.wav",
    "p270_194-0.wav",
    "p262_244-0.wav",
    "p273_208-0.wav",
    "p294_195-0.wav",
    "p334_355-0.wav",
    "p351_391-0.wav",
    "p279_340-0.wav",
    "p234_297-0.wav",
    "p247_350-0.wav",
    "p254_337-0.wav",
    "p362_044-0.wav",
    "p259_276-0.wav",
    "p278_334-0.wav",
    "p302_305-0.wav",
    "p336_153-0.wav",
    "p311_049-0.wav",
    "p317_295-0.wav",
    "p246_355-0.wav",
    "p293_216-0.wav",
    "p298_217-0.wav",
    "p229_352-0.wav",
    "p330_102-0.wav",
    "p283_421-0.wav",
    "p281_041-0.wav",
    "p318_236-0.wav",
    "p362_134-0.wav",
    "p345_347-0.wav",
    "p243_021-0.wav",
    "p282_013-0.wav",
    "p287_300-0.wav",
    "p343_240-0.wav",
    "p333_123-0.wav",
    "p259_153-0.wav",
    "p229_322-0.wav",
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
            audio, sr = torchaudio.load(str(audio_filename))

            # 適当にずらしてSEGMENT_SIZEで刻んでいく
            audio_start = random.randint(0, SEGMENT_SIZE)
            for start in range(audio_start, audio.shape[1], SEGMENT_SIZE):
                clip_audio = audio[:, start : start + SEGMENT_SIZE]

                if clip_audio.shape[1] < SEGMENT_SIZE:
                    clip_audio = F.pad(
                        clip_audio, (0, SEGMENT_SIZE - clip_audio.shape[1]), "constant"
                    )

                mel = torchaudio.transforms.MelSpectrogram(
                    n_fft=1024,
                    n_mels=80,
                    sample_rate=24000,
                    hop_length=256,
                    win_length=1024,
                )(clip_audio)[:, :, : SEGMENT_SIZE // 256]
                mel = log_melspectrogram(mel).squeeze(0)

                yield clip_audio, mel
