import numpy as np
import torch
import torchaudio.functional as F
import pyworld


# def compute_f0(wav):
#     # freq = F.detect_pitch_frequency(
#     #     wav,
#     #     16000,
#     #     freq_low=50,
#     #     freq_high=1100,
#     #     frame_time=320 / 16000,
#     #     win_length=30,
#     # )
#     # pad_size = wav.shape[1] // 320 - freq.shape[1]
#     # freq = torch.nn.functional.pad(freq, (0, pad_size))
#
#     import pysptk
#     freq = torch.zeros(wav.shape[0], wav.shape[1] // 320).to(wav.device)
#     for i in range(wav.shape[0]):
#         f = pysptk.swipe(
#             wav[i].cpu().numpy().astype(np.float32),
#             fs=16000,
#             hopsize=320,
#             min=50,
#             max=1100,
#         )
#         freq[i] = torch.as_tensor(f).to(wav.device)[: freq.shape[1]]
#     return freq


def compute_f0(wav):
    freq = torch.zeros(wav.shape[0], wav.shape[1] // 320).to(wav.device)
    for i in range(wav.shape[0]):
        wav_numpy = wav[i].cpu().numpy().astype(np.float64)
        f0, t = pyworld.dio(
            wav_numpy,
            fs=16000,
            f0_ceil=1100,
            frame_period=1000 * 320 / 16000,
        )
        f0 = pyworld.stonemask(wav_numpy, f0, t, 16000)
        freq[i] = torch.as_tensor(f0).to(dtype=torch.float, device=wav.device)[
            : freq.shape[1]
        ]

    return freq


def normalize_f0(f0, random_scale=False):
    means = torch.mean(f0, dim=1, keepdim=True)
    if random_scale:
        factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
    else:
        factor = torch.ones(f0.shape[0], 1).to(f0.device)

    f0_norm = (f0 - means) * factor
    return f0_norm
