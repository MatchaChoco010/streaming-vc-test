import numpy as np
import torch
import torchaudio.functional as F
import pyworld


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
