import torch


def log_melspectrogram(x, clip_val=1e-5):
    """
    メルスペクトログラムのlogを計算する関数
    """
    return torch.log(torch.clamp(x, min=clip_val))
