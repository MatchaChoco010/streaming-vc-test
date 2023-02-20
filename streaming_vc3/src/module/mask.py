import torch


def length_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """
    各バッチの長さに対するマスクを作る関数

    Arguments:
        lengths: Tensor (batch)
            各バッチの長さのTensor
        max_len: int | None
    Returns:
        mask: Tensor (batch, 1, seq_size)
    """
    batch_size = lengths.shape[0]

    if max_len is None:
        m = lengths.max().item()
        if isinstance(m, int):
            max_length = m
        else:
            raise ValueError(f"lengths.max() is not int: {m}")
    else:
        assert max_len >= lengths.max().item()
        max_length = max_len

    seq_range = torch.arange(0, max_length, dtype=torch.long)
    seq_range = seq_range.unsqueeze(0).expand(batch_size, max_length).to(lengths.device)
    lengths_expand = (
        lengths.unsqueeze(1).expand(batch_size, max_length).to(lengths.device)
    )
    mask = seq_range < lengths_expand

    assert mask.shape == (batch_size, max_length)
    return mask.unsqueeze(1)


def seq_mask(seq_size: int) -> torch.Tensor:
    """
    後ろを参照しないためのマスクを作る関数

    Arguments:
        seq_size: int
            マスクのサイズ
    Returns:
        mask: Tensor (1, seq_size, seq_size)
    """
    ret = torch.ones(seq_size, seq_size, dtype=torch.bool)
    return torch.tril(ret, out=ret).unsqueeze(0)


def chunk_mask(seq_size: int, chunk_size: int) -> torch.Tensor:
    """
    chunk_sizeごとにマスクを作る関数

    Arguments:
        seq_size: int
            マスクのサイズ
        chunk_size: int
            chunk_size
    Returns:
        mask: Tensor (1, seq_size, seq_size)
    """
    a = torch.matmul(
        torch.ones((seq_size, 1), dtype=torch.long), torch.arange(seq_size).unsqueeze(0)
    )
    b = torch.matmul(
        torch.arange(seq_size).unsqueeze(1), torch.ones((1, seq_size), dtype=torch.long)
    )
    ret = (a // chunk_size) <= (b // chunk_size)
    return ret.unsqueeze(0)
