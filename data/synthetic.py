import torch


def make_copy_batch(batch_size, half_len, vocab_size, device):
    """
    Copy 任务：随机 half_len 个 token，加 sep，再复制一遍。
    token 范围 0 ~ vocab_size-2，sep = vocab_size-1
    序列长 2*half_len + 1
    只在第二段位置算 loss。
    """
    seq = torch.randint(0, vocab_size - 1, (batch_size, half_len))
    sep = torch.full((batch_size, 1), vocab_size - 1)
    x = torch.cat([seq, sep, seq], dim=1)

    y = x.clone()
    y[:, :half_len + 1] = -100
    return x.to(device), y.to(device)