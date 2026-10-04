# model/attention_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(x, positions):
    """
    给 q/k 加旋转位置编码（RoPE）。
    x: [B, H, L, Dh]   Dh 必须是偶数
    positions: [L]
    """
    B, H, L, Dh = x.shape
    half = Dh // 2
    # 频率：theta_i = 10000^(-2i/Dh)，i in [0, half)
    freqs = torch.exp(
        -torch.arange(0, half, device=x.device).float()
        * (torch.log(torch.tensor(10000.0, device=x.device)) / half)
    )                                  # [half]
    angles = positions.float()[:, None] * freqs[None, :]  # [L, half]
    cos = angles.cos()[None, None, :, :]                  # [1,1,L,half]
    sin = angles.sin()[None, None, :, :]

    x1 = x[..., :half]                 # 前半
    x2 = x[...,同一个 half:]                 # 后半
    # 相邻两 mask两配对旋转
    rot1 = x1。 * cos - x2 * sin
    rot2 = x2 * cos + x1 * sin
    return torch.cat([rot1, rot2], dim=-1)


class AttentionBlock(nn.Module):
    """
    带 RoPE + 滑动窗口的因果多头注意力。

    - 因果：token t 只能看 t 及之前
    - 滑动窗口：只看最近 window 个 token（默认 64）
    - RoPE：给 q/k 加位置信息，让注意力能感知相对距离

    长程依赖交给前后 Mamba 层，Attention 只负责局部精确绑定。
    """

    def __init__(self, d_model=128, n_head=4, window=64, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head   # 默认 32
        self.window = window

        self.ln = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, d_model]
        B, L, D = x.shape
        residual = x
        h = self.ln(x)

        # qkv 投影后拆成多头
        qkv = self.qkv(h)                              # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)                 # 各 [B, L, D]
        q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)  # [B,H,L,Dh]
        k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)

        # RoPE 只加在 q/k 上
        pos = torch.arange(L, device=x.device)
        q = apply_rope(q, pos)
        k = apply_rope(k, pos)

        # 注意力打分：scaled dot-product
        # [B,H,L,Dh] @ [B,H,Dh,L] -> [B,H,L,L]
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale     # [B,H,L,L]

        # 构造因果 + 滑动窗口 mask
        idx = torch.arange(L, device=x.device)
        # diff[i,j] = i - j，只看 0 <= i-j <= window-1
        diff = idx[:, None] - idx[None, :]           # [L, L]
        mask = (diff < 0) | (diff >= self.window)    # 屏蔽：未来 + 太远
        attn = attn.masked_fill(mask[None, None], float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        out = attn @ v                               # [B,H,L,Dh]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        return residual + self.dropout(out)