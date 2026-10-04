%%writefile /content/mini-cog/model/hybrid.py
import torch
import torch.nn as nn

from .mamba_block import MambaBlock
from .attention_block import AttentionBlock


class HybridLM(nn.Module):
    """
    SSM + Attention 混合语言模型。

    层序（默认 n_layer=7, attn_layer=3）：
        SSM SSM SSM  Attention  SSM SSM SSM
    第 4 层（0-indexed 3）放 Attention，其余放 SSM。

    Mamba 负责流式、长程、递归状态；
    Attention 负责局部窗口内的精确绑定；
    两者叠加，既能流式也能精确检索。
    """

    def __init__(
        self,
        vocab_size=64,
        d_model=128,
        n_head=4,
        window=64,
        n_layer=7,
        attn_layer=3,
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # 位置嵌入只为给 SSM 一个位置提示；Attention 内部靠 RoPE
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        layers = []
        for i in range(n_layer):
            if i == attn_layer:
                layers.append(AttentionBlock(d_model, n_head, window, dropout))
            else:
                layers.append(MambaBlock(d_model, dropout=dropout))
        self.layers = nn.ModuleList(layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, x):
        # x: [B, L] 整数 token
        B, L = x.shape
        assert L <= self.max_len, f"序列长 {L} 超过 max_len {self.max_len}"

        pos = torch.arange(L, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)[None, :, :]
        h = self.drop(h)

        for layer in self.layers:
            h = layer(h)

        return self.head(self.ln_f(h))   # [B, L, vocab_size]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成，仅用于验证推理。
        idx: [B, T0] 起始 token
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx