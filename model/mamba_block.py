# model/mamba_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """
    简化版选择性状态空间块（Mamba-like）。
    """

    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv

        self.ln = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)

        A = torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        residual = x
        x = self.ln(x)

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv1d(x_ssm)[..., :L]
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = F.silu(x_ssm)

        BC = self.x_proj(x_ssm)
        Bp, Cp = BC.chunk(2, dim=-1)

        dt = F.softplus(self.dt_proj(x_ssm))
        A = -torch.exp(self.A_log)

        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * Bp.unsqueeze(2)

        h = torch.zeros(B, self.d_inner, self.d_state,
                        device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t]
            y_t = (h * Cp[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)

        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_ssm
        y = y * F.silu(z)
        y = self.out_proj(y)
        return residual + self.dropout(y)