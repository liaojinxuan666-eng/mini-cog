# model/mamba_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """
    简化版选择性状态空间块（Mamba-like）。

    真实 Mamba 用 CUDA kernel 做并行扫描，这里用纯 PyTorch 串行扫描，
    目的是先跑通结构、理解数据流。速度慢，后面可以换官方 mamba-ssm。

    核心公式（离散化后）：
        h_t = a_t * h_{t-1} + b_t        状态更新
        y_t = C_t · h_t                  读出
    其中 a_t = exp(dt_t * A) 由输入决定，这就是"选择性"。
    """

    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand   # 内部扩展维度，默认 256
        self.d_state = d_state            # 状态维度，默认 16
        self.d_conv = d_conv              # 卷积核大小，默认 4

        self.ln = nn.LayerNorm(d_model)

        # 输入投影：一份走 SSM，一份做门控
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)

        # 因果深度可分离卷积：每个通道独立卷积，提取局部特征
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,     # 深度可分离
            padding=d_conv - 1,      # 保证输出不短于输入，后面裁掉多的
        )

        # 从输入生成 B、C（选择性 SSM 的关键：B、C 依赖输入）
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)

        # 从输入生成 dt（时间步长）
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)

        # A 参数：每个通道一组状态衰减
        # 用 -exp(A_log) 保证值为负，离散化后 a_t 在 (0,1)，状态稳定
        A = torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D 参数：跳连，让模型能直接传递原始信号
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影，把 d_inner 降回 d_model
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, d_model]
        B, L, D = x.shape

        residual = x                       # 残差保存
        x = self.ln(x)                     # 预归一化

        # 一分为二：x_ssm 走状态空间，z 做门控
        xz = self.in_proj(x)               # [B, L, 2*d_inner]
        x_ssm, z = xz.chunk(2, dim=-1)     # 各 [B, L, d_inner]

        # 因果卷积（卷积要求 [B, C, L] 布局）
        x_ssm = x_ssm.transpose(1, 2)                 # [B, d_inner, L]
        x_ssm = self.conv1d(x_ssm)[..., :L]           # 裁掉 padding 多出的部分
        x_ssm = x_ssm.transpose(1, 2)                 # [B, L, d_inner]
        x_ssm = F.silu(x_ssm)

        # 生成 SSM 参数（都依赖输入，这就是 selectivity）
        BC = self.x_proj(x_ssm)                       # [B, L, 2*d_state]
        Bp, Cp = BC.chunk(2, dim=-1)                  # 各 [B, L, d_state]

        # dt 必须为正，用 softplus 保证
        dt = F.softplus(self.dt_proj(x_ssm))          # [B, L, d_inner]

        # A 取负
        A = -torch.exp(self.A_log)                    # [d_inner, d_state]

        # 离散化
        # dt: [B, L, d_inner]      -> [B, L, d_inner, 1]
        # A:  [d_inner, d_state]   -> [1, 1, d_inner, d_state]
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # dA: [B, L, d_inner, d_state]

        # dB: [B, L, d_inner, d_state]
        dB = dt.unsqueeze(-1) * Bp.unsqueeze(2)

        # ---- 串行扫描（慢，但清晰）----
        # h: [B, d_inner, d_state]
        h = torch.zeros(B, self.d_inner, self.d_state,
                        device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t]                       # [B, d_inner, d_state]
            # 用 C 读出：h 与 C 做点积，沿 d_state 求和
            # Cp[:, t]: [B, d_state] -> [B, 1, d_state]，广播到 d_inner
            y_t = (h * Cp[:, t].unsqueeze(1)).sum(-1)         # [B, d_inner]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                            # [B, L, d_inner]

        # 跳连
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_ssm

        # 门控：z 决定放行多少
        y = y * F.silu(z)

        # 输出投影 + 残差
        y = self.out_proj(y)
        return residual + self.dropout(y)