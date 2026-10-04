# config.py
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    vocab_size: int = 64          # Copy 任务词表，后面换语言模型会改
    d_model: int = 128            # 手机友好
    d_state: int = 16             # SSM 隐状态维度
    d_conv: int = 4               # 深度可分离卷积核
    expand: int = 2               # SSM 内部扩展倍数
    n_head: int = 4               # Attention 头数
    head_dim: int = 32            # 每头维度 = d_model / n_head
    window: int = 64              # 滑动窗口大小
    n_layer: int = 7              # 总层数
    attn_layer: int = 3           # 第几层放 Attention（0-indexed）
    max_len: int = 512
    dropout: float = 0.1

@dataclass
class TrainConfig:
    batch_size: int = 32
    half_len: int = 64            # Copy 任务半段长度
    steps: int = 2000
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup: int = 100
    log_every: int = 50
    eval_every: int = 200
    seed: int = 42
    out_dir: str = "checkpoints"

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "cuda"          # Colab 是 cuda，本地 CPU 会自动 fallback

CFG = Config()