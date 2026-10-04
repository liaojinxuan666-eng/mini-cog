# config.py
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 64
    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_head: int = 4
    window: int = 64
    n_layer: int = 7
    attn_layer: int = 3
    max_len: int = 512
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 32
    half_len: int = 64
    steps: int = 2000
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 50
    seed: int = 42
    out_dir: str = "checkpoints"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "cuda"


CFG = Config()