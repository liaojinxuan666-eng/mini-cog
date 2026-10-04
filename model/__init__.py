%%writefile /content/mini-cog/model/__init__.py
from .mamba_block import MambaBlock
from .attention_block import AttentionBlock
from .hybrid import HybridLM