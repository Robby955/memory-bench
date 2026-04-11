"""Model construction utilities."""

import torch
import torch.nn as nn
from nanochat.gpt import GPT, GPTConfig


def build_gpt_config(config, vocab_size: int) -> GPTConfig:
    """Build GPTConfig. config needs: depth, aspect_ratio, head_dim, max_seq_len, window_pattern."""
    base_dim = config.depth * config.aspect_ratio
    model_dim = ((base_dim + config.head_dim - 1) // config.head_dim) * config.head_dim
    num_heads = model_dim // config.head_dim
    return GPTConfig(
        sequence_len=config.max_seq_len,
        vocab_size=vocab_size,
        n_layer=config.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=config.window_pattern,
    )


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
