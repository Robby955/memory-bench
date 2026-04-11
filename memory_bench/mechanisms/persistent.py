"""
Persistent Memory Tokens.

Learned KV pairs prepended to every attention layer. No positional encoding
(no RoPE). Memory values use a zero-init residual scale so the mechanism
starts as identity and learns to contribute.

Uses flex_attention (PyTorch 2.9+) for efficient GPU training with custom
memory-prefix-causal masking. Falls back to SDPA with explicit mask on CPU.

Ref: Burtsev et al., "Memory Transformer" (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig, CausalSelfAttention, norm, apply_rotary_emb
from memory_bench.mechanisms.base import MemoryModule

# flex_attention: available in PyTorch 2.9+, CUDA only
_flex_available = False
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _flex_available = True
except ImportError:
    pass

# Cache for block masks keyed by (M, T, device) to avoid recomputation
_block_mask_cache: dict[tuple, object] = {}


def _get_block_mask(M: int, T: int, device: torch.device):
    """Get or create a cached block mask for memory-prefix-causal attention."""
    key = (M, T, device)
    if key not in _block_mask_cache:
        def memory_causal_mask(b, h, q_idx, kv_idx):
            is_memory = kv_idx < M
            is_causal = q_idx >= (kv_idx - M)
            return is_memory | is_causal

        _block_mask_cache[key] = create_block_mask(
            memory_causal_mask,
            B=1, H=1,
            Q_LEN=T, KV_LEN=M + T,
            device=device,
        )
    return _block_mask_cache[key]


class PersistentMemoryAttention(nn.Module):
    """CausalSelfAttention wrapper that prepends persistent memory KVs."""

    def __init__(
        self,
        original_attn: CausalSelfAttention,
        mem_k: nn.Parameter,
        mem_v: nn.Parameter,
        v_scale: nn.Parameter,
    ):
        super().__init__()
        self.attn = original_attn
        self.mem_k = mem_k          # (1, M, n_kv_head, head_dim)
        self.mem_v = mem_v          # (1, M, n_kv_head, head_dim)
        self.v_scale = v_scale      # (1,) per-layer residual scale

        # Copy attributes needed by parent model
        self.layer_idx = original_attn.layer_idx
        self.n_head = original_attn.n_head
        self.n_kv_head = original_attn.n_kv_head
        self.n_embd = original_attn.n_embd
        self.head_dim = original_attn.head_dim
        self.c_q = original_attn.c_q
        self.c_k = original_attn.c_k
        self.c_v = original_attn.c_v
        self.c_proj = original_attn.c_proj
        self.ve_gate = original_attn.ve_gate
        self.ve_gate_channels = original_attn.ve_gate_channels

    def _forward_flex(self, q, k_full, v_full, M, B, T):
        """Fast path: flex_attention with compiled memory-causal mask (CUDA only)."""
        # flex_attention expects (B, H, T, D)
        q_flex = q.transpose(1, 2)              # (B, n_head, T, D)
        k_flex = k_full.transpose(1, 2)         # (B, n_kv_head, M+T, D)
        v_flex = v_full.transpose(1, 2)         # (B, n_kv_head, M+T, D)

        block_mask = _get_block_mask(M, T, q.device)
        enable_gqa = self.n_head != self.n_kv_head
        y = flex_attention(q_flex, k_flex, v_flex,
                           block_mask=block_mask, enable_gqa=enable_gqa)
        return y.transpose(1, 2)  # (B, T, H, D)

    def _forward_sdpa(self, q, k_full, v_full, M, B, T):
        """Fallback path: SDPA with explicit float mask (CPU or non-compile)."""
        q_sdpa = q.transpose(1, 2)  # (B, H, T, D)
        n_rep = self.n_head // self.n_kv_head
        if n_rep > 1:
            k_sdpa = k_full.transpose(1, 2).repeat_interleave(n_rep, dim=1)
            v_sdpa = v_full.transpose(1, 2).repeat_interleave(n_rep, dim=1)
        else:
            k_sdpa = k_full.transpose(1, 2)
            v_sdpa = v_full.transpose(1, 2)

        mask = torch.zeros(T, M + T, dtype=q_sdpa.dtype, device=q.device)
        causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        mask[:, M:] = torch.where(causal, 0.0, float("-inf"))

        y = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=mask)
        return y.transpose(1, 2)  # (B, T, H, D)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Standard Q/K/V projection
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (same as original nanochat)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # RoPE to regular Q/K (not to memory, they are position-agnostic)
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2
        k = k * 1.2

        # Prepare memory KVs
        M = self.mem_k.size(1)
        mem_k = self.mem_k.expand(B, -1, -1, -1).to(dtype=k.dtype)
        mem_v_scaled = self.mem_v.expand(B, -1, -1, -1).to(dtype=v.dtype) * self.v_scale.to(v.dtype)
        mem_k = norm(mem_k) * 1.2

        # Concatenate: [memory_KV ; regular_KV]
        k_full = torch.cat([mem_k, k], dim=1)       # (B, M+T, n_kv_head, head_dim)
        v_full = torch.cat([mem_v_scaled, v], dim=1) # (B, M+T, n_kv_head, head_dim)

        # Use flex_attention on CUDA (fast), SDPA fallback on CPU
        if _flex_available and x.is_cuda:
            y = self._forward_flex(q, k_full, v_full, M, B, T)
        else:
            y = self._forward_sdpa(q, k_full, v_full, M, B, T)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class PersistentMemory(MemoryModule):
    def __init__(self, num_tokens: int = 32):
        self.num_tokens = num_tokens
        self._memory_params = []
        self._n_layer = 0
        self._n_kv_head = 0
        self._head_dim = 0

    def wrap_model(self, model: GPT, config: GPTConfig) -> GPT:
        self._n_layer = config.n_layer
        self._n_kv_head = config.n_kv_head
        self._head_dim = config.n_embd // config.n_head
        self._memory_params = []

        M = self.num_tokens
        for i, block in enumerate(model.transformer.h):
            device = block.attn.c_q.weight.device

            mem_k = nn.Parameter(torch.randn(1, M, self._n_kv_head, self._head_dim) * 0.02)
            mem_v = nn.Parameter(torch.randn(1, M, self._n_kv_head, self._head_dim) * 0.02)
            v_scale = nn.Parameter(torch.tensor(0.01))

            if device.type != "meta":
                mem_k = nn.Parameter(mem_k.to(device=device))
                mem_v = nn.Parameter(mem_v.to(device=device))
                v_scale = nn.Parameter(v_scale.to(device=device))

            self._memory_params.extend([mem_k, mem_v, v_scale])

            block.attn = PersistentMemoryAttention(
                block.attn, mem_k, mem_v, v_scale,
            )

        return model

    def extra_param_groups(self) -> list[dict]:
        return [{
            "kind": "adamw",
            "params": self._memory_params,
            "lr": 0.01,
            "betas": (0.8, 0.95),
            "eps": 1e-10,
            "weight_decay": 0.01,
        }]

    @property
    def name(self) -> str:
        return f"persistent-{self.num_tokens}"

    @property
    def num_memory_params(self) -> int:
        return sum(p.numel() for p in self._memory_params)
