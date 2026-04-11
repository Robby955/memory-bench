"""
Recurrent Memory Transformer (RMT).

Splits sequences into segments. M memory tokens carry hidden states between
segments via a learned projection. Truncated BPTT controls gradient depth.

Ref: Bulatov et al., "Recurrent Memory Transformer" (NeurIPS 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from nanochat.gpt import GPT, GPTConfig, norm
from nanochat.common import COMPUTE_DTYPE
from memory_bench.mechanisms.base import MemoryModule


class RMTMemoryHolder(nn.Module):
    """Holds RMT memory parameters as a proper nn.Module so they appear in model.parameters()."""

    def __init__(self, memory_init: nn.Parameter, memory_proj: nn.Linear):
        super().__init__()
        self.memory_init = memory_init
        self.memory_proj = memory_proj


class RMTMemory(MemoryModule):

    def __init__(
        self,
        num_tokens: int = 16,
        seg_length: int = 512,
        bptt_depth: int = 2,
        use_grad_checkpoint: bool = False,
    ):
        self._num_tokens = num_tokens
        self._seg_length = seg_length
        self._bptt_depth = bptt_depth
        self._use_grad_checkpoint = use_grad_checkpoint

        # Set in wrap_model
        self.memory_init = None
        self.memory_proj = None
        self._memory_params = []
        self._n_embd = 0
        self._memory_state = None

    def wrap_model(self, model: GPT, config: GPTConfig) -> GPT:
        self._n_embd = config.n_embd
        device = model.transformer.wte.weight.device
        dtype = COMPUTE_DTYPE if device.type != "meta" else torch.float32

        # Learned initial memory embeddings
        # Initialized from truncated normal (better than uniform for embeddings)
        self.memory_init = nn.Parameter(
            torch.randn(1, self._num_tokens, config.n_embd, dtype=dtype, device=device) * 0.02
        )

        # Learned output projection for memory states
        # Projects memory hidden states before passing to next segment
        # Initialized near identity for safe start
        self.memory_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        if device.type != "meta":
            self.memory_proj = self.memory_proj.to(device=device, dtype=dtype)
        with torch.no_grad():
            nn.init.eye_(self.memory_proj.weight)
            self.memory_proj.weight.mul_(0.5)  # Start as 0.5 * I (mild damping)
            nn.init.zeros_(self.memory_proj.bias)

        self._memory_params = [
            self.memory_init,
            self.memory_proj.weight,
            self.memory_proj.bias,
        ]

        # Register as a submodule so params appear in model.parameters()
        model.rmt_memory = RMTMemoryHolder(self.memory_init, self.memory_proj)

        return model

    def _run_transformer_layers(self, model, x, full_ids, cos_sin):
        x0 = x
        n_layer = model.config.n_layer
        backout_layer = n_layer // 2
        x_backout = None

        for i, block in enumerate(model.transformer.h):
            # Residual stream mixing (same as nanochat)
            x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0

            # Value embeddings (lookup by token id, zeros for memory)
            ve = (
                model.value_embeds[str(i)](full_ids).to(x.dtype)
                if str(i) in model.value_embeds
                else None
            )

            # Gradient checkpointing (optional)
            if self._use_grad_checkpoint and x.requires_grad:
                x = grad_checkpoint(
                    block, x, ve, cos_sin, model.window_sizes[i], None,
                    use_reentrant=False,
                )
            else:
                x = block(x, ve, cos_sin, model.window_sizes[i], kv_cache=None)

            if i == backout_layer:
                x_backout = x

        # Backout (subtract mid-layer residual)
        if x_backout is not None:
            x = x - model.backout_lambda.to(x.dtype) * x_backout

        x = norm(x)
        return x

    def _forward_segment_core(self, model, segment_tokens, memory_state=None):
        """Shared core: embed, smear, transform, project memory, compute logits.

        Returns (logits, new_memory_state).
        """
        B, S = segment_tokens.size()
        M = self._num_tokens

        if memory_state is None:
            memory_state = self.memory_init.expand(B, -1, -1)

        # Embed segment tokens
        x_tokens = model.transformer.wte(segment_tokens)
        x_tokens = x_tokens.to(COMPUTE_DTYPE)
        x_tokens = norm(x_tokens)

        # Prepend memory hidden states
        x = torch.cat([memory_state.to(x_tokens.dtype), x_tokens], dim=1)  # (B, M+S, D)

        # Token IDs for value embeddings (zeros for memory positions)
        dummy_mem_ids = torch.zeros(B, M, dtype=segment_tokens.dtype, device=segment_tokens.device)
        full_ids = torch.cat([dummy_mem_ids, segment_tokens], dim=1)

        # Smear: apply only among real tokens (memory excluded from predecessor chain)
        if S > 1:
            x_tok = x[:, M:]  # (B, S, D)
            gate = model.smear_lambda.to(x.dtype) * torch.sigmoid(
                model.smear_gate(x_tok[:, 1:, :24])
            )
            x_tok = torch.cat(
                [x_tok[:, :1], x_tok[:, 1:] + gate * x_tok[:, :-1]], dim=1
            )
            x = torch.cat([x[:, :M], x_tok], dim=1)

        # RoPE: memory tokens get positions 0..M-1, real tokens M..M+S-1
        T_full = M + S
        cos_sin = model.cos[:, :T_full], model.sin[:, :T_full]

        # Run through transformer layers
        x = self._run_transformer_layers(model, x, full_ids, cos_sin)

        # Extract and project memory states
        raw_memory = x[:, :M, :]
        new_memory_state = self.memory_proj(raw_memory)

        # Logits on token positions only (with softcap)
        x_tokens_out = x[:, M:, :]
        logits = model.lm_head(x_tokens_out)
        logits = logits[..., :model.config.vocab_size]
        logits = logits.float()
        logits = 15.0 * torch.tanh(logits / 15.0)

        return logits, new_memory_state

    def forward_segment(self, model, segment_tokens, segment_targets, memory_state=None):
        """Forward one segment, returning (loss, new_memory_state)."""
        logits, new_memory_state = self._forward_segment_core(model, segment_tokens, memory_state)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            segment_targets.reshape(-1),
            ignore_index=-1,
            reduction="mean",
        )
        return loss, new_memory_state

    def forward_segment_logits(self, model, segment_tokens, memory_state=None):
        """Forward one segment, returning (logits, new_memory_state) for eval."""
        return self._forward_segment_core(model, segment_tokens, memory_state)

    def on_segment_boundary(self, memory_state):
        return memory_state

    def extra_param_groups(self):
        return [{
            "kind": "adamw",
            "params": self._memory_params,
            "lr": 0.01,
            "betas": (0.8, 0.95),
            "eps": 1e-10,
            "weight_decay": 0.0,
        }]

    def reset(self):
        self._memory_state = None

    @property
    def name(self) -> str:
        return f"rmt-m{self._num_tokens}-s{self._seg_length}"

    @property
    def num_memory_params(self) -> int:
        return sum(p.numel() for p in self._memory_params)

    @property
    def requires_segments(self) -> bool:
        return True

    @property
    def segment_length(self) -> int:
        return self._seg_length

    @property
    def num_memory_tokens(self) -> int:
        return self._num_tokens

    @property
    def bptt_depth(self) -> int:
        return self._bptt_depth
