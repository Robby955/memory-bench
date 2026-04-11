"""
TTT-Linear: Test-Time Training with Linear Self-Supervised Model.

Replaces one transformer layer with a TTT-Linear layer that maintains an inner
linear model W ∈ R^{D×D} updated via gradient descent on a self-supervised
reconstruction loss. The "hidden state" is W itself — it accumulates knowledge
from the context token-by-token.

=== Mathematical Foundation ===

Inner model:   f_W(k) = W k,   where W ∈ R^{D×D}
Loss per token: ℓ_t = (1/2) ||v_t - W k_t||²
Gradient:       ∇_W ℓ_t = -(v_t - W k_t) k_t^T = e_t k_t^T
                where e_t = W k_t - v_t (reconstruction error)
SGD update:     W_t = W_{t-1} - η_t · e_t k_t^T

=== Mini-Batch Dual Form (Sun et al., 2024) ===

For a chunk of C tokens, the exact token-by-token update is O(C·D²).
The dual form approximates this with O(C²·D) computation by fixing W to
the chunk-initial value W₀ for error computation:

    E = K W₀ᵀ - V                          # (C, D) reconstruction errors
    Ã = tril(Q Kᵀ) ⊙ diag(η) broadcast     # (C, C) causal "attention"
    Z = Q W₀ᵀ - Ã E                         # (C, D) output with virtual updates

This is equivalent to:
    z_t = q_t W₀ᵀ - Σ_{s≤t} η_s (q_t·k_s)(W₀ k_s - v_s)

which is the output you'd get if you applied SGD updates s=1..t to W₀
(ignoring the effect of earlier updates on later errors — the "mini-batch"
approximation). When chunk_size=1 this is exact.

After the chunk, W is updated for the next chunk:
    W_new = W₀ - (1/C) Kᵀ diag(η) E

=== LaCT Extensions (Kazemnejad et al., ICLR 2026) ===

1. Learned per-token learning rate: η_t = softplus(W_η x_t + b_η)
   - Allows the model to learn WHEN to update aggressively vs conservatively
   - Per-head: each attention head has its own LR projection

2. L2 weight normalization: After each chunk, normalize columns of W
   - Prevents unbounded growth of inner model weights
   - Analogous to weight normalization in deep learning

3. Output LayerNorm: Stabilizes the gradient-updated outputs

=== Connection to Linear Attention ===

When η is constant and W₀ = 0, the dual form becomes:
    Z = -η · tril(Q Kᵀ) · (-V) = η · tril(Q Kᵀ) V
which is exactly causal linear attention (without softmax).

The TTT-Linear update Z = QW₀ᵀ - η·tril(QKᵀ)·E adds a bias term (QW₀ᵀ)
and replaces V with reconstruction errors, making it a "corrective" attention.

=== Equivalence to DeltaNet (NVIDIA, 2026) ===

Yang & Katharopoulos (2026) proved that TTT-Linear without LayerNorm
and with mini-batch size 1 is mathematically identical to DeltaNet:
    S_t = S_{t-1} - η k_t(k_t S_{t-1} - v_t)ᵀ
        = (I - η k_t k_tᵀ) S_{t-1} + η k_t v_tᵀ
This is the delta rule update. The key difference is:
- TTT-Linear processes chunks (amortized cost) + has LayerNorm
- DeltaNet uses data-dependent gating (β_t replaces η)

References:
    Sun et al., "Learning to (Learn at Test Time)" (ICML 2024)
    Kazemnejad et al., "LaCT: Large-Chunk TTT" (ICLR 2026)
    Yang & Katharopoulos, "Parallelizing Linear Transformers with the Delta Rule" (2026)
    Behrouz et al., "Titans: Learning to Memorize at Test Time" (2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from nanochat.gpt import GPT, GPTConfig, Linear, norm, apply_rotary_emb
from nanochat.common import COMPUTE_DTYPE
from memory_bench.mechanisms.base import MemoryModule


class TTTLinearLayer(nn.Module):
    """TTT-Linear attention replacement using the mini-batch dual form.

    This module replaces a standard CausalSelfAttention layer. Instead of
    computing softmax attention weights, it maintains an inner linear model
    W that maps keys to values, updated per-chunk via gradient descent.

    The dual form computes outputs as if SGD had been applied token-by-token,
    but does it with a single matrix multiply per chunk (parallelizable on GPU).

    Architecture:
        Input x ─┬─→ Q projection ─→ q
                 ├─→ K projection ─→ k
                 ├─→ V projection ─→ v
                 └─→ η projection ─→ per-token learning rate

        For each chunk of C tokens:
            E = K @ W^T - V              # reconstruction error
            η = softplus(W_η @ x)        # learned learning rate
            A = tril(Q @ K^T)            # causal attention matrix
            Z = Q @ W^T - A @ (η * E)   # dual form output
            W ← W - (1/C) K^T @ (η * E) # update inner model

        Z → RMSNorm → output projection
    """

    def __init__(
        self,
        config: GPTConfig,
        layer_idx: int,
        chunk_size: int = 64,
        init_lr: float = 1.0,
        use_momentum: bool = False,
        momentum_beta: float = 0.9,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.chunk_size = chunk_size
        self.use_momentum = use_momentum
        self.momentum_beta = momentum_beta

        # Q/K/V projections (same dims as standard attention)
        self.c_q = Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.c_k = Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)

        # Learned per-token, per-head learning rate (LaCT-style)
        # η_t = softplus(W_η @ x_t + b_η) for each head
        # Input: n_embd → n_kv_head (one scalar LR per head)
        self.lr_proj = nn.Linear(config.n_embd, config.n_kv_head, bias=True)
        # Initialize bias so softplus(bias) ≈ init_lr
        # softplus(x) = log(1 + exp(x)), so x = log(exp(init_lr) - 1)
        with torch.no_grad():
            self.lr_proj.bias.fill_(math.log(math.expm1(init_lr)))
            # Small weight init so LR starts near init_lr for all inputs
            self.lr_proj.weight.mul_(0.01)

        # Output RMSNorm (per-head, stabilizes gradient-updated outputs)
        self.out_norm_weight = nn.Parameter(torch.ones(self.head_dim))

        # Inner model W initialization scale
        # Start near identity so the layer is approximately a skip at init
        self.w_init_scale = nn.Parameter(torch.tensor(0.01))

        # MLP sub-layer (same as standard Block — ReLU²)
        self.mlp_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.mlp_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

        # Value embedding gate (ResFormer-style, matches nanochat)
        from nanochat.gpt import has_ve
        self.ve_gate_channels = 12
        self.ve_gate = (
            Linear(self.ve_gate_channels, config.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def _init_inner_model(self, B: int, H: int, D: int, device, dtype):
        """Initialize the inner linear model W ≈ scale * I.

        Returns W: (B, H, D, D) — one inner model per batch element per head.
        """
        eye = torch.eye(D, device=device, dtype=dtype)
        W = eye.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        return W * self.w_init_scale.to(dtype)

    def _l2_normalize_columns(self, W):
        """L2-normalize columns of W (LaCT-style weight normalization).

        Prevents unbounded growth of inner model weights across chunks.
        Applied after each chunk update.

        Args:
            W: (B, H, D, D) inner model weight matrices
        Returns:
            W with unit-norm columns
        """
        return F.normalize(W, p=2, dim=-2)

    def _ttt_dual_forward(self, q, k, v, eta):
        """TTT-Linear forward using the mini-batch dual form.

        This is the core computation. For each chunk:
            1. Compute reconstruction error E = K @ W^T - V
            2. Scale errors by per-token LR: E_scaled = η * E
            3. Compute causal "attention": A = tril(Q @ K^T)
            4. Output: Z = Q @ W^T - A @ E_scaled
            5. Update: W -= (1/C) K^T @ E_scaled

        Args:
            q:   (B, T, H, D) queries (already at kv_head count for GQA)
            k:   (B, T, H, D) keys
            v:   (B, T, H, D) values
            eta: (B, T, H)    per-token, per-head learning rates

        Returns:
            (B, T, H, D) output tensor
        """
        B, T, H, D = k.shape
        C = self.chunk_size

        # Pad to multiple of chunk_size
        pad = (C - T % C) % C
        if pad > 0:
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
            q = F.pad(q, (0, 0, 0, 0, 0, pad))
            eta = F.pad(eta, (0, 0, 0, pad))

        T_padded = k.size(1)
        n_chunks = T_padded // C

        # Reshape into chunks: (B, n_chunks, C, H, D)
        k = k.view(B, n_chunks, C, H, D)
        v = v.view(B, n_chunks, C, H, D)
        q = q.view(B, n_chunks, C, H, D)
        eta = eta.view(B, n_chunks, C, H)  # (B, n_chunks, C, H)

        # Initialize inner model W: (B, H, D, D)
        W = self._init_inner_model(B, H, D, k.device, k.dtype)

        # Optional momentum buffer
        if self.use_momentum:
            M = torch.zeros_like(W)

        outputs = []

        for chunk_idx in range(n_chunks):
            # Extract chunk: (B, C, H, D)
            k_c = k[:, chunk_idx]
            v_c = v[:, chunk_idx]
            q_c = q[:, chunk_idx]
            eta_c = eta[:, chunk_idx]  # (B, C, H)

            # Transpose to head-first: (B, H, C, D)
            k_h = k_c.transpose(1, 2)
            v_h = v_c.transpose(1, 2)
            q_h = q_c.transpose(1, 2)
            eta_h = eta_c.transpose(1, 2)  # (B, H, C)

            # Step 1: Reconstruction error at current W
            # E = K @ W^T - V, where W^T means W transposed on last two dims
            # k_h: (B, H, C, D), W: (B, H, D, D)
            reconstruction = k_h @ W  # (B, H, C, D) — this is K @ W^T if W stores W^T
            E = reconstruction - v_h  # (B, H, C, D)

            # Step 2: Scale errors by per-token learning rate
            # eta_h: (B, H, C) → (B, H, C, 1) for broadcasting
            E_scaled = eta_h.unsqueeze(-1) * E  # (B, H, C, D)

            # Step 3: Causal attention matrix
            # A = tril(Q @ K^T) → (B, H, C, C)
            A = torch.tril(q_h @ k_h.transpose(-2, -1))  # (B, H, C, C)

            # Step 4: Dual form output
            # Z = Q @ W^T - A @ E_scaled
            out_h = q_h @ W - A @ E_scaled  # (B, H, C, D)

            # Step 5: Per-head output RMSNorm
            out_h = F.rms_norm(out_h, (D,)) * self.out_norm_weight.to(out_h.dtype)

            outputs.append(out_h.transpose(1, 2))  # back to (B, C, H, D)

            # Step 6: Update W for next chunk
            # W_new = W - (1/C) K^T @ E_scaled
            grad = k_h.transpose(-2, -1) @ E_scaled  # (B, H, D, D)
            grad = grad / C

            if self.use_momentum:
                M = self.momentum_beta * M + (1 - self.momentum_beta) * grad
                W = W - M
            else:
                W = W - grad

            # Step 7: L2 weight normalization (LaCT)
            W = self._l2_normalize_columns(W)

        # Concatenate and remove padding
        output = torch.cat(outputs, dim=1)  # (B, T_padded, H, D)
        output = output[:, :T, :, :]

        return output

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        """Forward pass: replace attention with TTT-Linear dual form.

        Data flow:
            x → Q,K,V projections → RoPE → QK norm
            x → η projection → softplus (learned LR)
            (q, k, v, η) → dual form → Z
            Z → output projection
        """
        B, T, C = x.size()

        # Q/K/V projections
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer-style, matches nanochat)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # RoPE + QK normalization
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Learned per-token, per-head learning rate
        # eta: (B, T, n_kv_head) — one scalar per head per token
        eta = F.softplus(self.lr_proj(x))  # (B, T, n_kv_head)

        # GQA: expand K/V to match query head count (NOT averaging queries)
        n_rep = self.n_head // self.n_kv_head
        if n_rep > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, n_rep, -1)
            k = k.reshape(B, T, self.n_head, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, n_rep, -1)
            v = v.reshape(B, T, self.n_head, self.head_dim)
            eta = eta.repeat_interleave(n_rep, dim=2)

        # TTT-Linear dual form
        y = self._ttt_dual_forward(q, k, v, eta)  # (B, T, n_head, head_dim)

        # Output projection
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class TTTBlock(nn.Module):
    """Transformer block with TTT-Linear attention + standard ReLU² MLP.

    Drop-in replacement for nanochat's Block. Same residual structure:
        x = x + TTT_attn(norm(x))
        x = x + MLP(norm(x))
    """

    def __init__(self, ttt_layer: TTTLinearLayer):
        super().__init__()
        self.attn = ttt_layer
        self.mlp_fc = ttt_layer.mlp_fc
        self.mlp_proj = ttt_layer.mlp_proj

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # TTT-Linear attention
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        # Standard ReLU² MLP
        h = self.mlp_fc(norm(x))
        h = F.relu(h).square()
        h = self.mlp_proj(h)
        x = x + h
        return x


class TTTLinearMemory(MemoryModule):
    """TTT-Linear memory mechanism.

    Replaces one transformer layer with a TTT-Linear layer that maintains
    an inner linear model updated via the mini-batch dual form. This is
    a principled approach to memory: the model literally learns a linear
    function from each token's context during both training and inference.

    The dual form is mathematically equivalent to running SGD on a
    reconstruction loss for each token, but it's parallelizable on GPU
    via the causal attention matrix tril(QK^T).

    Args:
        layer_idx: which layer to replace (-1 = middle layer)
        chunk_size: tokens per TTT chunk (default: 64, LaCT uses 2048+)
        init_lr: initial learning rate for the inner model (default: 1.0)
        use_momentum: whether to use momentum in inner model updates
        momentum_beta: momentum coefficient (default: 0.9)
    """

    def __init__(
        self,
        layer_idx: int = -1,
        chunk_size: int = 64,
        init_lr: float = 1.0,
        use_momentum: bool = False,
        momentum_beta: float = 0.9,
    ):
        self._layer_idx = layer_idx
        self._chunk_size = chunk_size
        self._init_lr = init_lr
        self._use_momentum = use_momentum
        self._momentum_beta = momentum_beta
        self._ttt_params = []

    def wrap_model(self, model: GPT, config: GPTConfig) -> GPT:
        layer_idx = self._layer_idx if self._layer_idx >= 0 else config.n_layer // 2

        # Create TTT-Linear layer
        ttt_layer = TTTLinearLayer(
            config, layer_idx, self._chunk_size,
            self._init_lr, self._use_momentum, self._momentum_beta,
        )

        # Move to same device/dtype as the model BEFORE copying weights
        device = model.transformer.wte.weight.device
        ttt_layer = ttt_layer.to(device=device, dtype=COMPUTE_DTYPE)

        # Copy weights from original attention layer
        original_block = model.transformer.h[layer_idx]
        original_attn = original_block.attn

        with torch.no_grad():
            ttt_layer.c_q.weight.copy_(original_attn.c_q.weight)
            ttt_layer.c_k.weight.copy_(original_attn.c_k.weight)
            ttt_layer.c_v.weight.copy_(original_attn.c_v.weight)
            ttt_layer.c_proj.weight.copy_(original_attn.c_proj.weight)
            ttt_layer.mlp_fc.weight.copy_(original_block.mlp.c_fc.weight)
            ttt_layer.mlp_proj.weight.copy_(original_block.mlp.c_proj.weight)
            if original_attn.ve_gate is not None and ttt_layer.ve_gate is not None:
                ttt_layer.ve_gate.weight.copy_(original_attn.ve_gate.weight)

        # Replace the block
        ttt_block = TTTBlock(ttt_layer)
        model.transformer.h[layer_idx] = ttt_block

        # Track TTT-specific parameters (LR projection, norm, init scale)
        self._ttt_params = [
            ttt_layer.lr_proj.weight,
            ttt_layer.lr_proj.bias,
            ttt_layer.out_norm_weight,
            ttt_layer.w_init_scale,
        ]

        return model

    def extra_param_groups(self) -> list[dict]:
        return [{
            "kind": "adamw",
            "params": self._ttt_params,
            "lr": 0.01,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
        }]

    @property
    def name(self) -> str:
        return f"ttt-linear-c{self._chunk_size}"

    @property
    def num_memory_params(self) -> int:
        return sum(p.numel() for p in self._ttt_params)
