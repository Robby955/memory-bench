"""
Gated DeltaNet — Linear Attention with Delta Rule and Data-Dependent Gating.

Replaces standard softmax attention in specified layers with Gated DeltaNet,
a linear attention variant that maintains a compressed KV memory matrix S ∈ R^{D×D}
updated per token via a delta rule with learned gating.

=== Core Recurrence ===

    S_t = diag(α_t) · S_{t-1} − β_t · k_t · (k_tᵀ · S_{t-1})ᵀ + β_t · k_t · v_tᵀ
    o_t = q_tᵀ · S_t

Simplified (without the separate α gating):
    S_t = (I − β_t · k_t k_tᵀ) · S_{t-1} + β_t · k_t v_tᵀ
    o_t = q_tᵀ · S_t

Where:
    S_t ∈ R^{D×D}  — compressed KV memory matrix (the "state")
    β_t ∈ (0, 1)    — write gate (sigmoid), controls update strength
    α_t ∈ (0, 1)    — decay gate (per-head scalar), controls forgetting
    k_t, v_t, q_t ∈ R^D — key, value, query vectors

The delta rule is named because the update can be decomposed as:
    1. ERASE:  S ← diag(α) · (I − β · k kᵀ) · S   (forget old association for k)
    2. WRITE:  S ← S + β · k vᵀ                      (write new association)

=== Chunk-Parallel Training via WY Representation ===

For efficient GPU training, the recurrence is computed chunk-parallel using the
WY representation from numerical linear algebra. Within each chunk of C tokens:

    S_chunk = W @ Y @ S_prev + correction_term

Where W and Y are structured matrices derived from the keys and gates.
FLA's Triton kernels implement this as `chunk_gated_delta_rule(q, k, v, beta, gk)`.

=== Gating in Log-Space ===

The decay factor α is parameterized in log-space for numerical stability:
    gk = log(α) = -exp(A_log) * softplus(W_α · x + dt_bias)

This is critical because:
1. α values near 1.0 (slow decay) lose precision in float16
2. Log-space allows stable accumulation over many timesteps
3. FLA kernels can fuse raw-gate -> log-decay computation internally

=== Short Convolution ===

Following Qwen3.5 and the DeltaNet paper, a depthwise short convolution
(kernel_size=4) is applied to Q, K, V before the recurrence. This:
1. Provides local context mixing (cheap bigram/trigram features)
2. Improves gradient flow for the gating mechanism
3. Empirically helps stability at initialization

=== Production Deployments ===

- Qwen3.5 (Alibaba): 3:1 hybrid ratio (3 DeltaNet layers per 1 attention layer)
- Kimi Linear (Moonshot AI): DeltaNet backbone with attention for KV retrieval
- FLA library: 25+ architectures including DeltaNet, all with Triton kernels

References:
    Yang et al., "Gated Delta Networks" (ICLR 2025)
    Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient Training" (2024)
    Schlag et al., "Linear Transformers Are Secretly Fast Weight Programmers" (2021)
    Qwen Team, "Qwen3.5 Technical Report" (2025)
    FLA: github.com/fla-org/flash-linear-attention (MIT license)
"""

import math
import os
from importlib import metadata as importlib_metadata

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig, Linear, norm
from nanochat.common import COMPUTE_DTYPE
from memory_bench.mechanisms.base import MemoryModule


# Check for FLA availability
_HAS_FLA = False
_FLA_VERSION = None
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    _HAS_FLA = True
    for _dist_name in ("fla-core", "fla"):
        try:
            _FLA_VERSION = importlib_metadata.version(_dist_name)
            break
        except importlib_metadata.PackageNotFoundError:
            continue
except ImportError:
    pass


class ShortConv1d(nn.Module):
    """Depthwise causal short convolution.

    Applied to Q, K, V independently before the delta rule recurrence.
    Uses depthwise separable convolution for efficiency (one filter per channel).

    This is a standard component in modern linear attention architectures
    (Mamba, DeltaNet, Qwen3.5) that provides local context at negligible cost.

    Args:
        dim: number of channels (n_heads * head_dim)
        kernel_size: convolution kernel size (default: 4)
    """

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        # Depthwise conv: groups=dim means one filter per channel
        self.conv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,  # causal padding (left only)
            groups=dim,
            bias=False,
        )
        # Initialize as approximate identity (passthrough at init)
        with torch.no_grad():
            nn.init.zeros_(self.conv.weight)
            # Set the last tap (current position) to 1.0
            self.conv.weight[:, 0, -1] = 1.0

    def forward(self, x):
        """Apply causal depthwise convolution.

        Args:
            x: (B, T, D) input tensor
        Returns:
            (B, T, D) convolved tensor (causal: only uses past + current)
        """
        # Conv1d expects (B, D, T)
        y = self.conv(x.transpose(1, 2))
        # Remove right padding to maintain causality
        y = y[:, :, :x.size(1)]
        return y.transpose(1, 2)


class GatedDeltaNetAttention(nn.Module):
    """Gated DeltaNet attention layer with proper gating and short convolution.

    Architecture:
        Input x ─┬─→ Q projection ─→ ShortConv ─→ L2 norm ─→ q
                 ├─→ K projection ─→ ShortConv ─→ L2 norm ─→ k
                 ├─→ V projection ─→ ShortConv ─→ v
                 ├─→ β projection ─→ sigmoid     ─→ beta (write gate)
                 ├─→ α projection ─→ softplus/A_log ─→ gk   (decay in log-space)
                 └─→ g projection ─→ SiLU        ─→ gate  (output gate)

        (q, k, v, beta, gk) → DeltaNet recurrence → o
        o → RMSNorm → o * gate → output projection

    The recurrence (computed by FLA or naive fallback):
        S_t = diag(exp(gk_t)) · (I − β_t·k_t·k_tᵀ) · S_{t-1} + β_t·k_t·v_tᵀ
        o_t = q_tᵀ · S_t
    """

    def __init__(self, config: GPTConfig, layer_idx: int, conv_kernel: int = 4):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * self.head_dim

        # Q/K/V projections
        self.c_q = Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.c_k = Linear(config.n_embd, kv_dim, bias=False)
        self.c_v = Linear(config.n_embd, kv_dim, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)

        # Short convolutions on Q, K, V (depthwise, causal)
        self.conv_q = ShortConv1d(config.n_head * self.head_dim, kernel_size=conv_kernel)
        self.conv_k = ShortConv1d(kv_dim, kernel_size=conv_kernel)
        self.conv_v = ShortConv1d(kv_dim, kernel_size=conv_kernel)

        # Beta (write gate): per-head sigmoid gate
        # Controls how strongly to update the state for this token
        self.c_beta = nn.Linear(config.n_embd, config.n_kv_head, bias=True)
        with torch.no_grad():
            # Initialize beta bias so sigmoid(bias) ≈ 0.5 (moderate update)
            self.c_beta.bias.zero_()
            self.c_beta.weight.mul_(0.01)

        # Raw decay gate input. The actual log-decay is computed as:
        #   g = -exp(A_log) * softplus(c_gk(x) + dt_bias)
        # matching the official FLA GatedDeltaNet implementation.
        self.c_gk = nn.Linear(config.n_embd, config.n_kv_head, bias=False)
        with torch.no_grad():
            self.c_gk.weight.mul_(0.01)
        A = torch.empty(config.n_kv_head, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        dt_min, dt_max, dt_init_floor = 1e-3, 1e-1, 1e-4
        dt = torch.exp(
            torch.rand(config.n_kv_head, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Output gate: SiLU-gated (more expressive than sigmoid)
        self.c_gate = Linear(config.n_embd, config.n_head * self.head_dim, bias=False)

        # Per-token RMS norm on the output, matching official FLA GatedDeltaNet.
        # This normalizes over head_dim only; GroupNorm over [C, T] would couple
        # positions together and make the layer sequence-length dependent.
        self.o_norm = nn.RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

        # VE support (matches nanochat)
        from nanochat.gpt import has_ve
        self.ve_gate_channels = 12
        self.ve_gate = (
            Linear(self.ve_gate_channels, config.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def _expanded_decay_params(self, num_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
        if num_heads == self.n_kv_head:
            return self.A_log, self.dt_bias
        if num_heads == self.n_head:
            n_rep = self.n_head // self.n_kv_head
            return (
                self.A_log.repeat_interleave(n_rep),
                self.dt_bias.repeat_interleave(n_rep),
            )
        raise ValueError(f"Unsupported decay head count: {num_heads}")

    def _compute_log_decay(self, g_raw: torch.Tensor) -> torch.Tensor:
        """Convert raw gate activations into log-space decay values."""
        A_log, dt_bias = self._expanded_decay_params(g_raw.shape[-1])
        return (
            -A_log.float().exp().view(1, 1, -1)
            * F.softplus(g_raw.float() + dt_bias.float().view(1, 1, -1))
        )

    def _raise_nonfinite(self, stage: str, q, k, v, beta, gk_raw, output=None) -> None:
        log_decay = self._compute_log_decay(gk_raw)

        def _range(t: torch.Tensor) -> tuple[float, float]:
            finite = t[torch.isfinite(t)]
            if finite.numel() == 0:
                return float("nan"), float("nan")
            return float(finite.min()), float(finite.max())

        message = (
            f"DeltaNet non-finite at {stage}. "
            f"fla_version={_FLA_VERSION or 'unknown'} "
            f"q_absmax={float(q.abs().max()):.4f} "
            f"k_absmax={float(k.abs().max()):.4f} "
            f"v_absmax={float(v.abs().max()):.4f} "
            f"beta_range={_range(beta)} "
            f"g_raw_range={_range(gk_raw)} "
            f"log_decay_range={_range(log_decay)}"
        )
        if output is not None:
            message += f" output_absmax={float(output[torch.isfinite(output)].abs().max()) if torch.isfinite(output).any() else float('nan'):.4f}"
        raise RuntimeError(message)

    def _naive_recurrent_forward(self, q, k, v, beta, gk):
        """Pure PyTorch fallback implementing the full gated delta rule.

        This is O(T · D²) per step — correct but slow. Used for:
        1. CPU testing (no Triton)
        2. Numerical verification against FLA kernels
        3. Debugging

        Args:
            q:    (B, H, T, D) queries
            k:    (B, H, T, D) keys (L2 normalized)
            v:    (B, H, T, D) values
            beta: (B, H, T)    write gate ∈ (0, 1)
            gk:   (B, H, T) or (B, H, T, D) log-space decay ∈ (-∞, 0]

        Returns:
            (B, H, T, D) output
        """
        B, H, T, D = q.shape
        orig_dtype = q.dtype
        dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        device = q.device
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
        beta = beta.to(dtype)
        gk = gk.to(dtype)

        # State matrix S: (B, H, D, D)
        S = torch.zeros(B, H, D, D, device=device, dtype=dtype)
        outputs = []

        for t in range(T):
            q_t = q[:, :, t]       # (B, H, D)
            k_t = k[:, :, t]       # (B, H, D)
            v_t = v[:, :, t]       # (B, H, D)
            b_t = beta[:, :, t]    # (B, H)
            gk_t = gk[:, :, t]

            # Decay: either scalar per head or vector per dimension.
            alpha = torch.exp(gk_t)
            if alpha.ndim == 2:
                S = S * alpha.unsqueeze(-1).unsqueeze(-1)
            else:
                S = S * alpha.unsqueeze(-1)

            # Delta rule: erase then write
            b = b_t.unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            erase = k_t.unsqueeze(-1) * (k_t.unsqueeze(-2) @ S)
            write = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

            S = S - b * erase + b * write

            # Read
            o_t = (q_t.unsqueeze(-2) @ S).squeeze(-2) * (D ** -0.5)  # (B, H, D)
            outputs.append(o_t)

        return torch.stack(outputs, dim=2).to(orig_dtype)  # (B, H, T, D)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        """Forward pass with short conv, gating, and delta rule recurrence.

        Data flow:
            x → projections → short conv → norms/gates
            → delta rule recurrence (FLA or naive)
            → output norm → output gate → output projection
        """
        B, T, C = x.size()

        # Project Q, K, V
        q_proj = self.c_q(x)  # (B, T, n_head * head_dim)
        k_proj = self.c_k(x)  # (B, T, n_kv_head * head_dim)
        v_proj = self.c_v(x)  # (B, T, n_kv_head * head_dim)

        # Short convolution (causal, depthwise)
        q_proj = self.conv_q(q_proj)
        k_proj = self.conv_k(k_proj)
        v_proj = self.conv_v(v_proj)

        # Reshape to head layout
        q = q_proj.view(B, T, self.n_head, self.head_dim)
        k = k_proj.view(B, T, self.n_kv_head, self.head_dim)
        v = v_proj.view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer-style, matches nanochat)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate_ve = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate_ve.unsqueeze(-1) * ve

        # Compute gates
        beta = torch.sigmoid(self.c_beta(x))  # (B, T, n_kv_head) ∈ (0, 1)
        gk_raw = self.c_gk(x)                 # (B, T, n_kv_head)

        # Output gate (SiLU = x * sigmoid(x), more expressive than sigmoid alone)
        g = F.silu(self.c_gate(x).view(B, T, self.n_head, self.head_dim))

        # GQA: expand KV heads to match query heads
        n_rep = self.n_head // self.n_kv_head
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=2)
            v = v.repeat_interleave(n_rep, dim=2)
            beta = beta.repeat_interleave(n_rep, dim=2)
            gk_raw = gk_raw.repeat_interleave(n_rep, dim=2)

        # Run delta rule recurrence
        if _HAS_FLA and x.is_cuda:
            # Compute log-space decay externally (FLA 0.4.x expects g in log-space)
            gk = self._compute_log_decay(gk_raw)
            q_fla = q.contiguous()
            k_fla = k.contiguous()
            v_fla = v.contiguous()
            beta_fla = beta.contiguous()  # (B, T, H)
            gk_fla = gk.to(q_fla.dtype).contiguous()  # (B, T, H) log-space decay
            for name, tensor in (
                ("q", q_fla),
                ("k", k_fla),
                ("v", v_fla),
                ("beta", beta_fla),
                ("g_log", gk_fla),
            ):
                if not torch.isfinite(tensor).all():
                    self._raise_nonfinite(f"pre-kernel/{name}", q_fla, k_fla, v_fla, beta_fla, gk_raw.contiguous())
            o, _ = chunk_gated_delta_rule(
                q_fla, k_fla, v_fla, gk_fla, beta_fla,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            if not torch.isfinite(o).all():
                self._raise_nonfinite("post-kernel", q_fla, k_fla, v_fla, beta_fla, gk_raw.contiguous(), o)
            y = o  # already (B, T, H, D)
        else:
            # Pure PyTorch fallback with fp32 state accumulation.
            gk = self._compute_log_decay(gk_raw)
            q_h = F.normalize(q, p=2, dim=-1).transpose(1, 2)
            k_h = F.normalize(k, p=2, dim=-1).transpose(1, 2)
            v_h = v.transpose(1, 2)
            beta_h = beta.transpose(1, 2) # (B, H, T)
            gk_h = gk.transpose(1, 2)     # (B, H, T)

            y = self._naive_recurrent_forward(q_h, k_h, v_h, beta_h, gk_h)
            y = y.transpose(1, 2)  # (B, T, H, D)

        # Per-token RMS norm + output gate.
        y_gated = self.o_norm(y) * g

        # Output projection
        y_out = y_gated.contiguous().view(B, T, -1)
        y_out = self.c_proj(y_out)
        return y_out


class GatedDeltaNetBlock(nn.Module):
    """Transformer block with Gated DeltaNet attention + standard ReLU² MLP.

    Drop-in replacement for nanochat's Block.
    """

    def __init__(self, deltanet_attn: GatedDeltaNetAttention, original_mlp):
        super().__init__()
        self.attn = deltanet_attn
        self.mlp = original_mlp

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GatedDeltaNetMemory(MemoryModule):
    """Gated DeltaNet memory mechanism.

    Replaces attention in specified layers with Gated DeltaNet, a linear
    attention variant that maintains a compressed (D×D) state matrix updated
    via the gated delta rule.

    Key properties:
    - O(1) state per layer (fixed D×D matrix, regardless of sequence length)
    - O(T) computation per layer (vs O(T²) for softmax attention)
    - Data-dependent gating: model learns when to write vs forget
    - Log-space decay: numerically stable long-range memory

    Requires fla-core package for GPU Triton kernels (pip install fla-core).
    Falls back to O(T·D²) pure PyTorch on CPU.

    Args:
        layer_indices: which layers to replace (default: [n_layer//3])
        conv_kernel: short convolution kernel size (default: 4)
    """

    def __init__(self, layer_indices: list[int] = None, conv_kernel: int = 4):
        self._layer_indices = layer_indices
        self._conv_kernel = conv_kernel
        self._extra_params = []

    def wrap_model(self, model: GPT, config: GPTConfig) -> GPT:
        if self._layer_indices is None:
            self._layer_indices = [config.n_layer // 3]

        self._extra_params = []

        device = model.transformer.wte.weight.device

        for idx in self._layer_indices:
            assert 0 <= idx < config.n_layer, f"Layer {idx} out of range [0, {config.n_layer})"

            # Create DeltaNet attention
            gdn_attn = GatedDeltaNetAttention(config, idx, self._conv_kernel)

            # Move to same device/dtype as the model BEFORE copying weights
            gdn_attn = gdn_attn.to(device=device, dtype=COMPUTE_DTYPE)

            # Copy weights from original attention
            original_block = model.transformer.h[idx]
            original_attn = original_block.attn

            with torch.no_grad():
                gdn_attn.c_q.weight.copy_(original_attn.c_q.weight)
                gdn_attn.c_k.weight.copy_(original_attn.c_k.weight)
                gdn_attn.c_v.weight.copy_(original_attn.c_v.weight)
                gdn_attn.c_proj.weight.copy_(original_attn.c_proj.weight)
                if original_attn.ve_gate is not None and gdn_attn.ve_gate is not None:
                    gdn_attn.ve_gate.weight.copy_(original_attn.ve_gate.weight)

            # Replace block, keeping original MLP
            gdn_block = GatedDeltaNetBlock(gdn_attn, original_block.mlp)
            model.transformer.h[idx] = gdn_block

            # Track new parameters (gates, convolutions, output norm)
            self._extra_params.extend([
                gdn_attn.c_beta.weight,
                gdn_attn.c_beta.bias,
                gdn_attn.c_gk.weight,
                gdn_attn.A_log,
                gdn_attn.dt_bias,
                gdn_attn.c_gate.weight,
                *gdn_attn.conv_q.parameters(),
                *gdn_attn.conv_k.parameters(),
                *gdn_attn.conv_v.parameters(),
                *gdn_attn.o_norm.parameters(),
            ])

        return model

    def extra_param_groups(self) -> list[dict]:
        if not self._extra_params:
            return []
        decay_params = [
            p for p in self._extra_params
            if p is not None and not getattr(p, "_no_weight_decay", False)
        ]
        no_decay_params = [
            p for p in self._extra_params
            if p is not None and getattr(p, "_no_weight_decay", False)
        ]
        groups = []
        if decay_params:
            groups.append({
                "kind": "adamw",
                "params": decay_params,
                "lr": 0.003,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01,
            })
        if no_decay_params:
            groups.append({
                "kind": "adamw",
                "params": no_decay_params,
                "lr": 0.003,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
            })
        return groups

    @property
    def name(self) -> str:
        layers_str = ",".join(str(i) for i in self._layer_indices) if self._layer_indices else "auto"
        return f"deltanet-L{layers_str}"

    @property
    def num_memory_params(self) -> int:
        return sum(p.numel() for p in self._extra_params)
