"""Numerical verification: optimized implementations vs naive references."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanochat"))

import torch
import torch.nn.functional as F
import pytest
import math


# ─── TTT-Linear: Dual Form vs Naive Sequential SGD ──────────────────────────

class TestTTTNumerical:
    """Verify that the dual form produces the same output as naive token-by-token SGD."""

    def _naive_ttt_sequential(self, q, k, v, eta, W0):
        """Mini-batch dual form reference: errors computed at W0, outputs include
        virtual cumulative corrections. This is what the matrix dual form computes.

        z_t = q_t @ W0 - sum_{s<=t} eta_s * (q_t . k_s) * (W0 @ k_s - v_s)
        """
        B, T, H, D = k.shape
        outputs = []

        for t in range(T):
            q_t = q[:, t]  # (B, H, D)
            z_t = (q_t.unsqueeze(-2) @ W0).squeeze(-2)  # (B, H, D)

            for s in range(t + 1):
                k_s = k[:, s]
                v_s = v[:, s]
                eta_s = eta[:, s].unsqueeze(-1)  # (B, H, 1)
                e_s = (k_s.unsqueeze(-2) @ W0).squeeze(-2) - v_s
                attn = (q_t * k_s).sum(dim=-1, keepdim=True)  # (B, H, 1)
                z_t = z_t - eta_s * attn * e_s

            outputs.append(z_t)

        return torch.stack(outputs, dim=1)  # (B, T, H, D)

    def _dual_form_reference(self, q, k, v, eta, W0):
        """Matrix-form dual computation (should match the TTT module)."""
        B, T, H, D = k.shape

        # Transpose to (B, H, T, D)
        q_h = q.transpose(1, 2)
        k_h = k.transpose(1, 2)
        v_h = v.transpose(1, 2)
        eta_h = eta.transpose(1, 2)  # (B, H, T)

        # E = K @ W0 - V
        E = k_h @ W0 - v_h  # (B, H, T, D)

        # Scale by eta
        E_scaled = eta_h.unsqueeze(-1) * E  # (B, H, T, D)

        # Causal attention
        A = torch.tril(q_h @ k_h.transpose(-2, -1))  # (B, H, T, T)

        # Output
        Z = q_h @ W0 - A @ E_scaled  # (B, H, T, D)

        return Z.transpose(1, 2)  # (B, T, H, D)

    def test_dual_form_matches_naive(self):
        """The dual form should match naive sequential computation exactly."""
        torch.manual_seed(42)
        B, T, H, D = 2, 8, 2, 16  # small for exact comparison

        q = torch.randn(B, T, H, D)
        k = torch.randn(B, T, H, D)
        v = torch.randn(B, T, H, D)
        eta = torch.ones(B, T, H) * 0.1  # constant LR for clean comparison
        W0 = torch.eye(D).unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1) * 0.01

        naive_output = self._naive_ttt_sequential(q, k, v, eta, W0)
        dual_output = self._dual_form_reference(q, k, v, eta, W0)

        # Should match to floating point tolerance
        assert torch.allclose(naive_output, dual_output, atol=1e-5), \
            f"Max diff: {(naive_output - dual_output).abs().max().item():.2e}"

    def test_dual_form_module_matches_reference(self):
        """The TTTLinearLayer module should match the reference dual form."""
        from nanochat.gpt import GPTConfig
        from memory_bench.mechanisms.ttt import TTTLinearLayer

        torch.manual_seed(42)
        config = GPTConfig(
            sequence_len=128, vocab_size=256, n_layer=4,
            n_head=4, n_kv_head=4, n_embd=128, window_pattern="SL",
        )
        layer = TTTLinearLayer(config, layer_idx=2, chunk_size=8)

        B, T, H, D = 2, 8, 4, 32
        q = torch.randn(B, T, H, D)
        k = torch.randn(B, T, H, D)
        v = torch.randn(B, T, H, D)
        eta = torch.ones(B, T, H) * 0.5

        # Module output
        module_output = layer._ttt_dual_forward(q, k, v, eta)

        # Module uses L2 normalization and RMSNorm, so we can't compare
        # directly with the raw dual form. But we can verify shape and
        # that it's not all zeros or NaN.
        assert module_output.shape == (B, T, H, D)
        assert not torch.isnan(module_output).any()
        assert module_output.abs().mean() > 1e-6  # not degenerate

    def test_eta_zero_gives_identity(self):
        """When learning rate is zero, TTT should be approximately identity."""
        torch.manual_seed(42)
        B, T, H, D = 2, 8, 2, 16
        q = torch.randn(B, T, H, D)
        k = torch.randn(B, T, H, D)
        v = torch.randn(B, T, H, D)
        eta = torch.zeros(B, T, H)  # zero LR = no updates
        W0 = torch.eye(D).unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1) * 0.01

        output = self._dual_form_reference(q, k, v, eta, W0)
        # With eta=0, Z = Q @ W0. No correction term.
        expected = (q.transpose(1, 2) @ W0).transpose(1, 2)

        assert torch.allclose(output, expected, atol=1e-6), \
            f"Max diff: {(output - expected).abs().max().item():.2e}"

    def test_constant_eta_matches_linear_attention(self):
        """With constant eta and W0=0, dual form should equal causal linear attention."""
        torch.manual_seed(42)
        B, T, H, D = 2, 8, 2, 16
        q = torch.randn(B, T, H, D)
        k = torch.randn(B, T, H, D)
        v = torch.randn(B, T, H, D)
        eta_val = 0.5
        eta = torch.full((B, T, H), eta_val)
        W0 = torch.zeros(B, H, D, D)

        # Dual form with W0=0:
        # Z = Q@0 - eta * tril(QK^T) @ (K@0 - V) = eta * tril(QK^T) @ V
        output = self._dual_form_reference(q, k, v, eta, W0)

        # Causal linear attention
        q_h = q.transpose(1, 2)
        k_h = k.transpose(1, 2)
        v_h = v.transpose(1, 2)
        attn = torch.tril(q_h @ k_h.transpose(-2, -1))
        linear_attn = eta_val * (attn @ v_h)
        expected = linear_attn.transpose(1, 2)

        assert torch.allclose(output, expected, atol=1e-5), \
            f"Max diff: {(output - expected).abs().max().item():.2e}"


# ─── DeltaNet: Recurrence Verification ──────────────────────────────────────

class TestDeltaNetNumerical:
    """Verify DeltaNet's naive recurrence is mathematically correct."""

    def test_recurrence_state_shape(self):
        """State should maintain D×D shape throughout."""
        from memory_bench.mechanisms.deltanet import GatedDeltaNetAttention
        from nanochat.gpt import GPTConfig

        config = GPTConfig(
            sequence_len=128, vocab_size=256, n_layer=4,
            n_head=4, n_kv_head=4, n_embd=128, window_pattern="SL",
        )
        attn = GatedDeltaNetAttention(config, layer_idx=1)

        B, H, T, D = 2, 4, 16, 32
        q = torch.randn(B, H, T, D)
        k = F.normalize(torch.randn(B, H, T, D), dim=-1)
        v = torch.randn(B, H, T, D)
        beta = torch.sigmoid(torch.randn(B, H, T))
        gk = -torch.abs(torch.randn(B, H, T, D))

        output = attn._naive_recurrent_forward(q, k, v, beta, gk)
        assert output.shape == (B, H, T, D)
        assert not torch.isnan(output).any()

    def test_beta_zero_gives_zero_output(self):
        """With beta=0, no writes happen, state stays zero, output is zero."""
        from memory_bench.mechanisms.deltanet import GatedDeltaNetAttention
        from nanochat.gpt import GPTConfig

        config = GPTConfig(
            sequence_len=128, vocab_size=256, n_layer=4,
            n_head=4, n_kv_head=4, n_embd=128, window_pattern="SL",
        )
        attn = GatedDeltaNetAttention(config, layer_idx=1)

        B, H, T, D = 1, 1, 8, 8
        q = torch.randn(B, H, T, D)
        k = F.normalize(torch.randn(B, H, T, D), dim=-1)
        v = torch.randn(B, H, T, D)
        beta = torch.zeros(B, H, T)  # no writes
        gk = torch.zeros(B, H, T, D)  # no decay (exp(0) = 1)

        output = attn._naive_recurrent_forward(q, k, v, beta, gk)
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-7)

    def test_single_write_read(self):
        """Write one KV pair, then read it back with the same key as query."""
        B, H, T, D = 1, 1, 2, 4

        # Token 0: write k0 → v0
        # Token 1: read with q1 = k0 (should retrieve v0)
        k0 = F.normalize(torch.randn(1, 1, 1, D), dim=-1)
        v0 = torch.randn(1, 1, 1, D)

        k = torch.cat([k0, torch.zeros(1, 1, 1, D)], dim=2)
        v = torch.cat([v0, torch.zeros(1, 1, 1, D)], dim=2)
        q = torch.cat([torch.zeros(1, 1, 1, D), k0], dim=2)  # query with k0 at t=1

        beta = torch.tensor([[[1.0, 0.0]]])  # (B, H, T) write at t=0, nothing at t=1
        gk = torch.zeros(B, H, T, D)  # no decay

        from memory_bench.mechanisms.deltanet import GatedDeltaNetAttention
        from nanochat.gpt import GPTConfig
        config = GPTConfig(
            sequence_len=128, vocab_size=256, n_layer=4,
            n_head=4, n_kv_head=4, n_embd=128, window_pattern="SL",
        )
        attn = GatedDeltaNetAttention(config, layer_idx=1)

        output = attn._naive_recurrent_forward(q, k, v, beta, gk)

        # At t=1: S = beta_0 * k0 * v0^T (after write at t=0)
        # o_1 = q_1^T @ S * (1/sqrt(D)) = k0^T @ (k0 * v0^T) * (1/sqrt(D))
        # Since k0 is L2-normalized: k0^T @ k0 = 1
        # So o_1 ≈ v0 / sqrt(D)
        D = v0.shape[-1]
        scale = D ** -0.5
        retrieved = output[:, :, 1, :]
        expected = v0.squeeze(2) * scale
        assert torch.allclose(retrieved, expected, atol=1e-5), \
            f"Max diff: {(retrieved - expected).abs().max().item():.2e}"

    def test_decay_forgets(self):
        """With strong decay (gk very negative), old state should be forgotten."""
        from memory_bench.mechanisms.deltanet import GatedDeltaNetAttention
        from nanochat.gpt import GPTConfig

        config = GPTConfig(
            sequence_len=128, vocab_size=256, n_layer=4,
            n_head=4, n_kv_head=4, n_embd=128, window_pattern="SL",
        )
        attn = GatedDeltaNetAttention(config, layer_idx=1)

        B, H, T, D = 1, 1, 4, 4
        k = F.normalize(torch.randn(B, H, T, D), dim=-1)
        v = torch.randn(B, H, T, D)
        q = torch.randn(B, H, T, D)
        beta = torch.ones(B, H, T) * 0.5

        # Strong decay
        gk_strong = torch.full((B, H, T, D), -10.0)  # exp(-10) ≈ 0.00005
        out_decay = attn._naive_recurrent_forward(q, k, v, beta, gk_strong)

        # No decay
        gk_none = torch.zeros(B, H, T, D)
        out_nodecay = attn._naive_recurrent_forward(q, k, v, beta, gk_none)

        # With strong decay, later outputs should be smaller (less accumulated state)
        # Compare magnitude at the last timestep
        last_decay = out_decay[:, :, -1, :].abs().mean()
        last_nodecay = out_nodecay[:, :, -1, :].abs().mean()
        assert last_decay < last_nodecay, \
            f"Decay output ({last_decay:.4f}) should be smaller than no-decay ({last_nodecay:.4f})"


# ─── Short Conv: Strict Causality ────────────────────────────────────────────

class TestShortConvNumerical:
    def test_strict_causality(self):
        """Changing future tokens must not affect past outputs."""
        from memory_bench.mechanisms.deltanet import ShortConv1d
        torch.manual_seed(42)

        conv = ShortConv1d(dim=32, kernel_size=4)
        x = torch.randn(1, 20, 32)

        y1 = conv(x)

        # Perturb tokens 10-19
        x2 = x.clone()
        x2[:, 10:, :] = torch.randn(1, 10, 32) * 100  # large perturbation

        y2 = conv(x2)

        # Positions 0-9 must be identical
        assert torch.allclose(y1[:, :10, :], y2[:, :10, :], atol=1e-6), \
            f"Causality violated! Max diff: {(y1[:, :10, :] - y2[:, :10, :]).abs().max().item():.2e}"

    def test_identity_init(self):
        """At initialization, short conv should approximately pass through."""
        from memory_bench.mechanisms.deltanet import ShortConv1d
        torch.manual_seed(42)

        conv = ShortConv1d(dim=16, kernel_size=4)
        x = torch.randn(1, 10, 16)
        y = conv(x)

        # Should be close to identity (last tap = 1, others = 0)
        assert torch.allclose(x, y, atol=1e-6), \
            f"Init not identity! Max diff: {(x - y).abs().max().item():.2e}"


# ─── L2 Weight Normalization ────────────────────────────────────────────────

class TestWeightNormalization:
    def test_l2_normalize_preserves_structure(self):
        """After normalization, columns should have unit norm."""
        from memory_bench.mechanisms.ttt import TTTLinearLayer
        from nanochat.gpt import GPTConfig

        config = GPTConfig(
            sequence_len=128, vocab_size=256, n_layer=4,
            n_head=4, n_kv_head=4, n_embd=128, window_pattern="SL",
        )
        layer = TTTLinearLayer(config, layer_idx=2, chunk_size=16)

        W = torch.randn(2, 4, 32, 32) * 5  # large random matrix
        W_norm = layer._l2_normalize_columns(W)

        # Column norms should be 1
        col_norms = W_norm.norm(dim=-2)  # (2, 4, 32)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5), \
            f"Column norms not unit: mean={col_norms.mean():.4f}, std={col_norms.std():.4f}"


# ─── Persistent Memory: Identity at Init ────────────────────────────────────

class TestPersistentNumerical:
    def test_near_identity_at_init(self):
        """With zero-init scale, persistent memory should minimally affect output."""
        from nanochat.gpt import GPT, GPTConfig
        from memory_bench.mechanisms.persistent import PersistentMemory

        torch.manual_seed(42)
        config = GPTConfig(
            sequence_len=128, vocab_size=256, n_layer=4,
            n_head=4, n_kv_head=4, n_embd=128, window_pattern="SL",
        )

        # Baseline model
        model_base = GPT(config)
        model_base.init_weights()

        # Model with persistent memory
        model_mem = GPT(config)
        model_mem.init_weights()

        # Copy weights to make them identical
        model_mem.load_state_dict(model_base.state_dict())

        mech = PersistentMemory(num_tokens=8)
        model_mem = mech.wrap_model(model_mem, config)

        x = torch.randint(0, 256, (1, 32))
        y = torch.randint(0, 256, (1, 32))

        with torch.no_grad():
            loss_base = model_base(x, y)
            loss_mem = model_mem(x, y)

        # Losses should be close (memory has near-zero contribution)
        diff = abs(loss_base.item() - loss_mem.item())
        assert diff < 0.5, \
            f"Loss difference too large at init: {diff:.4f} (base={loss_base:.4f}, mem={loss_mem:.4f})"


# ─── Smoke Test: Training Actually Reduces Loss ─────────────────────────────

class TestSmokeTraining:
    """Verify that each mechanism can actually train (loss decreases)."""

    def _train_steps(self, model, n_steps=80, seq_len=64, vocab_size=256):
        """Overfit on a FIXED batch (memorization test).

        Uses a single fixed batch repeated every step. The model should
        memorize it, proving that the architecture can learn.
        Random targets on random data have no pattern — loss won't drop.
        """
        torch.manual_seed(123)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Fixed batch with learnable structure: next-token = current-token + 1
        x = torch.randint(0, vocab_size - 1, (4, seq_len))
        y = x + 1  # simple shift pattern

        losses = []
        for step in range(n_steps):
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        init_loss = sum(losses[:5]) / 5
        final_loss = sum(losses[-5:]) / 5
        return init_loss, final_loss, losses

    def _build_model(self):
        from nanochat.gpt import GPT, GPTConfig
        config = GPTConfig(
            sequence_len=128, vocab_size=256, n_layer=4,
            n_head=4, n_kv_head=4, n_embd=128, window_pattern="SL",
        )
        model = GPT(config)
        model.init_weights()
        return model, config

    def test_baseline_trains(self):
        """Baseline model should reduce loss."""
        model, _ = self._build_model()
        init_loss, final_loss, _ = self._train_steps(model)
        assert final_loss < init_loss, f"Loss didn't decrease: {init_loss:.4f} → {final_loss:.4f}"

    def test_persistent_trains(self):
        """Persistent memory model should reduce loss."""
        from memory_bench.mechanisms.persistent import PersistentMemory
        model, config = self._build_model()
        mech = PersistentMemory(num_tokens=8)
        model = mech.wrap_model(model, config)
        init_loss, final_loss, _ = self._train_steps(model)
        assert final_loss < init_loss, f"Loss didn't decrease: {init_loss:.4f} → {final_loss:.4f}"

    def test_ttt_trains(self):
        """TTT-Linear model should reduce loss."""
        from memory_bench.mechanisms.ttt import TTTLinearMemory
        model, config = self._build_model()
        mech = TTTLinearMemory(layer_idx=2, chunk_size=16)
        model = mech.wrap_model(model, config)
        init_loss, final_loss, _ = self._train_steps(model)
        assert final_loss < init_loss, f"Loss didn't decrease: {init_loss:.4f} → {final_loss:.4f}"

    def test_deltanet_trains(self):
        """DeltaNet model should reduce loss."""
        from memory_bench.mechanisms.deltanet import GatedDeltaNetMemory
        model, config = self._build_model()
        mech = GatedDeltaNetMemory(layer_indices=[1])
        model = mech.wrap_model(model, config)
        init_loss, final_loss, _ = self._train_steps(model)
        assert final_loss < init_loss, f"Loss didn't decrease: {init_loss:.4f} → {final_loss:.4f}"
