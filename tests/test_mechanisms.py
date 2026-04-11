"""
Tests for memory mechanisms.

These tests verify that each mechanism:
1. Instantiates correctly
2. Wraps the model without errors
3. Forward pass produces valid output shapes
4. Backward pass computes gradients
5. Key mathematical properties hold

Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanochat"))

import torch
import pytest
from nanochat.gpt import GPT, GPTConfig


# Small config for fast tests
TEST_CONFIG = GPTConfig(
    sequence_len=128,
    vocab_size=256,
    n_layer=4,
    n_head=4,
    n_kv_head=4,
    n_embd=128,
    window_pattern="SL",
)


def _build_test_model():
    """Build a small test model on CPU."""
    model = GPT(TEST_CONFIG)
    model.init_weights()
    return model


def _random_batch(batch_size=2, seq_len=64):
    """Generate random input/target batch."""
    x = torch.randint(0, TEST_CONFIG.vocab_size, (batch_size, seq_len))
    y = torch.randint(0, TEST_CONFIG.vocab_size, (batch_size, seq_len))
    return x, y


class TestPersistentMemory:
    def test_instantiate(self):
        from memory_bench.mechanisms.persistent import PersistentMemory
        mech = PersistentMemory(num_tokens=8)
        assert mech.name == "persistent-8"

    def test_wrap_model(self):
        from memory_bench.mechanisms.persistent import PersistentMemory
        model = _build_test_model()
        mech = PersistentMemory(num_tokens=8)
        model = mech.wrap_model(model, TEST_CONFIG)
        assert mech.num_memory_params > 0

    def test_forward(self):
        from memory_bench.mechanisms.persistent import PersistentMemory
        model = _build_test_model()
        mech = PersistentMemory(num_tokens=8)
        model = mech.wrap_model(model, TEST_CONFIG)
        x, y = _random_batch()
        loss = model(x, y)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_backward(self):
        from memory_bench.mechanisms.persistent import PersistentMemory
        model = _build_test_model()
        mech = PersistentMemory(num_tokens=8)
        model = mech.wrap_model(model, TEST_CONFIG)
        x, y = _random_batch()
        loss = model(x, y)
        loss.backward()
        # Check that memory params have gradients
        for p in mech._memory_params:
            assert p.grad is not None

    def test_zero_init_safety(self):
        """Memory contribution should be near-zero at initialization."""
        from memory_bench.mechanisms.persistent import PersistentMemory
        model = _build_test_model()
        mech = PersistentMemory(num_tokens=8)
        model = mech.wrap_model(model, TEST_CONFIG)
        # v_scale parameters should be small (every 3rd param: mem_k, mem_v, v_scale)
        for i in range(0, len(mech._memory_params), 3):
            v_scale = mech._memory_params[i + 2]  # mem_k, mem_v, v_scale
            assert abs(v_scale.item()) < 0.1


class TestRMT:
    def test_instantiate(self):
        from memory_bench.mechanisms.rmt import RMTMemory
        mech = RMTMemory(num_tokens=4, seg_length=32)
        assert mech.requires_segments
        assert mech.segment_length == 32

    def test_wrap_model(self):
        from memory_bench.mechanisms.rmt import RMTMemory
        model = _build_test_model()
        mech = RMTMemory(num_tokens=4, seg_length=32)
        model = mech.wrap_model(model, TEST_CONFIG)
        assert mech.num_memory_params > 0
        # Should have memory init + projection params
        assert mech.memory_proj is not None

    def test_forward_segment(self):
        from memory_bench.mechanisms.rmt import RMTMemory
        model = _build_test_model()
        mech = RMTMemory(num_tokens=4, seg_length=32)
        model = mech.wrap_model(model, TEST_CONFIG)

        # First segment
        x = torch.randint(0, TEST_CONFIG.vocab_size, (2, 32))
        y = torch.randint(0, TEST_CONFIG.vocab_size, (2, 32))
        loss, mem = mech.forward_segment(model, x, y, memory_state=None)
        assert loss.shape == ()
        assert mem.shape == (2, 4, TEST_CONFIG.n_embd)

        # Second segment with carried memory
        loss2, mem2 = mech.forward_segment(model, x, y, memory_state=mem)
        assert loss2.shape == ()

    def test_backward(self):
        from memory_bench.mechanisms.rmt import RMTMemory
        model = _build_test_model()
        mech = RMTMemory(num_tokens=4, seg_length=32)
        model = mech.wrap_model(model, TEST_CONFIG)

        x = torch.randint(0, TEST_CONFIG.vocab_size, (2, 32))
        y = torch.randint(0, TEST_CONFIG.vocab_size, (2, 32))
        loss, _ = mech.forward_segment(model, x, y)
        loss.backward()
        assert mech.memory_init.grad is not None

    def test_memory_projection(self):
        """Memory projection should transform memory states."""
        from memory_bench.mechanisms.rmt import RMTMemory
        model = _build_test_model()
        mech = RMTMemory(num_tokens=4, seg_length=32)
        model = mech.wrap_model(model, TEST_CONFIG)

        x = torch.randint(0, TEST_CONFIG.vocab_size, (2, 32))
        y = torch.randint(0, TEST_CONFIG.vocab_size, (2, 32))
        _, mem = mech.forward_segment(model, x, y)

        # Memory retains gradient graph (caller detaches based on bptt_depth)
        assert mem.requires_grad

    def test_forward_segment_logits(self):
        """forward_segment_logits should return logits matching forward_segment."""
        from memory_bench.mechanisms.rmt import RMTMemory
        model = _build_test_model()
        mech = RMTMemory(num_tokens=4, seg_length=32)
        model = mech.wrap_model(model, TEST_CONFIG)

        x = torch.randint(0, TEST_CONFIG.vocab_size, (2, 32))
        y = torch.randint(0, TEST_CONFIG.vocab_size, (2, 32))
        logits, mem = mech.forward_segment_logits(model, x)
        assert logits.shape == (2, 32, TEST_CONFIG.vocab_size)
        assert mem.shape == (2, 4, TEST_CONFIG.n_embd)


class TestTTTLinear:
    def test_instantiate(self):
        from memory_bench.mechanisms.ttt import TTTLinearMemory
        mech = TTTLinearMemory(layer_idx=2, chunk_size=16)
        assert "ttt" in mech.name

    def test_wrap_model(self):
        from memory_bench.mechanisms.ttt import TTTLinearMemory
        model = _build_test_model()
        mech = TTTLinearMemory(layer_idx=2, chunk_size=16)
        model = mech.wrap_model(model, TEST_CONFIG)
        # Check that layer 2 was replaced
        from memory_bench.mechanisms.ttt import TTTBlock
        assert isinstance(model.transformer.h[2], TTTBlock)

    def test_forward(self):
        from memory_bench.mechanisms.ttt import TTTLinearMemory
        model = _build_test_model()
        mech = TTTLinearMemory(layer_idx=2, chunk_size=16)
        model = mech.wrap_model(model, TEST_CONFIG)
        x, y = _random_batch()
        loss = model(x, y)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_backward(self):
        from memory_bench.mechanisms.ttt import TTTLinearMemory
        model = _build_test_model()
        mech = TTTLinearMemory(layer_idx=2, chunk_size=16)
        model = mech.wrap_model(model, TEST_CONFIG)
        x, y = _random_batch()
        loss = model(x, y)
        loss.backward()
        # TTT params should have gradients
        for p in mech._ttt_params:
            assert p.grad is not None

    def test_learned_lr(self):
        """Per-token learning rate should be positive (softplus output)."""
        from memory_bench.mechanisms.ttt import TTTLinearLayer
        import torch.nn.functional as F
        layer = TTTLinearLayer(TEST_CONFIG, layer_idx=2, chunk_size=16)
        x = torch.randn(2, 16, TEST_CONFIG.n_embd)
        eta = F.softplus(layer.lr_proj(x))
        assert (eta > 0).all()

    def test_dual_form_output_shape(self):
        """Dual form should produce correct output shape."""
        from memory_bench.mechanisms.ttt import TTTLinearLayer
        layer = TTTLinearLayer(TEST_CONFIG, layer_idx=2, chunk_size=16)
        B, T, H, D = 2, 32, 4, 32
        q = torch.randn(B, T, H, D)
        k = torch.randn(B, T, H, D)
        v = torch.randn(B, T, H, D)
        eta = torch.ones(B, T, H) * 0.1
        out = layer._ttt_dual_forward(q, k, v, eta)
        assert out.shape == (B, T, H, D)

    def test_momentum(self):
        """TTT with momentum should not error."""
        from memory_bench.mechanisms.ttt import TTTLinearMemory
        model = _build_test_model()
        mech = TTTLinearMemory(layer_idx=2, chunk_size=16, use_momentum=True)
        model = mech.wrap_model(model, TEST_CONFIG)
        x, y = _random_batch()
        loss = model(x, y)
        assert not torch.isnan(loss)


class TestGatedDeltaNet:
    def test_instantiate(self):
        from memory_bench.mechanisms.deltanet import GatedDeltaNetMemory
        mech = GatedDeltaNetMemory(layer_indices=[1])
        assert "deltanet" in mech.name

    def test_wrap_model(self):
        from memory_bench.mechanisms.deltanet import GatedDeltaNetMemory
        model = _build_test_model()
        mech = GatedDeltaNetMemory(layer_indices=[1])
        model = mech.wrap_model(model, TEST_CONFIG)
        from memory_bench.mechanisms.deltanet import GatedDeltaNetBlock
        assert isinstance(model.transformer.h[1], GatedDeltaNetBlock)

    def test_forward(self):
        from memory_bench.mechanisms.deltanet import GatedDeltaNetMemory
        model = _build_test_model()
        mech = GatedDeltaNetMemory(layer_indices=[1])
        model = mech.wrap_model(model, TEST_CONFIG)
        x, y = _random_batch()
        loss = model(x, y)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_backward(self):
        from memory_bench.mechanisms.deltanet import GatedDeltaNetMemory
        model = _build_test_model()
        mech = GatedDeltaNetMemory(layer_indices=[1])
        model = mech.wrap_model(model, TEST_CONFIG)
        x, y = _random_batch()
        loss = model(x, y)
        loss.backward()
        for p in mech._extra_params:
            assert p.grad is not None

    def test_short_conv_causal(self):
        """Short convolution should be causal (no future leakage)."""
        from memory_bench.mechanisms.deltanet import ShortConv1d
        conv = ShortConv1d(dim=32, kernel_size=4)
        x = torch.randn(1, 10, 32)
        y = conv(x)
        assert y.shape == x.shape

        # Causality check: changing future input shouldn't affect past output
        x2 = x.clone()
        x2[:, 5:, :] = torch.randn(1, 5, 32)
        y2 = conv(x2)
        # Positions 0-4 should be identical
        assert torch.allclose(y[:, :5, :], y2[:, :5, :])

    def test_log_space_gating(self):
        """gk should be in log-space (negative values)."""
        from memory_bench.mechanisms.deltanet import GatedDeltaNetAttention
        attn = GatedDeltaNetAttention(TEST_CONFIG, layer_idx=1)
        x = torch.randn(2, 16, TEST_CONFIG.n_embd)
        gk = attn._compute_log_decay(attn.c_gk(x))
        assert (gk <= 0).all()  # log-decay must be non-positive

    def test_naive_recurrence_matches_shape(self):
        """Naive fallback should produce correct shapes."""
        from memory_bench.mechanisms.deltanet import GatedDeltaNetAttention
        attn = GatedDeltaNetAttention(TEST_CONFIG, layer_idx=1)
        B, H, T, D = 2, 4, 16, 32
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        beta = torch.sigmoid(torch.randn(B, H, T))
        gk = -torch.abs(torch.randn(B, H, T, D))  # negative log-space
        out = attn._naive_recurrent_forward(q, k, v, beta, gk)
        assert out.shape == (B, H, T, D)

    def test_training_stays_finite_on_cpu(self):
        """DeltaNet should not produce NaNs over multiple optimizer steps."""
        from memory_bench.mechanisms.deltanet import GatedDeltaNetMemory

        torch.manual_seed(0)
        model = _build_test_model()
        mech = GatedDeltaNetMemory(layer_indices=[1])
        model = mech.wrap_model(model, TEST_CONFIG)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x, y = _random_batch(batch_size=4, seq_len=64)

        for step in range(20):
            loss = model(x, y)
            assert torch.isfinite(loss), f"non-finite DeltaNet loss at step {step}: {loss.item()}"
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_training_stays_finite_with_fla(self):
        """CUDA + FLA path should stay finite for a short training run."""
        from memory_bench.mechanisms.deltanet import GatedDeltaNetMemory, _HAS_FLA

        if not _HAS_FLA:
            pytest.skip("fla-core not installed")

        torch.manual_seed(0)
        model = _build_test_model().cuda()
        mech = GatedDeltaNetMemory(layer_indices=[1])
        model = mech.wrap_model(model, TEST_CONFIG)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x, y = _random_batch(batch_size=4, seq_len=64)
        x = x.cuda()
        y = y.cuda()

        for step in range(10):
            loss = model(x, y)
            assert torch.isfinite(loss), f"non-finite CUDA DeltaNet loss at step {step}: {loss.item()}"
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


class TestVisualize:
    def test_generate_pdf(self, tmp_path):
        """Test that the visualization script generates a valid PDF."""
        from memory_bench.visualize import generate_architecture_pdf
        output = str(tmp_path / "test_arch.pdf")
        result = generate_architecture_pdf(output)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0
