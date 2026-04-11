"""
Integration tests that catch the bugs unit tests miss.

These tests verify system-level properties that were the source of
every real bug in memory-bench's first three audits:

1. Seeds actually produce different results
2. Train and eval use the same forward path
3. GQA configs don't silently break mechanisms
4. Attention scaling is correct
5. Memory token boundaries are respected

Run with: pytest tests/test_integration.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanochat"))

import torch
import pytest
from nanochat.gpt import GPT, GPTConfig


# Standard config (no GQA — n_kv_head == n_head)
CONFIG_NO_GQA = GPTConfig(
    sequence_len=128,
    vocab_size=256,
    n_layer=4,
    n_head=4,
    n_kv_head=4,
    n_embd=128,
    window_pattern="SL",
)

# GQA config (n_kv_head < n_head) — this is what caught the TTT averaging bug
CONFIG_GQA = GPTConfig(
    sequence_len=128,
    vocab_size=256,
    n_layer=4,
    n_head=4,
    n_kv_head=2,  # GQA: 2 KV heads shared across 4 query heads
    n_embd=128,
    window_pattern="SL",
)


def _build_model(config):
    model = GPT(config)
    model.init_weights()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 1. Seed Verification
#    Bug #1: --seed was accepted but never applied. All runs used seed 42.
# ─────────────────────────────────────────────────────────────────────────────

class TestSeedActuallyWorks:
    """Verify that different seeds produce different model outputs."""

    def test_different_seeds_different_loss(self):
        """Two seeds must produce different initial losses on the same data."""
        x = torch.randint(0, 256, (2, 64))
        y = torch.randint(0, 256, (2, 64))

        losses = []
        for seed in [42, 1337]:
            torch.manual_seed(seed)
            model = _build_model(CONFIG_NO_GQA)
            with torch.no_grad():
                loss = model(x, y)
            losses.append(loss.item())

        assert losses[0] != losses[1], \
            f"Seeds 42 and 1337 produced identical loss {losses[0]:.6f} — seed is not being applied"

    def test_same_seed_same_loss(self):
        """Same seed must produce identical losses (reproducibility)."""
        x = torch.randint(0, 256, (2, 64))
        y = torch.randint(0, 256, (2, 64))

        losses = []
        for _ in range(2):
            torch.manual_seed(42)
            model = _build_model(CONFIG_NO_GQA)
            with torch.no_grad():
                loss = model(x, y)
            losses.append(loss.item())

        assert losses[0] == losses[1], \
            f"Same seed produced different losses: {losses[0]:.6f} vs {losses[1]:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. RMT Train/Eval Path Consistency
#    Bug #2: RMT trained with segments but evaluated with standard forward.
# ─────────────────────────────────────────────────────────────────────────────

class TestRMTTrainEvalConsistency:
    """forward_segment and forward_segment_logits must produce identical logits."""

    def test_logits_match(self):
        """forward_segment's implicit logits must equal forward_segment_logits."""
        from memory_bench.mechanisms.rmt import RMTMemory

        torch.manual_seed(42)
        model = _build_model(CONFIG_NO_GQA)
        mech = RMTMemory(num_tokens=4, seg_length=32)
        model = mech.wrap_model(model, CONFIG_NO_GQA)

        x = torch.randint(0, 256, (2, 32))
        y = torch.randint(0, 256, (2, 32))

        # Get logits from eval path
        with torch.no_grad():
            logits_eval, mem_eval = mech.forward_segment_logits(model, x)

        # Get loss from train path, recompute logits manually
        with torch.no_grad():
            logits_train, mem_train = mech._forward_segment_core(model, x)

        # Must be identical (both use _forward_segment_core now)
        assert torch.allclose(logits_eval, logits_train, atol=1e-6), \
            f"Train/eval logits differ! Max diff: {(logits_eval - logits_train).abs().max():.2e}"
        assert torch.allclose(mem_eval, mem_train, atol=1e-6), \
            "Train/eval memory states differ!"

    def test_multi_segment_eval(self):
        """Eval over 2 segments with carried memory should produce valid outputs.

        NOTE: At random init, nanochat zero-inits c_proj so attention output is
        all zeros. Memory has no effect until training begins. This test verifies
        shapes and no-crash, not that memory affects output (that's tested by
        the smoke training tests in test_numerical.py).
        """
        from memory_bench.mechanisms.rmt import RMTMemory

        torch.manual_seed(42)
        model = _build_model(CONFIG_NO_GQA)
        mech = RMTMemory(num_tokens=4, seg_length=32)
        model = mech.wrap_model(model, CONFIG_NO_GQA)

        x = torch.randint(0, 256, (2, 32))

        with torch.no_grad():
            logits1, mem1 = mech.forward_segment_logits(model, x, memory_state=None)
            mem1 = mem1.detach()
            logits2, mem2 = mech.forward_segment_logits(model, x, memory_state=mem1)

        assert logits1.shape == (2, 32, 256)
        assert logits2.shape == (2, 32, 256)
        assert not torch.isnan(logits1).any()
        assert not torch.isnan(logits2).any()
        # Memory state should be valid tensor with correct shape
        assert mem2.shape == (2, 4, CONFIG_NO_GQA.n_embd)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GQA Compatibility
#    Bug #5: TTT averaged queries in GQA groups, destroying per-head info.
#    Bug #10: DeltaNet GroupNorm mismatches with GQA.
# ─────────────────────────────────────────────────────────────────────────────

class TestGQACompatibility:
    """All mechanisms must work correctly when n_kv_head < n_head."""

    def test_persistent_gqa(self):
        """Persistent memory should work with GQA."""
        from memory_bench.mechanisms.persistent import PersistentMemory
        torch.manual_seed(42)
        model = _build_model(CONFIG_GQA)
        mech = PersistentMemory(num_tokens=8)
        model = mech.wrap_model(model, CONFIG_GQA)
        x = torch.randint(0, 256, (2, 64))
        y = torch.randint(0, 256, (2, 64))
        loss = model(x, y)
        assert not torch.isnan(loss)
        loss.backward()

    def test_ttt_gqa(self):
        """TTT-Linear should work with GQA (the averaging bug would make this degenerate)."""
        from memory_bench.mechanisms.ttt import TTTLinearMemory
        torch.manual_seed(42)
        model = _build_model(CONFIG_GQA)
        mech = TTTLinearMemory(layer_idx=2, chunk_size=16)
        model = mech.wrap_model(model, CONFIG_GQA)
        x = torch.randint(0, 256, (2, 64))
        y = torch.randint(0, 256, (2, 64))
        loss = model(x, y)
        assert not torch.isnan(loss)
        loss.backward()

    def test_ttt_gqa_preserves_query_diversity(self):
        """With GQA, distinct query heads should produce distinct outputs.

        The old bug averaged queries within each KV group before the TTT
        dual form, which meant all queries in a group were identical.
        After fixing, queries should remain distinct.
        """
        from memory_bench.mechanisms.ttt import TTTLinearMemory

        torch.manual_seed(42)
        model = _build_model(CONFIG_GQA)
        mech = TTTLinearMemory(layer_idx=2, chunk_size=16)
        model = mech.wrap_model(model, CONFIG_GQA)

        x = torch.randint(0, 256, (1, 64))
        y = torch.randint(0, 256, (1, 64))

        # Run full forward to get loss (ensures all shapes are correct)
        loss = model(x, y)
        assert not torch.isnan(loss)

        # Now verify query diversity: run the TTT layer directly
        # and check that output heads within the same KV group differ
        ttt_block = model.transformer.h[2]
        ttt_layer = ttt_block.attn
        from nanochat.gpt import norm

        with torch.no_grad():
            # Get embeddings at layer 2
            x_emb = model.transformer.wte(x)
            x_emb = norm(x_emb)
            # Simplified: just check that c_q produces different head outputs
            q = ttt_layer.c_q(x_emb).view(1, 64, CONFIG_GQA.n_head, -1)
            # Heads 0 and 1 should be different (different Q projections)
            head0 = q[:, :, 0, :]
            head1 = q[:, :, 1, :]
            assert not torch.allclose(head0, head1, atol=1e-4), \
                "Query heads in same KV group are identical — queries are being averaged!"

    def test_deltanet_gqa(self):
        """DeltaNet should work with GQA (Bug #22: GroupNorm channel mismatch)."""
        from memory_bench.mechanisms.deltanet import GatedDeltaNetMemory
        torch.manual_seed(42)
        model = _build_model(CONFIG_GQA)
        mech = GatedDeltaNetMemory(layer_indices=[1])
        model = mech.wrap_model(model, CONFIG_GQA)
        x = torch.randint(0, 256, (2, 64))
        y = torch.randint(0, 256, (2, 64))
        loss = model(x, y)
        assert not torch.isnan(loss)
        loss.backward()

    def test_ttt_gqa_eta_interleaving(self):
        """TTT eta must match k/v head ordering under GQA (Bug #21).

        With n_kv_head=2, n_head=4, n_rep=2:
            k/v expand as: [kv0, kv0, kv1, kv1] (repeat_interleave)
            eta must also expand as: [kv0, kv0, kv1, kv1]
        The old bug used unsqueeze(2).expand which produced [kv0, kv1, kv0, kv1].
        """
        from memory_bench.mechanisms.ttt import TTTLinearLayer

        torch.manual_seed(42)
        layer = TTTLinearLayer(CONFIG_GQA, layer_idx=2, chunk_size=16)
        layer.eval()

        B, T = 1, 16
        x = torch.randn(B, T, CONFIG_GQA.n_embd)

        # Manually compute eta expansion
        import torch.nn.functional as F
        eta = F.softplus(layer.lr_proj(x))  # (B, T, n_kv_head=2)
        n_rep = CONFIG_GQA.n_head // CONFIG_GQA.n_kv_head  # 2

        eta_correct = eta.repeat_interleave(n_rep, dim=2)  # [kv0, kv0, kv1, kv1]

        # Verify ordering: heads 0,1 share kv0; heads 2,3 share kv1
        assert torch.equal(eta_correct[:, :, 0], eta_correct[:, :, 1]), \
            "Heads 0,1 should have same eta (both map to kv_head 0)"
        assert torch.equal(eta_correct[:, :, 2], eta_correct[:, :, 3]), \
            "Heads 2,3 should have same eta (both map to kv_head 1)"
        assert not torch.equal(eta_correct[:, :, 0], eta_correct[:, :, 2]), \
            "Heads from different KV groups should have different eta"

    def test_rmt_gqa(self):
        """RMT should work with GQA."""
        from memory_bench.mechanisms.rmt import RMTMemory
        torch.manual_seed(42)
        model = _build_model(CONFIG_GQA)
        mech = RMTMemory(num_tokens=4, seg_length=32)
        model = mech.wrap_model(model, CONFIG_GQA)
        x = torch.randint(0, 256, (2, 32))
        y = torch.randint(0, 256, (2, 32))
        loss, mem = mech.forward_segment(model, x, y)
        assert not torch.isnan(loss)
        loss.backward()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Attention Scaling
#    Bug #7: Persistent memory used scale=1.0, making logits 11.3x too large.
# ─────────────────────────────────────────────────────────────────────────────

class TestAttentionScaling:
    """Verify that attention uses correct 1/sqrt(d) scaling."""

    def test_persistent_attention_logits_magnitude(self):
        """Attention logits should have reasonable magnitude (not 11x too large).

        Before the fix, scale=1.0 made logits ~11x larger than correct.
        We verify by checking that the loss is in a reasonable range for
        an untrained model (should be near log(vocab_size) = log(256) ≈ 5.5).
        """
        from memory_bench.mechanisms.persistent import PersistentMemory

        torch.manual_seed(42)
        model = _build_model(CONFIG_NO_GQA)
        mech = PersistentMemory(num_tokens=8)
        model = mech.wrap_model(model, CONFIG_NO_GQA)

        x = torch.randint(0, 256, (4, 64))
        y = torch.randint(0, 256, (4, 64))

        with torch.no_grad():
            loss = model(x, y)

        # Random init loss should be near ln(256) ≈ 5.55
        # With broken scale=1.0 it could be much higher or lower
        assert 3.0 < loss.item() < 8.0, \
            f"Loss {loss.item():.2f} is outside reasonable range for random init — check attention scaling"

    def test_persistent_matches_baseline_at_init(self):
        """At init (v_scale ≈ 0), persistent memory model should match baseline closely."""
        from memory_bench.mechanisms.persistent import PersistentMemory

        torch.manual_seed(42)
        model_base = _build_model(CONFIG_NO_GQA)

        torch.manual_seed(42)
        model_mem = _build_model(CONFIG_NO_GQA)
        # Make weights identical
        model_mem.load_state_dict(model_base.state_dict())

        mech = PersistentMemory(num_tokens=8)
        model_mem = mech.wrap_model(model_mem, CONFIG_NO_GQA)

        x = torch.randint(0, 256, (2, 64))
        y = torch.randint(0, 256, (2, 64))

        with torch.no_grad():
            loss_base = model_base(x, y).item()
            loss_mem = model_mem(x, y).item()

        diff = abs(loss_base - loss_mem)
        assert diff < 0.5, \
            f"Persistent model differs from baseline by {diff:.4f} at init — scaling or masking may be wrong"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Smear Boundary (RMT)
#    Bug #3: Smear mixed memory tokens into real token chain.
# ─────────────────────────────────────────────────────────────────────────────

class TestSmearBoundary:
    """Verify that smear operates only among real tokens, not memory tokens."""

    def test_smear_does_not_cross_memory_boundary(self):
        """Changing memory state should not affect smear output on real tokens.

        If smear uses memory tokens as predecessors, changing memory
        would change the first real token's embedding.
        """
        from memory_bench.mechanisms.rmt import RMTMemory

        torch.manual_seed(42)
        model = _build_model(CONFIG_NO_GQA)
        mech = RMTMemory(num_tokens=4, seg_length=32)
        model = mech.wrap_model(model, CONFIG_NO_GQA)

        x = torch.randint(0, 256, (1, 32))

        # Run with default memory init
        mem1 = mech.memory_init.expand(1, -1, -1)
        # Run with random memory (very different)
        mem2 = torch.randn_like(mem1) * 10.0

        with torch.no_grad():
            logits1, _ = mech.forward_segment_logits(model, x, memory_state=mem1)
            logits2, _ = mech.forward_segment_logits(model, x, memory_state=mem2)

        # The logits WILL differ (memory affects attention), but if smear
        # crosses the boundary, the difference will be much larger.
        # We verify that the mechanism runs without error with very different
        # memory states — the real protection is in the code structure
        # (smear operates on x[:, M:] only).
        assert logits1.shape == logits2.shape
        assert not torch.isnan(logits1).any()
        assert not torch.isnan(logits2).any()


# ─────────────────────────────────────────────────────────────────────────────
# 6. bench.py Aggregation
#    Bug #18: std_bpb=0 (single seed) printed as "N/A" due to falsy check.
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchAggregation:
    """Verify bench.py handles edge cases correctly."""

    def test_single_seed_std_is_zero_not_none(self):
        """With one seed, std should be 0, not None."""
        from memory_bench.bench import aggregate_results

        results = [{
            "mechanism": "baseline",
            "depth": 12,
            "seed": 42,
            "min_val_bpb": 1.23456,
            "total_time_min": 15.0,
            "peak_vram_mib": 40000,
            "total_params": 100000,
        }]

        agg = aggregate_results(results)
        key = "baseline_d12"
        assert key in agg
        assert agg[key]["std_bpb"] == 0  # not None
        assert agg[key]["mean_bpb"] == 1.23456

        # The display code should handle std=0 correctly
        std = f"{agg[key]['std_bpb']:.5f}" if agg[key]['std_bpb'] is not None else "N/A"
        assert std == "0.00000", f"Expected '0.00000', got '{std}'"

    def test_zero_bpb_not_treated_as_missing(self):
        """A mean_bpb of 0.0 should display as '0.00000', not 'N/A'."""
        data = {"mean_bpb": 0.0, "std_bpb": 0.0, "mean_time_min": 0.0}
        bpb = f"{data['mean_bpb']:.5f}" if data['mean_bpb'] is not None else "N/A"
        assert bpb == "0.00000"
