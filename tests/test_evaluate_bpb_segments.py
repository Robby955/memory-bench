"""
Tests for evaluate_bpb_segments() — the RMT-specific BPB evaluation.

This function is critical because it computes the reported BPB for RMT.
A bug here means all RMT BPB numbers are wrong, which would invalidate
the entire benchmark's RMT results.

The function processes validation data in segments with memory, matching
the training forward path, then computes BPB = total_nats / (ln2 * total_bytes).

Run with: pytest tests/test_evaluate_bpb_segments.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanochat"))

import math
import torch
import torch.nn.functional as F
import pytest
from nanochat.gpt import GPT, GPTConfig
from memory_bench.mechanisms.rmt import RMTMemory

try:
    from nanochat.loss_eval import evaluate_bpb
    HAS_EVALUATE_BPB = True
except ImportError:
    HAS_EVALUATE_BPB = False


TEST_CONFIG = GPTConfig(
    sequence_len=128,
    vocab_size=256,
    n_layer=4,
    n_head=4,
    n_kv_head=4,
    n_embd=128,
    window_pattern="SL",
)

SEED = 42
SEG_LEN = 32


def _build_rmt():
    torch.manual_seed(SEED)
    model = GPT(TEST_CONFIG)
    model.init_weights()
    mech = RMTMemory(num_tokens=4, seg_length=SEG_LEN)
    model = mech.wrap_model(model, TEST_CONFIG)
    return model, mech


def _make_val_batches(n_batches, batch_size=2, seq_len=64):
    """Generate fixed validation batches (x, y pairs)."""
    torch.manual_seed(SEED + 777)
    batches = []
    for _ in range(n_batches):
        x = torch.randint(0, TEST_CONFIG.vocab_size, (batch_size, seq_len))
        y = torch.randint(0, TEST_CONFIG.vocab_size, (batch_size, seq_len))
        batches.append((x, y))
    return batches


class TestEvaluateBpbSegments:
    """Tests for segment-based BPB evaluation."""

    def test_returns_finite_value(self):
        """evaluate_bpb_segments must return a finite float."""
        model, mech = _build_rmt()
        model.eval()
        token_bytes = torch.ones(TEST_CONFIG.vocab_size, dtype=torch.long)  # 1 byte per token
        batches = _make_val_batches(2, seq_len=64)

        # Import the function from train.py — it's defined inline there,
        # so we replicate the logic here for unit testing.
        bpb = _evaluate_bpb_segments(model, mech, iter(batches), 2, token_bytes, 64)

        assert math.isfinite(bpb), f"BPB is not finite: {bpb}"
        assert bpb > 0, f"BPB should be positive: {bpb}"

    def test_bpb_accounting_matches_manual(self):
        """Verify BPB = total_nats / (ln2 * total_bytes) with known inputs."""
        model, mech = _build_rmt()
        model.eval()

        # Use uniform token_bytes = 1 for easy math
        token_bytes = torch.ones(TEST_CONFIG.vocab_size, dtype=torch.long)
        batches = _make_val_batches(1, batch_size=2, seq_len=64)

        # Manual computation
        x, y = batches[0]
        n_segments = 64 // SEG_LEN
        total_nats = 0.0
        total_bytes = 0
        memory_state = None

        with torch.no_grad():
            for seg_idx in range(n_segments):
                seg_start = seg_idx * SEG_LEN
                seg_end = seg_start + SEG_LEN
                seg_x = x[:, seg_start:seg_end]
                seg_y = y[:, seg_start:seg_end]

                logits, memory_state = mech.forward_segment_logits(model, seg_x, memory_state)
                memory_state = memory_state.detach()
                memory_state = mech.on_segment_boundary(memory_state)

                loss2d = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    seg_y.reshape(-1),
                    ignore_index=-1,
                    reduction="none",
                ).view(seg_y.shape)

                y_flat = seg_y.reshape(-1)
                valid = y_flat >= 0
                # token_bytes = 1 for all tokens
                total_nats += loss2d.view(-1)[valid].sum().item()
                total_bytes += valid.sum().item()

        expected_bpb = total_nats / (math.log(2) * total_bytes)

        # Now use the function
        bpb = _evaluate_bpb_segments(
            model, mech, iter(batches), 1, token_bytes, 64
        )

        assert abs(bpb - expected_bpb) < 1e-5, \
            f"BPB mismatch: {bpb:.6f} vs manual {expected_bpb:.6f}"

    @pytest.mark.skipif(not HAS_EVALUATE_BPB, reason="evaluate_bpb not importable (missing rustbpe)")
    def test_segment_eval_reasonable_vs_unsegmented(self):
        """Segment BPB should be in the same ballpark as unsegmented BPB.

        They won't match exactly (memory tokens change attention patterns),
        but they should be within 2x of each other for a randomly initialized model.
        """
        model, mech = _build_rmt()
        model.eval()
        token_bytes = torch.ones(TEST_CONFIG.vocab_size, dtype=torch.long)
        batches = _make_val_batches(2, seq_len=64)

        seg_bpb = _evaluate_bpb_segments(model, mech, iter(batches), 2, token_bytes, 64)

        # For comparison: evaluate_bpb on a baseline model (no mechanism)
        torch.manual_seed(SEED)
        baseline = GPT(TEST_CONFIG)
        baseline.init_weights()
        baseline.eval()
        batches2 = _make_val_batches(2, seq_len=64)
        baseline_bpb = evaluate_bpb(baseline, iter(batches2), 2, token_bytes)

        # Both should be in the same order of magnitude (random init → ~ln(vocab)/ln(2))
        assert 0.5 * baseline_bpb < seg_bpb < 2.0 * baseline_bpb, \
            f"Segment BPB ({seg_bpb:.2f}) too far from baseline BPB ({baseline_bpb:.2f})"

    def test_different_seg_lengths_produce_different_bpb(self):
        """Different segment lengths should produce (slightly) different BPB.

        This verifies that the segment boundary processing is actually happening,
        not just computing standard BPB and ignoring the segmentation.
        """
        token_bytes = torch.ones(TEST_CONFIG.vocab_size, dtype=torch.long)

        # seg_length=32
        model1, mech1 = _build_rmt()
        model1.eval()
        batches1 = _make_val_batches(2, seq_len=64)
        bpb_32 = _evaluate_bpb_segments(model1, mech1, iter(batches1), 2, token_bytes, 64)

        # seg_length=16 (different segmentation)
        torch.manual_seed(SEED)
        model2 = GPT(TEST_CONFIG)
        model2.init_weights()
        mech2 = RMTMemory(num_tokens=4, seg_length=16)
        model2 = mech2.wrap_model(model2, TEST_CONFIG)
        model2.eval()
        batches2 = _make_val_batches(2, seq_len=64)
        bpb_16 = _evaluate_bpb_segments(model2, mech2, iter(batches2), 2, token_bytes, 64)

        # They should differ (different memory boundaries → different attention)
        # At init the difference may be small but non-zero
        assert bpb_32 != bpb_16, \
            f"Different seg_lengths produced identical BPB ({bpb_32:.6f}) — segmentation may not be working"


# ─────────────────────────────────────────────────────────────────────────────
# Local copy of evaluate_bpb_segments (from train.py lines 334-392)
# We replicate it here rather than importing to avoid running train.py's
# top-level argparse and DDP setup.
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate_bpb_segments(model, mechanism, batch_iter, steps, token_bytes, seq_len):
    """Local copy of evaluate_bpb_segments for unit testing."""
    total_nats = 0.0
    total_bytes = 0
    seg_len = mechanism.segment_length
    assert seq_len % seg_len == 0
    n_segments = seq_len // seg_len

    for _ in range(steps):
        x, y = next(batch_iter)
        memory_state = None
        for seg_idx in range(n_segments):
            seg_start = seg_idx * seg_len
            seg_end = seg_start + seg_len
            seg_x = x[:, seg_start:seg_end]
            seg_y = y[:, seg_start:seg_end]

            logits, memory_state = mechanism.forward_segment_logits(
                model, seg_x, memory_state
            )
            memory_state = memory_state.detach()
            memory_state = mechanism.on_segment_boundary(memory_state)

            loss2d = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                seg_y.reshape(-1),
                ignore_index=-1,
                reduction="none",
            ).view(seg_y.shape)

            y_flat = seg_y.reshape(-1)
            valid = y_flat >= 0
            y_safe = torch.where(valid, y_flat, torch.zeros_like(y_flat))
            num_bytes_flat = torch.where(
                valid, token_bytes[y_safe],
                torch.zeros_like(y_flat, dtype=token_bytes.dtype),
            )
            total_nats += (loss2d.view(-1) * (num_bytes_flat > 0)).sum().item()
            total_bytes += num_bytes_flat.sum().item()

    if total_bytes == 0:
        return float("inf")
    return total_nats / (math.log(2) * total_bytes)
