"""
Tests for synthetic evaluation tasks.

Verifies that prompt generation and evaluation logic work correctly
without requiring a trained model or GPU.

Run with: pytest tests/test_synthetic.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanochat"))

import random
import math
import torch
import torch.nn.functional as F
import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Minimal tokenizer stub (avoids needing sentencepiece / real tokenizer)
# ─────────────────────────────────────────────────────────────────────────────

class StubTokenizer:
    """Minimal tokenizer for testing prompt generation.

    Encodes each character as its ordinal. Good enough for verifying
    that the prompt structure is correct.
    """
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids if 0 <= i < 128)

    def get_bos_token_id(self) -> int:
        return 1

    def get_vocab_size(self) -> int:
        return 256


# ─────────────────────────────────────────────────────────────────────────────
# Associative Recall
# ─────────────────────────────────────────────────────────────────────────────

class TestAssociativeRecallPrompt:
    def test_prompt_structure(self):
        """Prompt should contain all KV pairs and a query."""
        from memory_bench.eval.synthetic import generate_assoc_recall_prompt

        tok = StubTokenizer()
        rng = random.Random(42)
        tokens, expected = generate_assoc_recall_prompt(
            tok, num_pairs=4, query_idx=2, rng=rng,
            with_filler=False, filler_tokens=0,
        )

        # Should start with BOS
        assert tokens[0] == 1

        # Should have non-empty expected answer
        assert len(expected) == 4  # 4-digit number
        assert expected.isdigit()

    def test_different_queries_different_answers(self):
        """Different query indices should produce different expected answers."""
        from memory_bench.eval.synthetic import generate_assoc_recall_prompt

        tok = StubTokenizer()
        answers = []
        for q_idx in range(4):
            rng = random.Random(42)  # same seed = same pairs
            _, expected = generate_assoc_recall_prompt(
                tok, num_pairs=4, query_idx=q_idx, rng=rng,
                with_filler=False,
            )
            answers.append(expected)

        # At least some should differ (all 4 pairs have different values)
        assert len(set(answers)) == 4, f"All queries returned same answer: {answers}"

    def test_filler_adds_tokens(self):
        """With filler enabled, prompt should be longer."""
        from memory_bench.eval.synthetic import generate_assoc_recall_prompt

        tok = StubTokenizer()
        rng1 = random.Random(42)
        tokens_no_fill, _ = generate_assoc_recall_prompt(
            tok, num_pairs=4, query_idx=0, rng=rng1,
            with_filler=False,
        )

        rng2 = random.Random(42)
        tokens_fill, _ = generate_assoc_recall_prompt(
            tok, num_pairs=4, query_idx=0, rng=rng2,
            with_filler=True, filler_tokens=50,
        )

        assert len(tokens_fill) > len(tokens_no_fill)

    def test_kv_pairs_unique(self):
        """Generated KV pairs should have unique keys and values."""
        from memory_bench.eval.synthetic import _generate_kv_pairs

        tok = StubTokenizer()
        rng = random.Random(42)
        pairs = _generate_kv_pairs(tok, num_pairs=16, rng=rng)

        keys = [k for k, v in pairs]
        values = [v for k, v in pairs]
        assert len(set(keys)) == 16, "Keys not unique"
        # Values are random 4-digit numbers, very likely unique for 16 pairs
        assert len(set(values)) >= 14, "Too many duplicate values"


# ─────────────────────────────────────────────────────────────────────────────
# BPB by Position
# ─────────────────────────────────────────────────────────────────────────────

class TestBPBByPosition:
    def _make_model_and_loader(self, seq_len=64, vocab_size=256, n_batches=5):
        """Create a simple model and fake dataloader for testing."""
        from nanochat.gpt import GPT, GPTConfig

        config = GPTConfig(
            sequence_len=seq_len, vocab_size=vocab_size, n_layer=2,
            n_head=4, n_kv_head=4, n_embd=64, window_pattern="SL",
        )
        torch.manual_seed(42)
        model = GPT(config)
        model.init_weights()

        # Fake dataloader: list of (x, y) tuples
        batches = []
        for _ in range(n_batches):
            x = torch.randint(0, vocab_size, (2, seq_len))
            y = torch.randint(0, vocab_size, (2, seq_len))
            batches.append((x, y))

        return model, batches, config

    def test_output_structure(self):
        """Should return dict with correct keys and bucket count."""
        from memory_bench.eval.synthetic import evaluate_bpb_by_position

        model, batches, config = self._make_model_and_loader()
        result = evaluate_bpb_by_position(
            model, batches, token_bytes=None,
            num_steps=3, num_buckets=8,
        )

        assert result["task"] == "bpb_by_position"
        assert result["sequence_length"] == 64
        assert result["num_buckets"] <= 8
        assert "buckets" in result
        assert "overall_mean_bpb" in result
        assert result["overall_mean_bpb"] > 0

    def test_buckets_cover_full_sequence(self):
        """Buckets should cover all positions."""
        from memory_bench.eval.synthetic import evaluate_bpb_by_position

        model, batches, config = self._make_model_and_loader()
        result = evaluate_bpb_by_position(
            model, batches, token_bytes=None,
            num_steps=3, num_buckets=4,
        )

        # Check that buckets span the full sequence
        positions_covered = set()
        for key, data in result["buckets"].items():
            start, end = map(int, key.split("-"))
            positions_covered.update(range(start, end))

        assert len(positions_covered) == 64

    def test_bpb_values_reasonable(self):
        """BPB should be positive and in a reasonable range for random model."""
        from memory_bench.eval.synthetic import evaluate_bpb_by_position

        model, batches, config = self._make_model_and_loader()
        result = evaluate_bpb_by_position(
            model, batches, token_bytes=None,
            num_steps=3, num_buckets=4,
        )

        for key, data in result["buckets"].items():
            bpb = data["bpb"]
            # Random model: ~log2(256) = 8 bits per token
            assert 0 < bpb < 20, f"BPB {bpb:.2f} out of reasonable range at bucket {key}"

    def test_with_token_bytes(self):
        """Should work with token_bytes tensor (BPB conversion)."""
        from memory_bench.eval.synthetic import evaluate_bpb_by_position

        model, batches, config = self._make_model_and_loader()
        # Fake token_bytes: each token is ~3 bytes on average
        token_bytes = torch.full((256,), 3, dtype=torch.long)

        result = evaluate_bpb_by_position(
            model, batches, token_bytes=token_bytes,
            num_steps=3, num_buckets=4,
        )

        # With 3 bytes/token, BPB should be ~1/3 of bits-per-token
        assert result["overall_mean_bpb"] > 0

    def test_handles_ignore_index(self):
        """Should correctly handle y=-1 (padding tokens)."""
        from memory_bench.eval.synthetic import evaluate_bpb_by_position
        from nanochat.gpt import GPT, GPTConfig

        config = GPTConfig(
            sequence_len=32, vocab_size=256, n_layer=2,
            n_head=4, n_kv_head=4, n_embd=64, window_pattern="SL",
        )
        torch.manual_seed(42)
        model = GPT(config)
        model.init_weights()

        # Half the targets are padding
        x = torch.randint(0, 256, (2, 32))
        y = torch.randint(0, 256, (2, 32))
        y[:, 16:] = -1  # padding

        result = evaluate_bpb_by_position(
            model, [(x, y)], token_bytes=None,
            num_steps=1, num_buckets=4,
        )

        # Should not crash and should produce valid output
        assert result["overall_mean_bpb"] > 0

    def test_dataloader_2tuple_works(self):
        """Dataloader yielding (x, y) tuples (not triples) should work."""
        from memory_bench.eval.synthetic import evaluate_bpb_by_position

        model, batches, config = self._make_model_and_loader()
        # batches are already (x, y) tuples — this is the bug we fixed
        result = evaluate_bpb_by_position(
            model, batches, token_bytes=None,
            num_steps=2, num_buckets=4,
        )
        assert result["task"] == "bpb_by_position"

    def test_dataloader_3tuple_works(self):
        """Dataloader yielding (x, y, extra) triples should also work."""
        from memory_bench.eval.synthetic import evaluate_bpb_by_position

        model, batches_2, config = self._make_model_and_loader()
        # Wrap as 3-tuples
        batches_3 = [(x, y, torch.zeros(1)) for x, y in batches_2]

        result = evaluate_bpb_by_position(
            model, batches_3, token_bytes=None,
            num_steps=2, num_buckets=4,
        )
        assert result["task"] == "bpb_by_position"


# ─────────────────────────────────────────────────────────────────────────────
# Variable shadowing regression
# ─────────────────────────────────────────────────────────────────────────────

class TestNoVariableShadow:
    def test_np_not_shadowed(self):
        """The variable 'np' should not shadow numpy in any function."""
        from memory_bench.eval import synthetic
        # If np were shadowed, this would fail
        assert synthetic.np is np
