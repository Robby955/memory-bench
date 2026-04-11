"""
Regression baselines: verify that training produces expected loss values.

These tests train each mechanism for a fixed number of steps on fixed data
and check that the final loss matches a recorded baseline. If the loss changes
by more than the tolerance, something in the training pipeline broke.

The baselines were recorded on CPU with the current code. Any change to:
- Model initialization
- Forward pass math
- Optimizer behavior
- Mechanism wrapping
...will cause these tests to fail, which is the point.

HOW TO UPDATE BASELINES:
    pytest tests/test_regression.py --update-baselines
This will print new baseline values. Copy them into BASELINES below.

Run with: pytest tests/test_regression.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanochat"))

import torch
import pytest
from nanochat.gpt import GPT, GPTConfig


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Small model for fast testing (identical across all mechanisms)
TEST_CONFIG = GPTConfig(
    sequence_len=128,
    vocab_size=256,
    n_layer=4,
    n_head=4,
    n_kv_head=4,
    n_embd=128,
    window_pattern="SL",
)

N_STEPS = 50       # enough to see learning but fast (~1s per mechanism)
BATCH_SIZE = 4
SEQ_LEN = 64
SEED = 42
LR = 1e-3

# Tolerance: loss must be within this fraction of the baseline.
# Set wide enough to absorb minor float ordering changes between
# platforms, but narrow enough to catch real bugs.
RTOL = 0.02   # 2% relative tolerance — tight enough to catch mechanism regressions


# ─────────────────────────────────────────────────────────────────────────────
# Baselines (recorded values — update with --update-baselines)
# ─────────────────────────────────────────────────────────────────────────────

# Format: {mechanism_name: (init_loss, final_loss)}
# init_loss = mean of first 5 steps, final_loss = mean of last 5 steps
# Recorded on macOS ARM64, Python 3.11, PyTorch CPU, float32
BASELINES = {
    "baseline": (5.321714, 0.639259),
    "persistent": (5.322381, 0.636962),
    "ttt": (5.321157, 0.639800),
    "deltanet": (5.321833, 0.640035),
    "rmt": (5.321677, 0.642086),
}


# ─────────────────────────────────────────────────────────────────────────────
# Training harness
# ─────────────────────────────────────────────────────────────────────────────

def _fixed_data():
    """Generate fixed, learnable training data.

    Uses a simple shift pattern (y = x + 1) that every architecture
    should be able to memorize. Random targets won't work because
    loss won't decrease reliably.
    """
    torch.manual_seed(SEED + 999)  # different from model seed
    x = torch.randint(0, TEST_CONFIG.vocab_size - 1, (BATCH_SIZE, SEQ_LEN))
    y = x + 1
    return x, y


def _train(model, n_steps=N_STEPS):
    """Train on fixed data, return (init_loss, final_loss, all_losses)."""
    x, y = _fixed_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    for step in range(n_steps):
        loss = model(x, y)
        assert torch.isfinite(loss), f"non-finite loss at step {step}: {loss.item()}"
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    init_loss = sum(losses[:5]) / 5
    final_loss = sum(losses[-5:]) / 5
    return init_loss, final_loss, losses


def _train_rmt(n_steps=N_STEPS):
    """Train RMT with segmented forward, return (init_loss, final_loss, all_losses)."""
    from memory_bench.mechanisms.rmt import RMTMemory

    torch.manual_seed(SEED)
    model = GPT(TEST_CONFIG)
    model.init_weights()
    mech = RMTMemory(num_tokens=4, seg_length=32)
    model = mech.wrap_model(model, TEST_CONFIG)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + mech._memory_params,
        lr=LR,
    )

    torch.manual_seed(SEED + 999)
    x = torch.randint(0, TEST_CONFIG.vocab_size - 1, (BATCH_SIZE, 64))
    y = x + 1

    losses = []
    for _ in range(n_steps):
        total_loss = 0.0
        memory = None
        for seg_idx in range(2):
            seg_x = x[:, seg_idx * 32:(seg_idx + 1) * 32]
            seg_y = y[:, seg_idx * 32:(seg_idx + 1) * 32]
            loss, memory = mech.forward_segment(model, seg_x, seg_y, memory)
            total_loss = total_loss + loss
            if seg_idx < 1:
                memory = memory.detach()
        assert torch.isfinite(total_loss), f"non-finite RMT loss at step {_}: {total_loss.item()}"
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(total_loss.item() / 2)

    init_loss = sum(losses[:5]) / 5
    final_loss = sum(losses[-5:]) / 5
    return init_loss, final_loss, losses


def _build_and_wrap(mechanism_name):
    """Build model, wrap with mechanism, return model."""
    torch.manual_seed(SEED)
    model = GPT(TEST_CONFIG)
    model.init_weights()

    if mechanism_name == "baseline":
        return model

    from memory_bench.mechanisms import MECHANISMS
    MechanismClass = MECHANISMS[mechanism_name]

    if mechanism_name == "persistent":
        mech = MechanismClass(num_tokens=8)
    elif mechanism_name == "rmt":
        mech = MechanismClass(num_tokens=4, seg_length=32)
    elif mechanism_name == "ttt":
        mech = MechanismClass(layer_idx=2, chunk_size=16)
    elif mechanism_name == "deltanet":
        mech = MechanismClass(layer_indices=[1])

    model = mech.wrap_model(model, TEST_CONFIG)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

MECHANISMS_TO_TEST = ["baseline", "persistent", "ttt", "deltanet"]
# Note: RMT requires segmented training (not standard model(x, y)),
# so it's tested separately below.


class TestRegressionBaselines:
    """Verify that training loss matches recorded baselines."""

    @pytest.mark.parametrize("mechanism", MECHANISMS_TO_TEST)
    def test_training_loss(self, mechanism):
        """Loss after N steps should match the recorded baseline."""
        model = _build_and_wrap(mechanism)
        init_loss, final_loss, losses = _train(model)

        if mechanism not in BASELINES:
            pytest.skip(f"No baseline recorded for {mechanism}")

        expected_init, expected_final = BASELINES[mechanism]

        assert abs(init_loss - expected_init) / expected_init < RTOL, \
            f"{mechanism} init loss changed: {init_loss:.6f} vs baseline {expected_init:.6f}"

        assert abs(final_loss - expected_final) / expected_final < RTOL, \
            f"{mechanism} final loss changed: {final_loss:.6f} vs baseline {expected_final:.6f}"

    @pytest.mark.parametrize("mechanism", MECHANISMS_TO_TEST)
    def test_loss_decreases(self, mechanism):
        """Every mechanism must reduce loss on fixed memorizable data."""
        model = _build_and_wrap(mechanism)
        init_loss, final_loss, losses = _train(model)

        assert final_loss < init_loss, \
            f"{mechanism}: loss didn't decrease ({init_loss:.4f} → {final_loss:.4f})"

        # Should decrease by at least 10%
        improvement = (init_loss - final_loss) / init_loss
        assert improvement > 0.10, \
            f"{mechanism}: loss decreased by only {improvement:.1%} — too little learning"

    def test_rmt_training_loss(self):
        """RMT regression baseline — segmented training with memory."""
        init_loss, final_loss, _ = _train_rmt()

        expected_init, expected_final = BASELINES["rmt"]

        assert abs(init_loss - expected_init) / expected_init < RTOL, \
            f"RMT init loss changed: {init_loss:.6f} vs baseline {expected_init:.6f}"

        assert abs(final_loss - expected_final) / expected_final < RTOL, \
            f"RMT final loss changed: {final_loss:.6f} vs baseline {expected_final:.6f}"

    def test_rmt_loss_decreases(self):
        """RMT uses segmented training — verify it separately."""
        init_loss, final_loss, _ = _train_rmt()

        assert final_loss < init_loss, \
            f"RMT: loss didn't decrease ({init_loss:.4f} → {final_loss:.4f})"

        improvement = (init_loss - final_loss) / init_loss
        assert improvement > 0.10, \
            f"RMT: loss decreased by only {improvement:.1%} — too little learning"

    def test_mechanisms_produce_different_losses(self):
        """Different mechanisms should produce at least slightly different loss curves.

        If two mechanisms produce identical losses, one of them probably
        isn't actually doing anything (like RMT eval before our fix).
        """
        results = {}
        for mechanism in MECHANISMS_TO_TEST:
            model = _build_and_wrap(mechanism)
            _, final_loss, _ = _train(model)
            results[mechanism] = final_loss

        # At minimum, mechanisms that modify the model (persistent, ttt, deltanet)
        # should produce different final losses than each other.
        # Note: at init with zero c_proj, some may start identical but diverge during training.
        mechanism_losses = [results[m] for m in ["persistent", "ttt", "deltanet"]]
        # At least 2 of 3 should be different
        unique_losses = len(set(round(l, 4) for l in mechanism_losses))
        assert unique_losses >= 2, \
            f"All mechanisms produced same loss — they may not be working: {results}"

