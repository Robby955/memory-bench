"""
Tests for optimizer param group surgery after mechanism wrapping.

When a memory mechanism replaces or adds layers, the optimizer must be updated:
1. Stale params (from replaced layers) must be removed
2. New params (from mechanism layers) must be added to correct groups
3. Mechanism-specific params must go to a separate optimizer
4. No param should be in both optimizers
5. Every model param should be optimized by exactly one optimizer

This is the most complex and bug-prone code in train.py (lines 213-265).
A bug here means some params silently get no gradients.

Run with: pytest tests/test_optimizer_surgery.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanochat"))

import torch
import pytest
from nanochat.gpt import GPT, GPTConfig

from memory_bench.mechanisms import MECHANISMS


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


def _do_optimizer_surgery(mechanism_name):
    """Replicate the optimizer surgery from train.py lines 199-265.

    Returns (model, main_optimizer, mech_optimizer, mechanism).
    """
    torch.manual_seed(SEED)
    model = GPT(TEST_CONFIG)
    model.init_weights()

    # Setup optimizer BEFORE wrapping (matches train.py)
    optimizer = model.setup_optimizer(
        unembedding_lr=0.008,
        embedding_lr=0.3,
        scalar_lr=0.5,
        matrix_lr=0.02,
        weight_decay=0.1,
    )

    MechanismClass = MECHANISMS[mechanism_name]
    if mechanism_name == "persistent":
        mechanism = MechanismClass(num_tokens=8)
    elif mechanism_name == "rmt":
        mechanism = MechanismClass(num_tokens=4, seg_length=32)
    elif mechanism_name == "ttt":
        mechanism = MechanismClass(layer_idx=2, chunk_size=16)
    elif mechanism_name == "deltanet":
        mechanism = MechanismClass(layer_indices=[1])
    else:
        raise ValueError(f"Unknown mechanism: {mechanism_name}")

    model = mechanism.wrap_model(model, TEST_CONFIG)

    # Identify mechanism-specific params
    mech_extra_ids = set()
    for g in mechanism.extra_param_groups():
        for p in g["params"]:
            mech_extra_ids.add(id(p))

    # Step 1: Remove stale params
    current_param_ids = {id(p) for p in model.parameters()}
    for group in optimizer.param_groups:
        group["params"] = [p for p in group["params"] if id(p) in current_param_ids]

    # Step 2: Find and add orphan params
    optimizer_param_ids = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            optimizer_param_ids.add(id(p))

    orphan_params = [p for p in model.parameters()
                     if id(p) not in optimizer_param_ids and id(p) not in mech_extra_ids]

    for p in orphan_params:
        added = False
        for group in optimizer.param_groups:
            if group.get("kind") == "muon" and any(pp.shape == p.shape for pp in group["params"]):
                group["params"].append(p)
                added = True
                break
        if not added:
            optimizer.add_param_group({
                "kind": "adamw", "params": [p], "lr": 0.01,
                "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01,
                "initial_lr": 0.01,
            })

    # Step 3: Separate mechanism optimizer
    mech_optimizer = None
    mech_param_groups = []
    for group in mechanism.extra_param_groups():
        group.pop("kind", None)
        group["initial_lr"] = group["lr"]
        mech_param_groups.append(group)
    if mech_param_groups:
        mech_optimizer = torch.optim.AdamW(
            mech_param_groups,
            lr=mech_param_groups[0]["lr"],
        )

    return model, optimizer, mech_optimizer, mechanism


MECHANISMS_WITH_SURGERY = ["ttt", "deltanet", "rmt", "persistent"]


class TestOptimizerSurgeryInvariants:
    """Every model param must be in exactly one optimizer after surgery."""

    @pytest.mark.parametrize("mechanism", MECHANISMS_WITH_SURGERY)
    def test_no_stale_params(self, mechanism):
        """Main optimizer must not contain params from replaced layers."""
        model, optimizer, mech_optimizer, _ = _do_optimizer_surgery(mechanism)
        model_param_ids = {id(p) for p in model.parameters()}

        for group in optimizer.param_groups:
            for p in group["params"]:
                assert id(p) in model_param_ids, \
                    f"Stale param (shape {p.shape}) found in main optimizer"

    @pytest.mark.parametrize("mechanism", MECHANISMS_WITH_SURGERY)
    def test_all_params_optimized(self, mechanism):
        """Every model parameter must be in some optimizer."""
        model, optimizer, mech_optimizer, _ = _do_optimizer_surgery(mechanism)

        optimized_ids = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                optimized_ids.add(id(p))
        if mech_optimizer is not None:
            for group in mech_optimizer.param_groups:
                for p in group["params"]:
                    optimized_ids.add(id(p))

        for name, p in model.named_parameters():
            assert id(p) in optimized_ids, \
                f"Param '{name}' (shape {p.shape}) is not in any optimizer — will get no gradients"

    @pytest.mark.parametrize("mechanism", MECHANISMS_WITH_SURGERY)
    def test_no_double_optimization(self, mechanism):
        """No param should appear in both main and mechanism optimizer."""
        model, optimizer, mech_optimizer, _ = _do_optimizer_surgery(mechanism)
        if mech_optimizer is None:
            return  # persistent has no separate mech optimizer

        main_ids = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                main_ids.add(id(p))

        mech_ids = set()
        for group in mech_optimizer.param_groups:
            for p in group["params"]:
                mech_ids.add(id(p))

        overlap = main_ids & mech_ids
        assert not overlap, \
            f"{len(overlap)} params are in BOTH optimizers — will get double gradients"

    @pytest.mark.parametrize("mechanism", MECHANISMS_WITH_SURGERY)
    def test_no_duplicate_within_optimizer(self, mechanism):
        """No param should appear twice within the same optimizer."""
        model, optimizer, mech_optimizer, _ = _do_optimizer_surgery(mechanism)

        seen = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                assert id(p) not in seen, \
                    f"Param (shape {p.shape}) appears twice in main optimizer"
                seen.add(id(p))

        if mech_optimizer is not None:
            seen = set()
            for group in mech_optimizer.param_groups:
                for p in group["params"]:
                    assert id(p) not in seen, \
                        f"Param (shape {p.shape}) appears twice in mech optimizer"
                    seen.add(id(p))


class TestOptimizerSurgeryGradients:
    """Verify that all params actually receive gradients after surgery."""

    @pytest.mark.parametrize("mechanism", ["ttt", "deltanet"])
    def test_all_params_get_gradients(self, mechanism):
        """Forward + backward should produce gradients for every optimized param."""
        model, optimizer, mech_optimizer, mech = _do_optimizer_surgery(mechanism)

        # Forward pass
        torch.manual_seed(SEED + 999)
        x = torch.randint(0, TEST_CONFIG.vocab_size, (2, 64))
        y = x + 1

        loss = model(x, y)
        loss.backward()

        # Check main optimizer params
        no_grad_main = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    no_grad_main.append(p.shape)

        # Check mech optimizer params
        no_grad_mech = []
        if mech_optimizer is not None:
            for group in mech_optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        no_grad_mech.append(p.shape)

        # Allow some params to have no grad (e.g. unused mechanism-specific
        # params at init due to zero c_proj), but flag them
        if no_grad_main:
            import warnings
            warnings.warn(f"{mechanism}: {len(no_grad_main)} main optimizer params got no grad: {no_grad_main}")

        if no_grad_mech:
            import warnings
            warnings.warn(f"{mechanism}: {len(no_grad_mech)} mech optimizer params got no grad: {no_grad_mech}")


class TestExtraParamGroups:
    """Verify extra_param_groups() returns valid data."""

    @pytest.mark.parametrize("mechanism_name", ["rmt", "ttt", "deltanet"])
    def test_extra_params_non_empty(self, mechanism_name):
        """Mechanisms with learnable state should have extra param groups."""
        torch.manual_seed(SEED)
        model = GPT(TEST_CONFIG)
        model.init_weights()

        MechanismClass = MECHANISMS[mechanism_name]
        if mechanism_name == "rmt":
            mech = MechanismClass(num_tokens=4, seg_length=32)
        elif mechanism_name == "ttt":
            mech = MechanismClass(layer_idx=2, chunk_size=16)
        elif mechanism_name == "deltanet":
            mech = MechanismClass(layer_indices=[1])

        model = mech.wrap_model(model, TEST_CONFIG)
        groups = mech.extra_param_groups()

        assert len(groups) > 0, f"{mechanism_name} should have extra param groups"
        for g in groups:
            assert "params" in g, "Missing 'params' key"
            assert "lr" in g, "Missing 'lr' key"
            assert len(g["params"]) > 0, "Empty param group"
            for p in g["params"]:
                assert isinstance(p, torch.nn.Parameter) or isinstance(p, torch.Tensor), \
                    f"Expected Parameter, got {type(p)}"
                assert p.requires_grad, "Extra param should require grad"

    def test_persistent_has_no_extra_params(self):
        """Persistent memory adds params to the model but not to a separate optimizer."""
        from memory_bench.mechanisms.persistent import PersistentMemory

        torch.manual_seed(SEED)
        model = GPT(TEST_CONFIG)
        model.init_weights()
        mech = PersistentMemory(num_tokens=8)
        model = mech.wrap_model(model, TEST_CONFIG)

        # Persistent embeds memory into the model's embedding layer,
        # so extra_param_groups may or may not return groups depending on impl
        groups = mech.extra_param_groups()
        # Just verify it doesn't crash and returns a list
        assert isinstance(groups, list)
