"""Tests for analyze_results.py — grouping, statistics, and report generation."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# --------------- helpers to build fake result dicts ---------------

def _fake_result(mechanism="none", seed=42, ctx=2048, bpb=0.85,
                 window_pattern="SSSL", n_buckets=32):
    """Create a minimal result dict matching train.py output."""
    buckets = {}
    bucket_size = ctx // n_buckets
    for i in range(n_buckets):
        start = i * bucket_size
        end = start + bucket_size
        center = (start + end) / 2
        # BPB decreases linearly from 1.3 to bpb across positions
        pos_bpb = 1.3 - (1.3 - bpb) * (i / (n_buckets - 1)) * 0.8
        buckets[f"{start}-{end}"] = {
            "center_position": center,
            "bpb": pos_bpb,
        }
    return {
        "mechanism": mechanism,
        "depth": 12,
        "max_seq_len": ctx,
        "window_pattern": window_pattern,
        "seed": seed,
        "val_bpb": bpb,
        "min_val_bpb": bpb,
        "total_params": 286_000_000,
        "base_params": 285_500_000,
        "memory_params": 500_000 if mechanism != "none" else 0,
        "param_overhead_pct": 0.17 if mechanism != "none" else 0.0,
        "total_time_min": 13.0 if ctx == 2048 else 25.0,
        "peak_vram_mib": 20000,
        "flops_per_step": 1e12,
        "total_flops": 2.5e15,
        "bpb_by_position": {
            "task": "bpb_by_position",
            "num_buckets": n_buckets,
            "sequence_length": ctx,
            "buckets": buckets,
            "overall_mean_bpb": bpb + 0.1,
        },
    }


def _build_direction_b_dataset():
    """Build a realistic Direction B dataset with 3 conditions x 3 seeds x 2 contexts."""
    results = []
    seeds = [42, 1337, 3141]
    for ctx in [2048, 4096]:
        for seed in seeds:
            # Local (SSSL) baseline
            results.append(_fake_result("none", seed, ctx, bpb=0.845 + ctx * 1e-6))
            # Global baseline
            results.append(_fake_result("none", seed, ctx, bpb=0.844 + ctx * 1e-6,
                                        window_pattern="L"))
            # Persistent memory
            results.append(_fake_result("persistent-32", seed, ctx, bpb=0.843 + ctx * 1e-6))
    return results


# --------------- tests ---------------

class TestNormalizeMechanism:
    def test_baseline_aliases(self):
        from analyze_results import normalize_mechanism
        assert normalize_mechanism("none") == "baseline"
        assert normalize_mechanism("baseline") == "baseline"

    def test_persistent_variants(self):
        from analyze_results import normalize_mechanism
        assert normalize_mechanism("persistent-32") == "persistent"
        assert normalize_mechanism("persistent-64") == "persistent"

    def test_other_mechanisms(self):
        from analyze_results import normalize_mechanism
        assert normalize_mechanism("rmt-m16-s512") == "rmt"
        assert normalize_mechanism("ttt-linear-c64") == "ttt"
        assert normalize_mechanism("deltanet-L4") == "deltanet"


class TestGroupResults:
    def test_excludes_global_runs(self):
        from analyze_results import group_results
        results = _build_direction_b_dataset()
        groups = group_results(results)
        # Global runs (window_pattern=L) should be excluded
        for key in groups:
            mech, ctx = key
            assert all(r.get("window_pattern", "SSSL") == "SSSL" for r in groups[key])

    def test_keys_are_tuples(self):
        from analyze_results import group_results
        results = _build_direction_b_dataset()
        groups = group_results(results)
        for key in groups:
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], str)
            assert isinstance(key[1], int)


class TestGroupByRole:
    def test_roles(self):
        from analyze_results import group_by_role
        results = _build_direction_b_dataset()
        roles = group_by_role(results)
        # Should have local, global, and memory roles
        role_names = set(k[0] for k in roles)
        assert "local" in role_names
        assert "global" in role_names
        assert "memory" in role_names

    def test_local_is_sssl_baseline(self):
        from analyze_results import group_by_role
        results = _build_direction_b_dataset()
        roles = group_by_role(results)
        for key, runs in roles.items():
            if key[0] == "local":
                for r in runs:
                    assert r.get("window_pattern", "SSSL") == "SSSL"
                    assert r["mechanism"] in ("none", "baseline")

    def test_global_is_L_baseline(self):
        from analyze_results import group_by_role
        results = _build_direction_b_dataset()
        roles = group_by_role(results)
        for key, runs in roles.items():
            if key[0] == "global":
                for r in runs:
                    assert r["window_pattern"] == "L"


class TestCrossoverAnalysis:
    def test_persistent_always_better(self):
        from analyze_results import crossover_analysis
        results = _build_direction_b_dataset()
        analysis = crossover_analysis(results)
        # Persistent is better than baseline at all contexts
        assert "persistent" in analysis
        info = analysis["persistent"]
        assert info["crossover_ctx"] is not None
        assert "<" in str(info["crossover_ctx"])  # "<2048"

    def test_returns_deltas(self):
        from analyze_results import crossover_analysis
        results = _build_direction_b_dataset()
        analysis = crossover_analysis(results)
        for mech, info in analysis.items():
            for ctx, (delta, pval, cohd) in info["deltas"].items():
                assert isinstance(delta, float)


class TestDeficitAnalysis:
    def test_computes_for_persistent(self):
        from analyze_results import compute_deficit_analysis
        results = _build_direction_b_dataset()
        analysis = compute_deficit_analysis(results)
        assert any("persistent" in str(k) for k in analysis)

    def test_deficit_shape(self):
        from analyze_results import compute_deficit_analysis
        results = _build_direction_b_dataset()
        analysis = compute_deficit_analysis(results)
        for key, data in analysis.items():
            assert len(data["deficit"]) == len(data["centers"])
            assert len(data["gain"]) == len(data["centers"])
            assert len(data["closure"]) == len(data["centers"])


class TestRegionalClosure:
    def test_computes_closure(self):
        from analyze_results import compute_regional_closure
        results = _build_direction_b_dataset()
        closure = compute_regional_closure(results)
        assert len(closure) > 0

    def test_closure_has_expected_keys(self):
        from analyze_results import compute_regional_closure
        results = _build_direction_b_dataset()
        closure = compute_regional_closure(results)
        for key, data in closure.items():
            assert "late_closure" in data
            assert "late_deficit" in data
            assert "late_gain" in data


class TestBlockBootstrap:
    def test_returns_dict_with_cis(self):
        from analyze_results import compute_block_bootstrap_ci
        rng = np.random.default_rng(42)
        deficit = rng.normal(0.001, 0.002, size=32)
        gain = rng.normal(0.003, 0.002, size=32)
        result = compute_block_bootstrap_ci(deficit, gain)
        assert "deficit_ci" in result
        assert "gain_ci" in result
        assert "closure_ci" in result
        lo, hi = result["deficit_ci"]
        assert lo < hi

    def test_known_deficit(self):
        from analyze_results import compute_block_bootstrap_ci
        deficit = np.ones(32) * 0.005
        gain = np.ones(32) * 0.003
        result = compute_block_bootstrap_ci(deficit, gain)
        lo, hi = result["deficit_ci"]
        # With constant series, CI should be tight around 0.005
        assert abs(lo - 0.005) < 0.001
        assert abs(hi - 0.005) < 0.001


class TestEffectiveDOF:
    def test_uncorrelated_near_n(self):
        from analyze_results import compute_effective_dof
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, size=100)
        n_eff, rho = compute_effective_dof(series)
        # Uncorrelated data should have n_eff close to n
        assert n_eff >= 50  # at least half of n

    def test_correlated_reduces_dof(self):
        from analyze_results import compute_effective_dof
        # Highly correlated series
        series = np.cumsum(np.ones(100))
        n_eff, rho = compute_effective_dof(series)
        assert n_eff < 20  # much less than 100


class TestReportGeneration:
    def test_generates_markdown(self):
        from analyze_results import generate_report
        results = _build_direction_b_dataset()
        report = generate_report(results)
        assert "# memory-bench" in report
        assert "## Results" in report
        assert "Persistent Memory" in report

    def test_contains_tables(self):
        from analyze_results import generate_report
        results = _build_direction_b_dataset()
        report = generate_report(results)
        assert "|" in report
        assert "mBPB" in report

    def test_deficit_section_present(self):
        from analyze_results import generate_report
        results = _build_direction_b_dataset()
        report = generate_report(results)
        assert "Positional Context Deficit" in report
        assert "Falsification" in report
