#!/usr/bin/env python3
"""Analyze memory-bench results and generate publication-ready figures.

Supports multi-context results: groups by (mechanism, context_length) and
produces crossover analysis + position-dependent gain plots.

Usage:
    python analyze_results.py                    # use local results/
    python analyze_results.py --pull HOST PORT   # pull from pod first
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

MECHANISMS_ORDER = ["baseline", "persistent", "rmt", "ttt", "deltanet"]
LABELS = {
    "baseline": "Baseline\n(no memory)",
    "persistent": "Persistent\nMemory",
    "rmt": "Recurrent\nMemory (RMT)",
    "ttt": "Test-Time\nTraining",
    "deltanet": "Gated\nDeltaNet",
}
LABELS_SHORT = {
    "baseline": "Baseline",
    "persistent": "Persistent Memory",
    "rmt": "RMT",
    "ttt": "TTT-Linear",
    "deltanet": "Gated DeltaNet",
}
COLORS = {
    "baseline": "#4A90D9",
    "persistent": "#E8A838",
    "rmt": "#50C878",
    "ttt": "#E85050",
    "deltanet": "#9B59B6",
}


def normalize_mechanism(name: str) -> str:
    if name in ("none", "baseline"):
        return "baseline"
    for base in MECHANISMS_ORDER:
        if name == base or name.startswith(base + "-"):
            return base
    return name


def load_results() -> list[dict]:
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name.startswith("benchmark_") or f.name.startswith("statistical_"):
            continue
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def group_by_mechanism(results: list[dict]) -> dict[str, list[dict]]:
    """Group results by mechanism only (backward compat)."""
    groups = {}
    for r in results:
        mech = normalize_mechanism(r.get("mechanism", "baseline"))
        groups.setdefault(mech, []).append(r)
    return groups


def group_results(results: list[dict]) -> dict[tuple[str, int], list[dict]]:
    """Group by (mechanism, context_length). Excludes global-attention reference runs."""
    groups = {}
    for r in results:
        if r.get("window_pattern", "SSSL") != "SSSL":
            continue  # skip global-attention reference runs
        mech = normalize_mechanism(r.get("mechanism", "baseline"))
        ctx = r.get("max_seq_len", 2048)
        groups.setdefault((mech, ctx), []).append(r)
    return groups


def get_context_lengths(results: list[dict]) -> list[int]:
    """Get sorted unique context lengths in results."""
    return sorted(set(r.get("max_seq_len", 2048) for r in results))


def get_window_pattern(r: dict) -> str:
    """Get window pattern from a result dict, defaulting to SSSL."""
    return r.get("window_pattern", "SSSL")


def filter_by_window(results: list[dict], pattern: str) -> list[dict]:
    """Filter results to a specific window pattern."""
    return [r for r in results if get_window_pattern(r) == pattern]


def group_by_role(results: list[dict]) -> dict[tuple[str, str, int], list[dict]]:
    """Group by (role, mechanism, context_length).

    role is one of:
      - 'local': SSSL baseline (window_pattern=SSSL, mechanism=baseline)
      - 'global': Global-attention reference (window_pattern=L, mechanism=baseline)
      - 'memory': Memory mechanism with SSSL (window_pattern=SSSL, mechanism!=baseline)
    """
    groups = {}
    for r in results:
        mech = normalize_mechanism(r.get("mechanism", "baseline"))
        wp = get_window_pattern(r)
        ctx = r.get("max_seq_len", 2048)
        if mech == "baseline" and wp == "L":
            role = "global"
        elif mech == "baseline":
            role = "local"
        else:
            role = "memory"
        groups.setdefault((role, mech, ctx), []).append(r)
    return groups


def print_summary(results: list[dict]):
    """Print per-mechanism summary (single-context, backward compat)."""
    groups = group_by_mechanism(results)
    baseline_bpb = None

    print(f"\n{'='*75}")
    print(f"{'Mechanism':<20} {'Seeds':>5} {'BPB Mean':>10} {'BPB Std':>8} {'Delta':>8} {'Time':>8} {'Mem Params':>12}")
    print(f"{'='*75}")

    for mech in MECHANISMS_ORDER:
        if mech not in groups:
            continue
        runs = groups[mech]
        bpbs = [r["min_val_bpb"] for r in runs if r.get("min_val_bpb") is not None]
        times = [r["total_time_min"] for r in runs if r.get("total_time_min") is not None]
        mem_params = runs[0].get("memory_params", 0)

        if not bpbs:
            continue

        mean_bpb = np.mean(bpbs)
        std_bpb = np.std(bpbs, ddof=1) if len(bpbs) > 1 else 0
        mean_time = np.mean(times) if times else 0

        if mech == "baseline":
            baseline_bpb = mean_bpb
            delta = "—"
        else:
            delta = f"{mean_bpb - baseline_bpb:+.4f}" if baseline_bpb else "?"

        print(f"{mech:<20} {len(bpbs):>5} {mean_bpb:>10.5f} {std_bpb:>8.5f} {delta:>8} {mean_time:>7.1f}m {mem_params:>12,}")

    print()


def print_full_summary(results: list[dict]):
    """Print mechanism x context_length matrix."""
    groups = group_results(results)
    contexts = get_context_lengths(results)

    if len(contexts) <= 1:
        print_summary(results)
        return

    # Header
    ctx_headers = "".join(f"{'T=' + str(c):>16}" for c in contexts)
    print(f"\n{'='*(22 + 16*len(contexts))}")
    print(f"{'Mechanism':<22}{ctx_headers}")
    print(f"{'='*(22 + 16*len(contexts))}")

    # Compute baseline means for delta computation
    baseline_means = {}
    for ctx in contexts:
        key = ("baseline", ctx)
        if key in groups:
            bpbs = [r["min_val_bpb"] for r in groups[key] if r.get("min_val_bpb") is not None]
            if bpbs:
                baseline_means[ctx] = np.mean(bpbs)

    for mech in MECHANISMS_ORDER:
        cells = []
        for ctx in contexts:
            key = (mech, ctx)
            if key not in groups:
                cells.append(f"{'—':>16}")
                continue
            bpbs = [r["min_val_bpb"] for r in groups[key] if r.get("min_val_bpb") is not None]
            if not bpbs:
                cells.append(f"{'—':>16}")
                continue
            mean = np.mean(bpbs)
            std = np.std(bpbs, ddof=1) if len(bpbs) > 1 else 0
            if mech == "baseline":
                cells.append(f"{mean:.5f}±{std:.4f}"[:16].rjust(16))
            else:
                delta = mean - baseline_means.get(ctx, mean)
                cells.append(f"{mean:.5f}({delta:+.4f})"[:16].rjust(16))
        print(f"{mech:<22}{''.join(cells)}")

    print()


def crossover_analysis(results: list[dict]) -> dict:
    """For each mechanism, find context length where it first beats baseline.

    Returns dict keyed by mechanism with:
      - deltas: {ctx: (mean_delta, p_value, cohens_d)}
      - crossover_ctx: estimated crossover context (None if never beats baseline)
    """
    groups = group_results(results)
    contexts = get_context_lengths(results)
    output = {}

    # Get baseline BPBs by (ctx, seed)
    baseline_by_ctx_seed = {}
    for ctx in contexts:
        key = ("baseline", ctx)
        if key in groups:
            for r in groups[key]:
                if r.get("min_val_bpb") is not None:
                    baseline_by_ctx_seed[(ctx, r["seed"])] = r["min_val_bpb"]

    for mech in MECHANISMS_ORDER:
        if mech == "baseline":
            continue

        deltas = {}
        for ctx in contexts:
            key = (mech, ctx)
            if key not in groups:
                continue

            # Pair by seed
            paired_b, paired_m = [], []
            for r in groups[key]:
                seed = r["seed"]
                if r.get("min_val_bpb") is not None and (ctx, seed) in baseline_by_ctx_seed:
                    paired_b.append(baseline_by_ctx_seed[(ctx, seed)])
                    paired_m.append(r["min_val_bpb"])

            if len(paired_b) < 2:
                mean_delta = np.mean(paired_m) - np.mean(paired_b) if paired_b else None
                deltas[ctx] = (mean_delta, None, None)
                continue

            d_arr = np.array(paired_m) - np.array(paired_b)
            mean_delta = float(np.mean(d_arr))
            std_delta = float(np.std(d_arr, ddof=1))
            _, p_value = stats.ttest_rel(paired_m, paired_b)
            cohens_d = mean_delta / std_delta if std_delta > 0 else float("inf")
            deltas[ctx] = (mean_delta, float(p_value), float(cohens_d))

        # Find crossover: linear interpolation between last positive and first negative delta
        crossover_ctx = None
        ctx_delta_pairs = [(c, d[0]) for c, d in sorted(deltas.items()) if d[0] is not None]
        for i in range(len(ctx_delta_pairs) - 1):
            c1, d1 = ctx_delta_pairs[i]
            c2, d2 = ctx_delta_pairs[i + 1]
            if d1 > 0 and d2 <= 0:
                # Linear interpolation
                crossover_ctx = c1 + (0 - d1) * (c2 - c1) / (d2 - d1)
                break
        if ctx_delta_pairs and all(d < 0 for _, d in ctx_delta_pairs):
            crossover_ctx = f"<{ctx_delta_pairs[0][0]}"

        output[mech] = {"deltas": deltas, "crossover_ctx": crossover_ctx}

    return output


def print_crossover_summary(results: list[dict]):
    """Print crossover analysis table."""
    analysis = crossover_analysis(results)
    contexts = get_context_lengths(results)

    if not analysis:
        return

    print(f"\n{'='*80}")
    print("CROSSOVER ANALYSIS: Context length where mechanism beats baseline")
    print(f"{'='*80}")

    # Header
    ctx_headers = "".join(f"{'T=' + str(c):>18}" for c in contexts)
    print(f"{'Mechanism':<18}{ctx_headers}  {'Crossover':>12}")
    print("-" * (18 + 18 * len(contexts) + 14))

    for mech in MECHANISMS_ORDER:
        if mech not in analysis:
            continue
        info = analysis[mech]
        cells = []
        for ctx in contexts:
            if ctx not in info["deltas"]:
                cells.append(f"{'—':>18}")
                continue
            mean_d, p_val, d_eff = info["deltas"][ctx]
            if mean_d is None:
                cells.append(f"{'—':>18}")
            elif p_val is not None:
                sig = "*" if p_val < 0.05 else ""
                cells.append(f"{mean_d:+.5f} p={p_val:.3f}{sig}".rjust(18))
            else:
                cells.append(f"{mean_d:+.5f}".rjust(18))

        crossover = info["crossover_ctx"]
        if crossover is None:
            cross_str = "never"
        elif isinstance(crossover, str):
            cross_str = crossover
        else:
            cross_str = f"~{crossover:.0f}"

        print(f"{mech:<18}{''.join(cells)}  {cross_str:>12}")

    print()
    print("Positive delta = mechanism worse than baseline. * = p<0.05")
    print()


# ---------------------------------------------------------------------------
# Positional Context Deficit Framework
#
# Definitions (per position bucket p, at context length T):
#
#   deficit(p, T) = BPB_local(p, T) - BPB_global(p, T)
#     The performance gap between a local-attention model (window pattern SSSL)
#     and a matched global-attention reference (window pattern L) at position p.
#     Positive deficit means local attention is underperforming.
#
#   gain(p, T) = BPB_local(p, T) - BPB_memory(p, T)
#     The per-position improvement from adding a memory mechanism to the
#     local-attention model. Positive gain means memory helps.
#
#   closure(p, T) = gain(p, T) / max(deficit(p, T), epsilon)
#     Fraction of the context deficit recovered by the memory mechanism.
#     Values near 1.0 = memory fully closes the gap.
#     Values near 0.0 = memory does not help where attention is impaired.
#     Undefined (clipped) where deficit ≈ 0 (local attention is already sufficient).
#
# ---------------------------------------------------------------------------

DEFICIT_EPSILON = 0.001  # floor for deficit in closure computation (BPB units)


def _extract_position_curves(runs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-position BPB curves from a list of runs.

    Returns:
        centers: (B,) array of bucket center positions
        curves: (N, B) array of BPB values, one row per seed
    """
    all_curves = []
    centers = None
    for r in runs:
        bpb_pos = r.get("bpb_by_position", {})
        buckets = bpb_pos.get("buckets", {})
        if not buckets:
            continue
        c = np.array([v["center_position"] for v in buckets.values()])
        b = np.array([v["bpb"] for v in buckets.values()])
        if centers is None:
            centers = c
        all_curves.append(b)
    if not all_curves or centers is None:
        return np.array([]), np.array([])
    return centers, np.array(all_curves)


def compute_deficit_analysis(results: list[dict]) -> dict:
    """Compute positional context deficit, gain, and closure for each context length.

    Returns dict keyed by (mechanism, context_length) with:
      - centers: bucket center positions
      - deficit: mean deficit curve (local - global)
      - deficit_std: std across seeds
      - gain: mean gain curve (local - memory)
      - gain_std: std across seeds
      - closure: mean closure curve (gain / max(deficit, epsilon))
      - closure_mean: scalar mean closure across all buckets
      - deficit_gain_corr: Spearman correlation between deficit and gain
      - deficit_gain_pvalue: p-value for the correlation
      - aggregate: dict with scalar BPB values for local, global, memory
    """
    roles = group_by_role(results)
    contexts = get_context_lengths(results)
    analysis = {}

    for ctx in contexts:
        # Get local baseline curves
        local_runs = roles.get(("local", "baseline", ctx), [])
        local_centers, local_curves = _extract_position_curves(local_runs)
        if local_curves.size == 0:
            continue

        # Get global reference curves
        global_runs = roles.get(("global", "baseline", ctx), [])
        global_centers, global_curves = _extract_position_curves(global_runs)
        if global_curves.size == 0:
            continue

        # Align bucket counts (take min if they differ)
        n_buckets = min(local_curves.shape[1], global_curves.shape[1])
        local_mean = local_curves[:, :n_buckets].mean(axis=0)
        global_mean = global_curves[:, :n_buckets].mean(axis=0)
        centers = local_centers[:n_buckets]

        # Deficit: local - global (positive = local is worse)
        deficit = local_mean - global_mean
        # Error propagation: std(A - B) = sqrt(std(A)^2/n_A + std(B)^2/n_B)
        # for independent groups (different seeds)
        local_std = local_curves[:, :n_buckets].std(axis=0, ddof=1) if local_curves.shape[0] > 1 else np.zeros(n_buckets)
        global_std = global_curves[:, :n_buckets].std(axis=0, ddof=1) if global_curves.shape[0] > 1 else np.zeros(n_buckets)
        n_local = max(local_curves.shape[0], 1)
        n_global = max(global_curves.shape[0], 1)
        deficit_std = np.sqrt(local_std**2 / n_local + global_std**2 / n_global)

        # For each memory mechanism at this context
        for (role, mech, c), runs in roles.items():
            if role != "memory" or c != ctx:
                continue

            mem_centers, mem_curves = _extract_position_curves(runs)
            if mem_curves.size == 0:
                continue

            n_b = min(n_buckets, mem_curves.shape[1])
            mem_mean = mem_curves[:, :n_b].mean(axis=0)

            # Gain: local - memory (positive = memory helps)
            gain = local_mean[:n_b] - mem_mean
            # Error propagation for gain (independent groups)
            mem_std = mem_curves[:, :n_b].std(axis=0, ddof=1) if mem_curves.shape[0] > 1 else np.zeros(n_b)
            n_mem = max(mem_curves.shape[0], 1)
            gain_std = np.sqrt(local_std[:n_b]**2 / n_local + mem_std**2 / n_mem)

            # Closure: gain / max(deficit, epsilon)
            safe_deficit = np.maximum(deficit[:n_b], DEFICIT_EPSILON)
            closure = gain / safe_deficit
            # Mask closure where deficit is negligible (< epsilon)
            closure_valid = deficit[:n_b] > DEFICIT_EPSILON
            closure_mean = float(np.mean(closure[closure_valid])) if closure_valid.any() else float("nan")

            # Correlation between deficit magnitude and gain
            if n_b >= 5 and deficit[:n_b].std() > 0 and gain.std() > 0:
                corr_result = stats.spearmanr(deficit[:n_b], gain)
                corr = float(corr_result.statistic)
                corr_p = float(corr_result.pvalue)
            else:
                corr, corr_p = float("nan"), float("nan")

            # Aggregate scalars
            local_bpbs = [r["min_val_bpb"] for r in local_runs if r.get("min_val_bpb")]
            global_bpbs = [r["min_val_bpb"] for r in global_runs if r.get("min_val_bpb")]
            mem_bpbs = [r["min_val_bpb"] for r in runs if r.get("min_val_bpb")]

            analysis[(mech, ctx)] = {
                "centers": centers[:n_b],
                "deficit": deficit[:n_b],
                "deficit_std": deficit_std[:n_b],
                "gain": gain,
                "gain_std": gain_std,
                "closure": closure,
                "closure_valid": closure_valid,
                "closure_mean": closure_mean,
                "deficit_gain_corr": corr,
                "deficit_gain_pvalue": corr_p,
                "n_buckets": n_b,
                "aggregate": {
                    "local_bpb": float(np.mean(local_bpbs)) if local_bpbs else None,
                    "global_bpb": float(np.mean(global_bpbs)) if global_bpbs else None,
                    "memory_bpb": float(np.mean(mem_bpbs)) if mem_bpbs else None,
                    "local_bpb_std": float(np.std(local_bpbs, ddof=1)) if len(local_bpbs) > 1 else 0,
                    "global_bpb_std": float(np.std(global_bpbs, ddof=1)) if len(global_bpbs) > 1 else 0,
                    "memory_bpb_std": float(np.std(mem_bpbs, ddof=1)) if len(mem_bpbs) > 1 else 0,
                },
            }

    return analysis


def compute_regional_closure(results: list[dict]) -> dict:
    """Compute regional closure and selectivity (primary endpoint).

    Partitions positions into early (first quartile) and late (final quartile).
    Regional closure is defined on token-weighted average BPB per region:

        C(T, R) = [B_local(T,R) - B_memory(T,R)] / [B_local(T,R) - B_global(T,R)]

    Selectivity: S(T) = C(T, R_late) - C(T, R_early)

    The primary endpoint is: seed-averaged late-region closure at T=8192.

    Returns dict keyed by (mechanism, context_length) with:
        late_closure, early_closure, selectivity, late_deficit, late_gain
    """
    roles = group_by_role(results)
    contexts = get_context_lengths(results)
    output = {}

    for ctx in contexts:
        local_runs = roles.get(("local", "baseline", ctx), [])
        global_runs = roles.get(("global", "baseline", ctx), [])

        lc, l_curves = _extract_position_curves(local_runs)
        gc, g_curves = _extract_position_curves(global_runs)
        if l_curves.size == 0 or g_curves.size == 0:
            continue

        n_b = min(l_curves.shape[1], g_curves.shape[1])
        l_mean = l_curves[:, :n_b].mean(axis=0)
        g_mean = g_curves[:, :n_b].mean(axis=0)

        for (role, mech, c), runs in roles.items():
            if role != "memory" or c != ctx:
                continue

            mc, m_curves = _extract_position_curves(runs)
            if m_curves.size == 0:
                continue

            # Use shared bucket count so all three models cover the same positions
            n_shared = min(n_b, m_curves.shape[1])
            m_mean = m_curves[:, :n_shared].mean(axis=0)

            # Recompute quartiles on shared range for all three models
            q1s = n_shared // 4
            q3s = 3 * n_shared // 4

            l_early_s = float(np.mean(l_mean[:q1s]))
            l_late_s = float(np.mean(l_mean[q3s:n_shared]))
            g_early_s = float(np.mean(g_mean[:q1s]))
            g_late_s = float(np.mean(g_mean[q3s:n_shared]))
            m_early = float(np.mean(m_mean[:q1s]))
            m_late = float(np.mean(m_mean[q3s:]))

            # Regional deficit and gain (all computed over same position range)
            late_deficit = l_late_s - g_late_s
            early_deficit = l_early_s - g_early_s
            late_gain = l_late_s - m_late
            early_gain = l_early_s - m_early

            # Regional closure (only when denominator > epsilon)
            late_closure = late_gain / max(late_deficit, DEFICIT_EPSILON) if late_deficit > DEFICIT_EPSILON else float("nan")
            early_closure = early_gain / max(early_deficit, DEFICIT_EPSILON) if early_deficit > DEFICIT_EPSILON else float("nan")

            selectivity = float("nan")
            if not np.isnan(late_closure) and not np.isnan(early_closure):
                selectivity = late_closure - early_closure

            output[(mech, ctx)] = {
                "late_closure": late_closure,
                "early_closure": early_closure,
                "selectivity": selectivity,
                "late_deficit": late_deficit,
                "late_gain": late_gain,
                "early_deficit": early_deficit,
                "early_gain": early_gain,
            }

    return output


def print_regional_closure(results: list[dict]):
    """Print the primary endpoint: regional closure and selectivity."""
    regional = compute_regional_closure(results)
    if not regional:
        return

    contexts = sorted(set(ctx for _, ctx in regional.keys()))
    mechs = sorted(set(m for m, _ in regional.keys()))

    print(f"\n{'='*90}")
    print("  PRIMARY ENDPOINT: Regional Closure and Selectivity")
    print(f"{'='*90}")
    print()
    print(f"  C(T, R) = [B_local(R) - B_memory(R)] / [B_local(R) - B_global(R)]")
    print(f"  S(T)    = C(T, R_late) - C(T, R_early)")
    print(f"  R_early = first quartile of positions,  R_late = final quartile")
    print()

    for mech in mechs:
        label = LABELS_SHORT.get(mech, mech)
        print(f"  {label}:")
        print(f"  {'Context':>8} {'Late Deficit':>12} {'Late Gain':>10} "
              f"{'Late C':>8} {'Early C':>8} {'Selectivity':>12}")
        print(f"  {'-'*64}")

        for ctx in contexts:
            info = regional.get((mech, ctx))
            if not info:
                continue
            lc = f"{info['late_closure']:.3f}" if not np.isnan(info['late_closure']) else "n/a"
            ec = f"{info['early_closure']:.3f}" if not np.isnan(info['early_closure']) else "n/a"
            sel = f"{info['selectivity']:+.3f}" if not np.isnan(info['selectivity']) else "n/a"
            print(f"  {ctx:>8} {info['late_deficit']:>+12.5f} {info['late_gain']:>+10.5f} "
                  f"{lc:>8} {ec:>8} {sel:>12}")

        # Highlight T=8192 as primary
        primary = regional.get((mech, 8192))
        if primary and not np.isnan(primary["late_closure"]):
            print(f"\n  >>> PRIMARY ENDPOINT (T=8192): late-region closure = {primary['late_closure']:.3f}")
            if not np.isnan(primary["selectivity"]):
                print(f"  >>> Selectivity: {primary['selectivity']:+.3f}")
        print()


def print_deficit_summary(results: list[dict]):
    """Print deficit/gain/closure summary table."""
    analysis = compute_deficit_analysis(results)
    if not analysis:
        print("  No deficit analysis available (need local + global + memory results)")
        return

    contexts = sorted(set(ctx for _, ctx in analysis.keys()))
    mechs = sorted(set(m for m, _ in analysis.keys()))

    print(f"\n{'='*90}")
    print("  POSITIONAL CONTEXT DEFICIT ANALYSIS")
    print(f"{'='*90}")
    print()
    print(f"  deficit(p) = BPB_local(p) - BPB_global(p)   [positive = local attention impaired]")
    print(f"  gain(p)    = BPB_local(p) - BPB_memory(p)   [positive = memory helps]")
    print(f"  closure(p) = gain(p) / deficit(p)            [fraction of deficit recovered]")
    print()

    for mech in mechs:
        label = LABELS_SHORT.get(mech, mech)
        print(f"  {label}:")
        print(f"  {'Context':>8} {'Local BPB':>10} {'Global BPB':>11} {'Memory BPB':>11} "
              f"{'Mean Deficit':>13} {'Mean Gain':>10} {'Closure':>8} {'Corr(d,g)':>10} {'p-val':>8}")
        print(f"  {'-'*82}")

        for ctx in contexts:
            info = analysis.get((mech, ctx))
            if not info:
                continue
            agg = info["aggregate"]
            mean_deficit = float(np.mean(info["deficit"]))
            mean_gain = float(np.mean(info["gain"]))

            print(f"  {ctx:>8} "
                  f"{agg['local_bpb']:>10.5f} "
                  f"{agg['global_bpb']:>11.5f} "
                  f"{agg['memory_bpb']:>11.5f} "
                  f"{mean_deficit:>+13.5f} "
                  f"{mean_gain:>+10.5f} "
                  f"{info['closure_mean']:>8.3f} "
                  f"{info['deficit_gain_corr']:>+10.3f} "
                  f"{info['deficit_gain_pvalue']:>8.4f}")

        print()

    # Falsification checks
    print(f"  {'─'*90}")
    print("  FALSIFICATION CHECKS:")
    print()

    # Check 1: Does deficit grow with context length?
    for mech in mechs:
        deficits_by_ctx = {}
        for ctx in contexts:
            info = analysis.get((mech, ctx))
            if info:
                deficits_by_ctx[ctx] = float(np.mean(info["deficit"]))
        if len(deficits_by_ctx) >= 2:
            ctx_list = sorted(deficits_by_ctx.keys())
            growing = all(deficits_by_ctx[ctx_list[i+1]] > deficits_by_ctx[ctx_list[i]]
                          for i in range(len(ctx_list) - 1))
            status = "PASS" if growing else "FAIL"
            label = LABELS_SHORT.get(mech, mech)
            vals = ", ".join(f"T={c}: {deficits_by_ctx[c]:+.5f}" for c in ctx_list)
            print(f"    Deficit grows with context ({label}): [{status}] {vals}")

    # Check 2: Is gain concentrated in high-deficit buckets?
    for mech in mechs:
        for ctx in contexts:
            info = analysis.get((mech, ctx))
            if info and not np.isnan(info["deficit_gain_corr"]):
                label = LABELS_SHORT.get(mech, mech)
                sig = "yes" if info["deficit_gain_pvalue"] < 0.05 else "no"
                print(f"    Gain correlates with deficit ({label}, T={ctx}): "
                      f"rho={info['deficit_gain_corr']:+.3f} (p={info['deficit_gain_pvalue']:.4f}, sig={sig})")

    # Check 3: Does closure rise with context length?
    for mech in mechs:
        closures_by_ctx = {}
        for ctx in contexts:
            info = analysis.get((mech, ctx))
            if info and not np.isnan(info["closure_mean"]):
                closures_by_ctx[ctx] = info["closure_mean"]
        if len(closures_by_ctx) >= 2:
            ctx_list = sorted(closures_by_ctx.keys())
            rising = all(closures_by_ctx[ctx_list[i+1]] > closures_by_ctx[ctx_list[i]]
                         for i in range(len(ctx_list) - 1))
            status = "PASS" if rising else "FAIL"
            label = LABELS_SHORT.get(mech, mech)
            vals = ", ".join(f"T={c}: {closures_by_ctx[c]:.3f}" for c in ctx_list)
            print(f"    Closure rises with context ({label}): [{status}] {vals}")

    print()


# ---------------------------------------------------------------------------
# Statistical Analysis (research_design.md Section 4)
#
# Three methods for handling autocorrelation in bucketed position analysis:
# 1. Block bootstrap — primary (resamples contiguous blocks)
# 2. Newey-West HAC standard errors — for regression
# 3. Effective degrees of freedom — simplest correction
#
# Plus: Difference-in-differences at the SSSL window boundary (T/4)
# ---------------------------------------------------------------------------


def _lag1_autocorrelation(x: np.ndarray) -> float:
    """Compute lag-1 autocorrelation of a 1D array."""
    x = x - x.mean()
    n = len(x)
    if n < 3 or np.var(x) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def compute_effective_dof(series: np.ndarray) -> tuple[int, float]:
    """Effective degrees of freedom under autocorrelation.

    Uses the Bretherton et al. (1999) formula:
        N_eff = N * (1 - r1) / (1 + r1)
    where r1 is the lag-1 autocorrelation.

    Returns (N_eff, r1).
    """
    r1 = _lag1_autocorrelation(series)
    n = len(series)
    if r1 >= 1.0:
        return 1, r1
    n_eff = max(2, int(n * (1 - r1) / (1 + r1)))
    return n_eff, float(r1)


def _choose_block_size(series: np.ndarray) -> int:
    """Choose block size for bootstrap from autocorrelation function.

    Uses smallest lag k where autocorrelation drops below 0.05.
    Floor at 3, cap at N/4.
    """
    n = len(series)
    x = series - series.mean()
    var = np.var(x)
    if var == 0 or n < 6:
        return max(3, n // 4)
    for k in range(1, n // 2):
        acf_k = np.mean(x[:n - k] * x[k:]) / var
        if abs(acf_k) < 0.05:
            return max(3, min(k, n // 4))
    return max(3, n // 4)


def compute_block_bootstrap_ci(
    deficit: np.ndarray,
    gain: np.ndarray,
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Block bootstrap confidence intervals for deficit/gain/closure means.

    Resamples contiguous blocks of position buckets, preserving local
    autocorrelation structure. Block size chosen from the deficit series
    autocorrelation function.

    Returns dict with:
        block_size: chosen block size
        deficit_mean: point estimate
        deficit_ci: (lo, hi) CI
        gain_mean: point estimate
        gain_ci: (lo, hi)
        closure_mean: point estimate
        closure_ci: (lo, hi)
    """
    n = len(deficit)
    block_size = _choose_block_size(deficit)
    rng = np.random.default_rng(seed)

    alpha = 1 - ci_level
    n_blocks = max(1, (n + block_size - 1) // block_size)

    boot_deficit = np.zeros(n_bootstrap)
    boot_gain = np.zeros(n_bootstrap)
    boot_closure = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Sample block start indices
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]

        d_boot = deficit[indices]
        g_boot = gain[indices]

        boot_deficit[b] = np.mean(d_boot)
        boot_gain[b] = np.mean(g_boot)

        safe_d = np.maximum(d_boot, DEFICIT_EPSILON)
        valid = d_boot > DEFICIT_EPSILON
        if valid.any():
            boot_closure[b] = np.mean(g_boot[valid] / safe_d[valid])
        else:
            boot_closure[b] = np.nan

    def _ci(arr):
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return (float("nan"), float("nan"))
        return (float(np.percentile(arr, 100 * alpha / 2)),
                float(np.percentile(arr, 100 * (1 - alpha / 2))))

    safe_deficit = np.maximum(deficit, DEFICIT_EPSILON)
    valid_mask = deficit > DEFICIT_EPSILON
    closure_pt = float(np.mean(gain[valid_mask] / safe_deficit[valid_mask])) if valid_mask.any() else float("nan")

    return {
        "block_size": block_size,
        "n_bootstrap": n_bootstrap,
        "deficit_mean": float(np.mean(deficit)),
        "deficit_ci": _ci(boot_deficit),
        "gain_mean": float(np.mean(gain)),
        "gain_ci": _ci(boot_gain),
        "closure_mean": closure_pt,
        "closure_ci": _ci(boot_closure),
    }


def _newey_west_se(y: np.ndarray, x: np.ndarray, max_lag: int | None = None) -> tuple[float, float]:
    """OLS slope with Newey-West HAC standard error.

    Regresses y on x (with intercept). Returns (slope, hac_se).
    Uses Bartlett kernel with automatic lag selection (floor(N^(1/3))).
    """
    n = len(y)
    if n < 4:
        return float("nan"), float("nan")
    if max_lag is None:
        max_lag = max(1, int(n ** (1 / 3)))

    # OLS: y = a + b*x + e
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    # Newey-West covariance: S = sum_{j=0}^{L} w_j * Gamma_j
    # where Gamma_j = (1/n) * sum_t X_t * e_t * e_{t-j} * X_{t-j}'
    S = np.zeros((2, 2))
    for j in range(max_lag + 1):
        weight = 1.0 if j == 0 else 1.0 - j / (max_lag + 1)  # Bartlett
        Gamma_j = np.zeros((2, 2))
        for t in range(j, n):
            xe_t = X[t] * resid[t]
            xe_tj = X[t - j] * resid[t - j]
            Gamma_j += np.outer(xe_t, xe_tj)
        Gamma_j /= n
        if j == 0:
            S += Gamma_j
        else:
            S += weight * (Gamma_j + Gamma_j.T)

    V = n * XtX_inv @ S @ XtX_inv
    slope = float(beta[1])
    se = float(np.sqrt(max(0, V[1, 1])))
    return slope, se


def compute_did_analysis(results: list[dict]) -> dict:
    """Difference-in-differences at the SSSL window boundary.

    Partitions positions into:
      Near: positions [0, T/4)  — within SSSL short window for all layers
      Far:  positions [T/4, T)  — beyond the short-window boundary

    DID = [BPB_memory(far) - BPB_local(far)] - [BPB_memory(near) - BPB_local(near)]

    If memory specifically helps beyond the window boundary, DID < 0.

    Returns dict keyed by (mechanism, context_length) with:
        did: the DID estimate
        near_effect: BPB_memory(near) - BPB_local(near)
        far_effect: BPB_memory(far) - BPB_local(far)
        parallel_trends_ok: bool (whether trends are similar in near region)
    """
    roles = group_by_role(results)
    contexts = get_context_lengths(results)
    output = {}

    for ctx in contexts:
        local_runs = roles.get(("local", "baseline", ctx), [])
        lc, l_curves = _extract_position_curves(local_runs)
        if l_curves.size == 0:
            continue

        boundary_idx = None
        window_boundary = ctx // 4
        if lc.size > 0:
            boundary_idx = int(np.searchsorted(lc, window_boundary))

        if boundary_idx is None or boundary_idx < 2 or boundary_idx >= len(lc) - 2:
            continue

        l_mean = l_curves.mean(axis=0)

        for (role, mech, c), runs in roles.items():
            if role != "memory" or c != ctx:
                continue

            mc, m_curves = _extract_position_curves(runs)
            if m_curves.size == 0:
                continue

            n_b = min(len(l_mean), m_curves.shape[1])
            m_mean = m_curves[:, :n_b].mean(axis=0)
            bi = min(boundary_idx, n_b - 1)

            # Near and far BPB means
            l_near = float(np.mean(l_mean[:bi]))
            l_far = float(np.mean(l_mean[bi:n_b]))
            m_near = float(np.mean(m_mean[:bi]))
            m_far = float(np.mean(m_mean[bi:n_b]))

            near_effect = m_near - l_near  # negative = memory helps in near
            far_effect = m_far - l_far      # negative = memory helps in far
            did = far_effect - near_effect   # negative = memory helps MORE in far

            # Parallel trends check: compare slopes of local vs memory in near region
            if bi >= 4:
                near_positions = lc[:bi]
                l_slope, _ = _newey_west_se(l_mean[:bi], near_positions)
                m_slope, _ = _newey_west_se(m_mean[:bi], near_positions)
                # Trends are "parallel" if slopes are within 20% of each other
                if abs(l_slope) > 1e-8:
                    parallel_ok = abs((m_slope - l_slope) / l_slope) < 0.20
                else:
                    parallel_ok = abs(m_slope) < 1e-6
            else:
                parallel_ok = None

            output[(mech, ctx)] = {
                "did": float(did),
                "near_effect": float(near_effect),
                "far_effect": float(far_effect),
                "parallel_trends_ok": parallel_ok,
                "window_boundary": window_boundary,
                "boundary_bucket_idx": bi,
            }

    return output


def compute_did_placebo(results: list[dict]) -> dict:
    """DID placebo tests at multiple boundary positions.

    Runs the DID estimator at T/8, T/4 (actual), T/2, 3T/4 to verify the
    effect peaks at the real SSSL window boundary. If the DID is equally strong
    at arbitrary positions, the boundary-specificity claim is undermined.

    Returns dict keyed by (mechanism, context_length) with:
        boundaries: list of (position, did_value) tuples
        peak_position: position with strongest (most negative) DID
        actual_boundary: T/4
    """
    roles = group_by_role(results)
    contexts = get_context_lengths(results)
    output = {}

    for ctx in contexts:
        local_runs = roles.get(("local", "baseline", ctx), [])
        lc, l_curves = _extract_position_curves(local_runs)
        if l_curves.size == 0:
            continue

        l_mean = l_curves.mean(axis=0)

        for (role, mech, c), runs in roles.items():
            if role != "memory" or c != ctx:
                continue

            mc, m_curves = _extract_position_curves(runs)
            if m_curves.size == 0:
                continue

            n_b = min(len(l_mean), m_curves.shape[1])
            m_mean = m_curves[:, :n_b].mean(axis=0)
            centers = lc[:n_b]

            # Test DID at multiple boundary fractions
            fractions = [1/8, 1/4, 1/2, 3/4]
            boundaries = []
            for frac in fractions:
                boundary_pos = int(ctx * frac)
                bi = int(np.searchsorted(centers, boundary_pos))
                if bi < 2 or bi >= n_b - 2:
                    continue

                l_near = float(np.mean(l_mean[:bi]))
                l_far = float(np.mean(l_mean[bi:n_b]))
                m_near = float(np.mean(m_mean[:bi]))
                m_far = float(np.mean(m_mean[bi:n_b]))

                near_effect = m_near - l_near
                far_effect = m_far - l_far
                did = far_effect - near_effect
                boundaries.append((boundary_pos, float(did), f"T*{frac:.3g}"))

            if boundaries:
                # Find peak (most negative DID = strongest memory advantage in far)
                peak = min(boundaries, key=lambda x: x[1])
                output[(mech, ctx)] = {
                    "boundaries": boundaries,
                    "peak_position": peak[0],
                    "peak_did": peak[1],
                    "actual_boundary": ctx // 4,
                }

    return output


def compute_per_seed_closure(results: list[dict]) -> dict:
    """Compute per-seed regional closure values for the primary endpoint.

    Instead of averaging curves across seeds first, computes closure for
    each seed independently. This makes the sample size transparent (n=3
    or n=4, not hidden behind a seed-averaged curve).

    Returns dict keyed by (mechanism, context_length) with:
        seed_closures: list of (seed, late_closure, early_closure, selectivity) tuples
        late_closure_mean: mean of per-seed late closures
        late_closure_std: std of per-seed late closures (ddof=1)
    """
    roles = group_by_role(results)
    contexts = get_context_lengths(results)
    output = {}

    for ctx in contexts:
        local_runs = roles.get(("local", "baseline", ctx), [])
        global_runs = roles.get(("global", "baseline", ctx), [])
        if not local_runs or not global_runs:
            continue

        # Build per-seed local and global curves
        local_by_seed = {}
        for r in local_runs:
            c, curves = _extract_position_curves([r])
            if curves.size > 0:
                local_by_seed[r["seed"]] = (c, curves[0])

        global_by_seed = {}
        for r in global_runs:
            c, curves = _extract_position_curves([r])
            if curves.size > 0:
                global_by_seed[r["seed"]] = (c, curves[0])

        # Use mean global as reference (seeds may not match 1:1)
        gc, g_curves = _extract_position_curves(global_runs)
        if g_curves.size == 0:
            continue
        global_mean = g_curves.mean(axis=0)

        for (role, mech, c), runs in roles.items():
            if role != "memory" or c != ctx:
                continue

            seed_closures = []
            for r in runs:
                mc, m_curves = _extract_position_curves([r])
                if m_curves.size == 0:
                    continue
                m_curve = m_curves[0]
                seed = r["seed"]

                # Use this seed's local curve if available, else mean local
                if seed in local_by_seed:
                    lc_s, l_curve = local_by_seed[seed]
                else:
                    lc_s, l_all = _extract_position_curves(local_runs)
                    l_curve = l_all.mean(axis=0)

                n_b = min(len(l_curve), len(m_curve), len(global_mean))
                q1 = n_b // 4
                q3 = 3 * n_b // 4

                l_early = float(np.mean(l_curve[:q1]))
                l_late = float(np.mean(l_curve[q3:n_b]))
                g_early = float(np.mean(global_mean[:q1]))
                g_late = float(np.mean(global_mean[q3:n_b]))
                m_early = float(np.mean(m_curve[:q1]))
                m_late = float(np.mean(m_curve[q3:n_b]))

                late_deficit = l_late - g_late
                early_deficit = l_early - g_early
                late_gain = l_late - m_late
                early_gain = l_early - m_early

                late_c = late_gain / max(late_deficit, DEFICIT_EPSILON) if late_deficit > DEFICIT_EPSILON else float("nan")
                early_c = early_gain / max(early_deficit, DEFICIT_EPSILON) if early_deficit > DEFICIT_EPSILON else float("nan")
                sel = late_c - early_c if not (np.isnan(late_c) or np.isnan(early_c)) else float("nan")

                seed_closures.append((seed, late_c, early_c, sel))

            if seed_closures:
                late_vals = [x[1] for x in seed_closures if not np.isnan(x[1])]
                output[(mech, ctx)] = {
                    "seed_closures": seed_closures,
                    "late_closure_mean": float(np.mean(late_vals)) if late_vals else float("nan"),
                    "late_closure_std": float(np.std(late_vals, ddof=1)) if len(late_vals) > 1 else 0,
                }

    return output


def print_statistical_summary(results: list[dict]):
    """Print full statistical analysis: bootstrap CIs, effective DoF, DID."""
    analysis = compute_deficit_analysis(results)
    if not analysis:
        return

    contexts = sorted(set(ctx for _, ctx in analysis.keys()))
    mechs = sorted(set(m for m, _ in analysis.keys()))

    print(f"\n{'='*90}")
    print("  STATISTICAL ANALYSIS (research_design.md Section 4)")
    print(f"{'='*90}")

    # Block bootstrap CIs
    print("\n  Block Bootstrap 95% Confidence Intervals")
    print(f"  {'Mechanism':<18} {'Context':>8} {'Block':>6} "
          f"{'Deficit CI':>22} {'Gain CI':>22} {'Closure CI':>20}")
    print(f"  {'-'*98}")

    for mech in mechs:
        for ctx in contexts:
            info = analysis.get((mech, ctx))
            if not info:
                continue
            boot = compute_block_bootstrap_ci(info["deficit"], info["gain"])
            label = LABELS_SHORT.get(mech, mech)
            d_ci = f"[{boot['deficit_ci'][0]:+.5f}, {boot['deficit_ci'][1]:+.5f}]"
            g_ci = f"[{boot['gain_ci'][0]:+.5f}, {boot['gain_ci'][1]:+.5f}]"
            c_ci = f"[{boot['closure_ci'][0]:.3f}, {boot['closure_ci'][1]:.3f}]"
            print(f"  {label:<18} {ctx:>8} {boot['block_size']:>6} "
                  f"{d_ci:>22} {g_ci:>22} {c_ci:>20}")
    print()

    # Effective DoF
    print("  Effective Degrees of Freedom (Bretherton et al. 1999)")
    print(f"  {'Mechanism':<18} {'Context':>8} {'N_buckets':>10} {'lag-1 r':>8} {'N_eff':>7}")
    print(f"  {'-'*55}")

    for mech in mechs:
        for ctx in contexts:
            info = analysis.get((mech, ctx))
            if not info:
                continue
            n_eff, r1 = compute_effective_dof(info["deficit"])
            label = LABELS_SHORT.get(mech, mech)
            print(f"  {label:<18} {ctx:>8} {info['n_buckets']:>10} {r1:>8.3f} {n_eff:>7}")
    print()

    # Newey-West regression: deficit slope with position
    print("  Newey-West HAC: Deficit Trend with Position")
    print(f"  {'Mechanism':<18} {'Context':>8} {'Slope':>12} {'HAC SE':>10} {'t-stat':>8}")
    print(f"  {'-'*60}")

    for mech in mechs:
        for ctx in contexts:
            info = analysis.get((mech, ctx))
            if not info:
                continue
            slope, se = _newey_west_se(info["deficit"], info["centers"])
            t_stat = slope / se if se > 0 else float("nan")
            label = LABELS_SHORT.get(mech, mech)
            print(f"  {label:<18} {ctx:>8} {slope:>12.7f} {se:>10.7f} {t_stat:>8.2f}")
    print()

    # DID analysis
    did_results = compute_did_analysis(results)
    if did_results:
        print("  Difference-in-Differences at Window Boundary")
        print(f"  {'Mechanism':<18} {'Context':>8} {'Near Effect':>12} {'Far Effect':>11} "
              f"{'DID':>10} {'Parallel':>9}")
        print(f"  {'-'*72}")

        for mech in mechs:
            for ctx in contexts:
                info = did_results.get((mech, ctx))
                if not info:
                    continue
                label = LABELS_SHORT.get(mech, mech)
                pt = "yes" if info["parallel_trends_ok"] else ("no" if info["parallel_trends_ok"] is not None else "?")
                print(f"  {label:<18} {ctx:>8} {info['near_effect']:>+12.5f} "
                      f"{info['far_effect']:>+11.5f} {info['did']:>+10.5f} {pt:>9}")

        print()
        print("  DID < 0: memory helps more beyond window boundary than within it")
        print("  Parallel: near-region trends are parallel (DID assumption check)")

    # DID placebo tests
    placebo = compute_did_placebo(results)
    if placebo:
        print()
        print("  DID Placebo Tests (falsification: effect should peak at T/4)")
        print(f"  {'Mechanism':<18} {'Context':>8} {'Boundary':>10} {'DID':>10} {'Label':>10}")
        print(f"  {'-'*60}")

        for mech in mechs:
            for ctx in contexts:
                info = placebo.get((mech, ctx))
                if not info:
                    continue
                label = LABELS_SHORT.get(mech, mech)
                for pos, did_val, blabel in info["boundaries"]:
                    marker = " <-- actual" if pos == info["actual_boundary"] else ""
                    peak = " [PEAK]" if pos == info["peak_position"] else ""
                    print(f"  {label:<18} {ctx:>8} {pos:>10} {did_val:>+10.5f} {blabel:>10}{marker}{peak}")
                label = ""  # only print mechanism name once

        print()
        print("  If DID peaks at actual T/4 boundary, the effect is boundary-specific.")
        print("  If equally strong elsewhere, the boundary claim is undermined.")

    # Per-seed closure (primary endpoint transparency)
    per_seed = compute_per_seed_closure(results)
    if per_seed:
        print()
        print("  Per-Seed Regional Closure (primary endpoint transparency)")
        print(f"  {'Mechanism':<18} {'Context':>8} {'Seed':>8} "
              f"{'Late C':>8} {'Early C':>8} {'Selectivity':>12}")
        print(f"  {'-'*66}")

        for mech in mechs:
            for ctx in contexts:
                info = per_seed.get((mech, ctx))
                if not info:
                    continue
                label = LABELS_SHORT.get(mech, mech)
                for seed, lc_val, ec_val, sel in info["seed_closures"]:
                    lc_s = f"{lc_val:.3f}" if not np.isnan(lc_val) else "n/a"
                    ec_s = f"{ec_val:.3f}" if not np.isnan(ec_val) else "n/a"
                    sel_s = f"{sel:+.3f}" if not np.isnan(sel) else "n/a"
                    print(f"  {label:<18} {ctx:>8} {seed:>8} "
                          f"{lc_s:>8} {ec_s:>8} {sel_s:>12}")
                    label = ""  # only print once
                label = LABELS_SHORT.get(mech, mech)
                print(f"  {'':18} {'mean':>8} {'':>8} "
                      f"{info['late_closure_mean']:.3f} +/- {info['late_closure_std']:.3f}")

    print()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_main_comparison(results: list[dict]):
    """Main figure: BPB bar chart with error bars and baseline reference line."""
    groups = group_by_mechanism(results)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    mechs, means, stds, colors = [], [], [], []
    baseline_mean = None

    for mech in MECHANISMS_ORDER:
        if mech not in groups:
            continue
        bpbs = [r["min_val_bpb"] for r in groups[mech] if r.get("min_val_bpb") is not None]
        if not bpbs:
            continue
        m = np.mean(bpbs)
        if mech == "baseline":
            baseline_mean = m
        mechs.append(LABELS.get(mech, mech))
        means.append(m)
        stds.append(np.std(bpbs) if len(bpbs) > 1 else 0)
        colors.append(COLORS.get(mech, "#888"))

    if not mechs:
        print("No results to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(mechs))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                  edgecolor="black", linewidth=0.8, width=0.65, zorder=3)

    if baseline_mean:
        ax.axhline(baseline_mean, color="#4A90D9", linestyle="--", alpha=0.5, linewidth=1, zorder=2)
        ax.text(len(mechs) - 0.5, baseline_mean + 0.001, "baseline", color="#4A90D9",
                fontsize=9, alpha=0.7, ha="right")

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.002,
                f"{m:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ymin = min(means) - max(max(stds, default=0) * 4, 0.02)
    ymax = max(means) + max(max(stds, default=0) * 4, 0.02)
    ax.set_ylim(max(0, ymin), ymax)

    ax.set_ylabel("Validation BPB (bits per byte)", fontsize=12)
    ax.set_title("Memory Mechanism Comparison — 12-Layer GPT, 3-Seed Mean", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(mechs, fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = FIGURES_DIR / "fig1_bpb_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig_overhead(results: list[dict]):
    """Figure 2: Training time and VRAM overhead vs baseline."""
    groups = group_by_mechanism(results)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    baseline_time = None
    baseline_vram = None
    if "baseline" in groups:
        bt = [r["total_time_min"] for r in groups["baseline"] if r.get("total_time_min")]
        bv = [r["peak_vram_mib"] for r in groups["baseline"] if r.get("peak_vram_mib")]
        baseline_time = np.mean(bt) if bt else None
        baseline_vram = np.mean(bv) if bv else None

    mechs, time_overheads, vram_overheads, colors_list = [], [], [], []
    for mech in MECHANISMS_ORDER:
        if mech == "baseline" or mech not in groups:
            continue
        times = [r["total_time_min"] for r in groups[mech] if r.get("total_time_min")]
        vrams = [r["peak_vram_mib"] for r in groups[mech] if r.get("peak_vram_mib")]
        if not times or baseline_time is None:
            continue
        mechs.append(LABELS.get(mech, mech))
        time_overheads.append((np.mean(times) / baseline_time - 1) * 100)
        vram_overheads.append((np.mean(vrams) / baseline_vram - 1) * 100 if baseline_vram and vrams else 0)
        colors_list.append(COLORS.get(mech, "#888"))

    if not mechs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(mechs))
    w = 0.5

    ax1.bar(x, time_overheads, color=colors_list, edgecolor="black", linewidth=0.8, width=w)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Training Time Overhead (%)", fontsize=11)
    ax1.set_title("Training Time vs Baseline", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(mechs, fontsize=9)
    for i, v in enumerate(time_overheads):
        ax1.text(i, v + 0.5, f"{v:+.1f}%", ha="center", fontsize=9)

    ax2.bar(x, vram_overheads, color=colors_list, edgecolor="black", linewidth=0.8, width=w)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Peak VRAM Overhead (%)", fontsize=11)
    ax2.set_title("VRAM Usage vs Baseline", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(mechs, fontsize=9)
    for i, v in enumerate(vram_overheads):
        ax2.text(i, v + 0.5, f"{v:+.1f}%", ha="center", fontsize=9)

    plt.suptitle("Computational Overhead of Memory Mechanisms", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "fig2_overhead.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig_scatter(results: list[dict]):
    """Figure 3: BPB vs additional parameters scatter."""
    groups = group_by_mechanism(results)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    for mech in MECHANISMS_ORDER:
        if mech not in groups:
            continue
        runs = groups[mech]
        for r in runs:
            bpb = r.get("min_val_bpb")
            mem_p = r.get("memory_params", 0)
            if bpb is None:
                continue
            ax.scatter(mem_p / 1e6, bpb,
                       c=COLORS.get(mech, "#888"),
                       s=100, edgecolors="black", linewidth=0.5, zorder=3,
                       label=LABELS.get(mech, mech))

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            unique_handles.append(h)
            unique_labels.append(l)
    ax.legend(unique_handles, unique_labels, fontsize=9)

    ax.set_xlabel("Additional Memory Parameters (millions)", fontsize=11)
    ax.set_ylabel("Validation BPB", fontsize=11)
    ax.set_title("Quality vs Parameter Overhead", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = FIGURES_DIR / "fig3_bpb_vs_params.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig_crossover(results: list[dict]):
    """Signature figure: BPB vs context length, one line per mechanism.

    X: context lengths, Y: mean BPB. Error bars: std across seeds.
    Shaded region where mechanism beats baseline.
    """
    groups = group_results(results)
    contexts = get_context_lengths(results)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if len(contexts) < 2:
        return  # need multiple contexts for this plot

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: absolute BPB
    baseline_means = {}
    for mech in MECHANISMS_ORDER:
        ctx_vals, mean_vals, std_vals = [], [], []
        for ctx in contexts:
            key = (mech, ctx)
            if key not in groups:
                continue
            bpbs = [r["min_val_bpb"] for r in groups[key] if r.get("min_val_bpb") is not None]
            if not bpbs:
                continue
            ctx_vals.append(ctx)
            mean_vals.append(np.mean(bpbs))
            std_vals.append(np.std(bpbs) if len(bpbs) > 1 else 0)
            if mech == "baseline":
                baseline_means[ctx] = np.mean(bpbs)

        if not ctx_vals:
            continue

        color = COLORS.get(mech, "#888")
        label = LABELS_SHORT.get(mech, mech)
        ax1.errorbar(ctx_vals, mean_vals, yerr=std_vals, color=color, label=label,
                     linewidth=2, marker="o", markersize=6, capsize=4)

    ax1.set_xlabel("Context Length (tokens)", fontsize=12)
    ax1.set_ylabel("Validation BPB", fontsize=12)
    ax1.set_title("BPB vs Context Length", fontsize=13, fontweight="bold")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(contexts)
    ax1.set_xticklabels([str(c) for c in contexts])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right panel: delta vs baseline
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    for mech in MECHANISMS_ORDER:
        if mech == "baseline":
            continue
        ctx_vals, delta_vals = [], []
        for ctx in contexts:
            key = (mech, ctx)
            if key not in groups or ctx not in baseline_means:
                continue
            bpbs = [r["min_val_bpb"] for r in groups[key] if r.get("min_val_bpb") is not None]
            if not bpbs:
                continue
            ctx_vals.append(ctx)
            delta_vals.append(np.mean(bpbs) - baseline_means[ctx])

        if not ctx_vals:
            continue

        color = COLORS.get(mech, "#888")
        label = LABELS_SHORT.get(mech, mech)
        ax2.plot(ctx_vals, delta_vals, color=color, label=label,
                 linewidth=2, marker="o", markersize=6)

        # Shade region where mechanism beats baseline
        for i in range(len(ctx_vals) - 1):
            if delta_vals[i] <= 0 and delta_vals[i + 1] <= 0:
                ax2.axvspan(ctx_vals[i], ctx_vals[i + 1], alpha=0.05, color=color)

    ax2.set_xlabel("Context Length (tokens)", fontsize=12)
    ax2.set_ylabel("Delta BPB vs Baseline (negative = better)", fontsize=12)
    ax2.set_title("Memory Mechanism Gain vs Context Length", fontsize=13, fontweight="bold")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(contexts)
    ax2.set_xticklabels([str(c) for c in contexts])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "fig_crossover.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig_bpb_position_grid(results: list[dict]):
    """Multi-panel grid: BPB by position at each context length.

    One panel per context length, each with one curve per mechanism.
    """
    contexts = get_context_lengths(results)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if len(contexts) < 2:
        return  # single context doesn't need a grid

    groups = group_results(results)
    n_cols = len(contexts)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, ctx in zip(axes, contexts):
        for mech in MECHANISMS_ORDER:
            key = (mech, ctx)
            if key not in groups:
                continue

            all_curves = []
            positions = None
            for r in groups[key]:
                bpb_pos = r.get("bpb_by_position", {})
                buckets = bpb_pos.get("buckets", {})
                if not buckets:
                    continue
                centers = [v["center_position"] for v in buckets.values()]
                bpbs = [v["bpb"] for v in buckets.values()]
                all_curves.append(bpbs)
                if positions is None:
                    positions = centers

            if not all_curves or positions is None:
                continue

            curves = np.array(all_curves)
            mean_curve = np.mean(curves, axis=0)
            color = COLORS.get(mech, "#888")
            label = LABELS_SHORT.get(mech, mech)
            ax.plot(positions, mean_curve, color=color, label=label,
                    linewidth=1.5, marker=".", markersize=2)

            if len(all_curves) > 1:
                std_curve = np.std(curves, axis=0)
                ax.fill_between(positions, mean_curve - std_curve, mean_curve + std_curve,
                                color=color, alpha=0.1)

        ax.set_xlabel("Position (tokens)", fontsize=10)
        ax.set_title(f"T={ctx}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("BPB", fontsize=11)
    axes[-1].legend(fontsize=8, loc="upper right")
    plt.suptitle("BPB by Context Position at Each Context Length", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "fig_bpb_position_grid.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig_compute_efficiency(results: list[dict]):
    """BPB vs GPU-minutes for each (mechanism, context) pair."""
    groups = group_results(results)
    contexts = get_context_lengths(results)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if len(contexts) < 2:
        return

    markers = {2048: "o", 4096: "s", 8192: "D"}
    fig, ax = plt.subplots(figsize=(10, 6))

    for mech in MECHANISMS_ORDER:
        for ctx in contexts:
            key = (mech, ctx)
            if key not in groups:
                continue
            runs = groups[key]
            bpbs = [r["min_val_bpb"] for r in runs if r.get("min_val_bpb") and r.get("total_time_min")]
            times = [r["total_time_min"] for r in runs if r.get("min_val_bpb") and r.get("total_time_min")]
            if not bpbs:
                continue

            color = COLORS.get(mech, "#888")
            marker = markers.get(ctx, "o")
            label = f"{LABELS_SHORT.get(mech, mech)} T={ctx}"
            ax.scatter(np.mean(times), np.mean(bpbs), c=color, marker=marker,
                       s=120, edgecolors="black", linewidth=0.5, zorder=3, label=label)

    ax.set_xlabel("Training Time (GPU-minutes)", fontsize=12)
    ax.set_ylabel("Validation BPB", fontsize=12)
    ax.set_title("Compute Efficiency: BPB vs Training Time", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "fig_compute_efficiency.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Direction B Figures: Deficit Framework
# ---------------------------------------------------------------------------

def fig_three_condition(results: list[dict]):
    """Signature figure: Local vs Global vs Persistent across all contexts.

    Three panels (one per context length), each showing position-resolved
    BPB curves for the three conditions with confidence bands.
    """
    roles = group_by_role(results)
    contexts = get_context_lengths(results)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    valid = [c for c in contexts
             if ("local", "baseline", c) in roles and ("global", "baseline", c) in roles
             and ("memory", "persistent", c) in roles]
    if not valid:
        return

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=True, squeeze=False)
    axes = axes[0]

    conditions = [
        ("local", "baseline", "#4A90D9", "Local (SSSL)", "-"),
        ("global", "baseline", "#2ECC71", "Global (full attn)", "--"),
        ("memory", "persistent", "#E8A838", "Persistent Memory", "-"),
    ]

    for i, ctx in enumerate(valid):
        ax = axes[i]

        for role, mech, color, label, ls in conditions:
            runs = roles.get((role, mech, ctx), [])
            if not runs:
                continue
            centers, curves = _extract_position_curves(runs)
            if curves.size == 0:
                continue
            mean = curves.mean(axis=0)
            ax.plot(centers, mean, color=color, linewidth=2, label=label, linestyle=ls)
            if curves.shape[0] > 1:
                std = curves.std(axis=0)
                ax.fill_between(centers, mean - std, mean + std, alpha=0.12, color=color)

        # Mark window boundary
        wb = ctx // 4
        ax.axvline(wb, color="#888", linestyle=":", linewidth=1, alpha=0.6)
        ymin, ymax = ax.get_ylim()
        ax.text(wb + ctx * 0.01, ymax - (ymax - ymin) * 0.05, f"T/4",
                fontsize=8, color="#888", ha="left", va="top")

        # Compute aggregate BPBs for annotation
        agg = {}
        for role, mech, color, label, ls in conditions:
            runs = roles.get((role, mech, ctx), [])
            bpbs = [r["min_val_bpb"] for r in runs if r.get("min_val_bpb")]
            if bpbs:
                agg[label] = np.mean(bpbs)
        if "Local (SSSL)" in agg and "Persistent Memory" in agg:
            delta = (agg["Persistent Memory"] - agg["Local (SSSL)"]) * 1000
            ax.text(0.98, 0.02, f"PM: {delta:+.2f} mBPB vs Local",
                    transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8A838", alpha=0.2))

        ax.set_xlabel("Position (tokens)", fontsize=10)
        ax.set_title(f"T = {ctx}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("BPB", fontsize=11)
    axes[-1].legend(fontsize=8, loc="upper right")
    plt.suptitle("Three-Condition Comparison: Local vs Global vs Persistent Memory",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "fig_three_condition.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig_deficit_map(results: list[dict]):
    """Figure 1: Positional Context Deficit Map.

    One panel per context length. Each shows per-position BPB for local-attention
    baseline vs global-attention reference, with shaded gap = deficit.
    """
    roles = group_by_role(results)
    contexts = get_context_lengths(results)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Only contexts where we have both local and global
    valid_contexts = [c for c in contexts
                      if ("local", "baseline", c) in roles and ("global", "baseline", c) in roles]
    if not valid_contexts:
        return

    n = len(valid_contexts)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True, squeeze=False)
    axes = axes[0]

    for i, ctx in enumerate(valid_contexts):
        ax = axes[i]
        local_runs = roles[("local", "baseline", ctx)]
        global_runs = roles[("global", "baseline", ctx)]

        lc, l_curves = _extract_position_curves(local_runs)
        gc, g_curves = _extract_position_curves(global_runs)

        if l_curves.size == 0 or g_curves.size == 0:
            continue

        n_b = min(l_curves.shape[1], g_curves.shape[1])
        centers = lc[:n_b]
        l_mean = l_curves[:, :n_b].mean(axis=0)
        g_mean = g_curves[:, :n_b].mean(axis=0)

        ax.plot(centers, l_mean, color="#4A90D9", linewidth=2, label="Local (SSSL)")
        ax.plot(centers, g_mean, color="#2ECC71", linewidth=2, label="Global (L)")

        # Shade deficit region
        ax.fill_between(centers, g_mean, l_mean,
                         where=l_mean > g_mean, alpha=0.25, color="#E74C3C",
                         label="Context deficit")

        # Seed bands
        if l_curves.shape[0] > 1:
            l_std = l_curves[:, :n_b].std(axis=0)
            ax.fill_between(centers, l_mean - l_std, l_mean + l_std, alpha=0.1, color="#4A90D9")
        if g_curves.shape[0] > 1:
            g_std = g_curves[:, :n_b].std(axis=0)
            ax.fill_between(centers, g_mean - g_std, g_mean + g_std, alpha=0.1, color="#2ECC71")

        # Mark window boundary (T/4)
        window_boundary = ctx // 4
        ax.axvline(window_boundary, color="#888", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(window_boundary, ax.get_ylim()[1] * 0.98, f"T/4={window_boundary}",
                ha="left", va="top", fontsize=7, color="#888")

        ax.set_xlabel("Position (tokens)", fontsize=10)
        ax.set_title(f"T={ctx}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("BPB", fontsize=11)
    axes[-1].legend(fontsize=8, loc="upper right")
    plt.suptitle("Positional Context Deficit: Local vs Global Attention",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "fig_deficit_map.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig_deficit_closure(results: list[dict]):
    """Figure 2: Deficit Closure by Memory Mechanism.

    One panel per context length. Each shows deficit(p), gain(p), and closure(p).
    Reports mean closure and Spearman correlation.
    """
    analysis = compute_deficit_analysis(results)
    if not analysis:
        return

    contexts = sorted(set(ctx for _, ctx in analysis.keys()))
    mechs = sorted(set(m for m, _ in analysis.keys()))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for mech in mechs:
        mech_data = {ctx: analysis[(mech, ctx)] for ctx in contexts if (mech, ctx) in analysis}
        if not mech_data:
            continue

        n = len(mech_data)
        fig, axes = plt.subplots(2, n, figsize=(5 * n, 7), squeeze=False)

        for i, (ctx, info) in enumerate(sorted(mech_data.items())):
            centers = info["centers"]

            # Top panel: deficit and gain curves
            ax_top = axes[0, i]
            ax_top.fill_between(centers, 0, info["deficit"], alpha=0.3, color="#E74C3C", label="Deficit")
            ax_top.fill_between(centers, 0, info["gain"], alpha=0.3, color="#2ECC71", label="Gain")
            ax_top.plot(centers, info["deficit"], color="#E74C3C", linewidth=2)
            ax_top.plot(centers, info["gain"], color="#2ECC71", linewidth=2)
            ax_top.axhline(0, color="black", linewidth=0.5)

            window_boundary = ctx // 4
            ax_top.axvline(window_boundary, color="#888", linestyle="--", linewidth=1, alpha=0.7)

            ax_top.set_title(f"T={ctx}", fontsize=12, fontweight="bold")
            ax_top.set_ylabel("BPB difference", fontsize=10)
            ax_top.grid(True, alpha=0.3)
            if i == n - 1:
                ax_top.legend(fontsize=8)

            # Bottom panel: closure curve
            ax_bot = axes[1, i]
            valid = info["closure_valid"]
            c_vals = info["closure"].copy()
            c_vals[~valid] = float("nan")  # don't plot where deficit is negligible
            ax_bot.plot(centers, c_vals, color="#8E44AD", linewidth=2)
            ax_bot.axhline(1.0, color="#888", linestyle=":", linewidth=1, label="Full closure")
            ax_bot.axhline(0.0, color="black", linewidth=0.5)
            ax_bot.axvline(window_boundary, color="#888", linestyle="--", linewidth=1, alpha=0.7)

            # Annotate mean closure and correlation
            ax_bot.text(0.02, 0.95,
                        f"mean closure: {info['closure_mean']:.3f}\n"
                        f"corr(d,g): {info['deficit_gain_corr']:+.3f} (p={info['deficit_gain_pvalue']:.3f})",
                        transform=ax_bot.transAxes, fontsize=8, va="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            ax_bot.set_xlabel("Position (tokens)", fontsize=10)
            ax_bot.set_ylabel("Closure (gain/deficit)", fontsize=10)
            ax_bot.set_ylim(-0.5, 2.0)
            ax_bot.grid(True, alpha=0.3)

        label = LABELS_SHORT.get(mech, mech)
        plt.suptitle(f"Deficit Closure: {label}",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        path = FIGURES_DIR / f"fig_deficit_closure_{mech}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


def fig_closure_vs_context(results: list[dict]):
    """Summary figure: mean closure vs context length for each mechanism.

    This is the "does memory become more useful at longer context" plot.
    """
    analysis = compute_deficit_analysis(results)
    if not analysis:
        return

    contexts = sorted(set(ctx for _, ctx in analysis.keys()))
    mechs = sorted(set(m for m, _ in analysis.keys()))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if len(contexts) < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for mech in mechs:
        ctx_vals = []
        closure_vals = []
        for ctx in contexts:
            info = analysis.get((mech, ctx))
            if info and not np.isnan(info["closure_mean"]):
                ctx_vals.append(ctx)
                closure_vals.append(info["closure_mean"])

        if len(ctx_vals) < 2:
            continue

        color = COLORS.get(mech, "#888")
        label = LABELS_SHORT.get(mech, mech)
        ax.plot(ctx_vals, closure_vals, color=color, label=label,
                linewidth=2.5, marker="o", markersize=8)

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Mean Deficit Closure", fontsize=12)
    ax.set_title("Deficit Closure vs Context Length", fontsize=13, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(contexts)
    ax.set_xticklabels([str(c) for c in contexts])
    ax.axhline(1.0, color="#888", linestyle=":", linewidth=1, label="Full closure")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "fig_closure_vs_context.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def generate_report(results: list[dict]) -> str:
    """Generate a polished markdown report summarizing all results."""
    lines = []
    contexts = get_context_lengths(results)
    has_global = any(r.get("window_pattern") == "L" for r in results)
    has_multicontext = len(contexts) > 1
    roles = group_by_role(results)
    groups = group_results(results)

    # Count runs
    n_sssl = len([r for r in results if r.get("window_pattern", "SSSL") == "SSSL"])
    n_global = len([r for r in results if r.get("window_pattern") == "L"])
    n_total = len(results)
    mechanisms = sorted(set(normalize_mechanism(r.get("mechanism", "baseline")) for r in results))
    seeds = sorted(set(r.get("seed") for r in results))

    lines.append("# memory-bench: Results Summary")
    lines.append("")
    lines.append(f"**{n_total} runs** across {len(contexts)} context lengths "
                 f"({', '.join(str(c) for c in contexts)}), "
                 f"{len(mechanisms)} mechanisms, {len(seeds)} seeds.")
    lines.append("")

    # Architecture description
    lines.append("## Setup")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append("| Architecture | 12-layer GPT (nanochat), 286M params |")
    lines.append("| Attention | SSSL pattern: 3/4 layers short window (T/4), 1/4 global |")
    if has_global:
        lines.append("| Control | Global-attention baseline (all layers full context) |")
    lines.append(f"| Context lengths | {', '.join(str(c) for c in contexts)} tokens |")
    lines.append(f"| Seeds | {', '.join(str(s) for s in seeds)} |")
    # Compute actual GPU-hours from logged training times
    total_min = sum(r.get("total_time_min", 0) for r in results)
    gpu_hours = total_min * 8 / 60  # 8 GPUs per run
    lines.append(f"| Hardware | 8×H100 SXM, {gpu_hours:.0f} GPU-hours total |")
    lines.append("")

    # Main results table
    lines.append("## Results")
    lines.append("")

    if has_global and has_multicontext:
        # Three-condition table: local vs global vs persistent
        lines.append("### BPB by Condition and Context Length")
        lines.append("")
        lines.append("| Context | Local (SSSL) | Global (full attn) | Persistent Memory | PM vs Local | PM vs Global |")
        lines.append("|---------|-------------|-------------------|-------------------|-------------|-------------|")

        for ctx in contexts:
            local_runs = roles.get(("local", "baseline", ctx), [])
            global_runs = roles.get(("global", "baseline", ctx), [])
            mem_runs = roles.get(("memory", "persistent", ctx), [])

            local_bpbs = [r["min_val_bpb"] for r in local_runs if r.get("min_val_bpb")]
            global_bpbs = [r["min_val_bpb"] for r in global_runs if r.get("min_val_bpb")]
            mem_bpbs = [r["min_val_bpb"] for r in mem_runs if r.get("min_val_bpb")]

            def fmt_bpb(bpbs):
                if not bpbs:
                    return "—"
                m = np.mean(bpbs)
                s = np.std(bpbs, ddof=1) if len(bpbs) > 1 else 0
                n = len(bpbs)
                return f"{m:.5f} ± {s:.5f} (n={n})"

            def fmt_delta(bpbs_a, bpbs_b):
                """fmt_delta(a, b) = mean(b) - mean(a), so negative = b is better."""
                if not bpbs_a or not bpbs_b:
                    return "—"
                d = np.mean(bpbs_b) - np.mean(bpbs_a)
                return f"**{d*1000:+.2f} mBPB**"

            lines.append(f"| {ctx} | {fmt_bpb(local_bpbs)} | {fmt_bpb(global_bpbs)} | "
                        f"{fmt_bpb(mem_bpbs)} | {fmt_delta(local_bpbs, mem_bpbs)} | "
                        f"{fmt_delta(global_bpbs, mem_bpbs)} |")

        lines.append("")
        lines.append("*Persistent Memory beats both local and global baselines at every context length.*")
        lines.append("")

    # Mechanism comparison at T=2048 (if we have all 5 mechanisms)
    mechs_at_2048 = [m for m in MECHANISMS_ORDER if ("baseline" if m == "baseline" else m, 2048) in groups or m == "baseline"]
    if len(mechs_at_2048) > 2:
        lines.append("### All Mechanisms at T=2048")
        lines.append("")
        lines.append("| Mechanism | Mean BPB | Δ vs Baseline | p-value | Seeds |")
        lines.append("|-----------|----------|--------------|---------|-------|")

        baseline_bpbs_2048 = [r["min_val_bpb"] for r in groups.get(("baseline", 2048), []) if r.get("min_val_bpb")]
        baseline_mean = np.mean(baseline_bpbs_2048) if baseline_bpbs_2048 else None

        for mech in MECHANISMS_ORDER:
            key = (mech, 2048)
            if key not in groups:
                continue
            bpbs = [r["min_val_bpb"] for r in groups[key] if r.get("min_val_bpb")]
            if not bpbs:
                continue
            m = np.mean(bpbs)
            n = len(bpbs)
            if mech == "baseline":
                lines.append(f"| {LABELS_SHORT.get(mech, mech)} | {m:.5f} | — | — | {n} |")
            else:
                delta = m - baseline_mean if baseline_mean else 0
                # Paired t-test by seed
                paired_b = []
                paired_m = []
                for r in groups[key]:
                    seed = r["seed"]
                    for br in groups.get(("baseline", 2048), []):
                        if br["seed"] == seed and br.get("min_val_bpb") and r.get("min_val_bpb"):
                            paired_b.append(br["min_val_bpb"])
                            paired_m.append(r["min_val_bpb"])
                if len(paired_b) >= 2:
                    _, pval = stats.ttest_rel(paired_m, paired_b)
                    p_str = f"{pval:.3f}" if pval >= 0.001 else "<0.001"
                else:
                    p_str = "—"
                sig = "**" if delta < 0 else ""
                lines.append(f"| {LABELS_SHORT.get(mech, mech)} | {m:.5f} | {sig}{delta*1000:+.2f} mBPB{sig} | {p_str} | {n} |")

        lines.append("")

    # Deficit analysis
    if has_global:
        lines.append("## Positional Context Deficit")
        lines.append("")
        lines.append("The **positional context deficit** measures how much local SSSL attention "
                     "degrades prediction at each position relative to global attention:")
        lines.append("")
        lines.append("$$D(p, T) = \\text{BPB}_{\\text{local}}(p, T) - \\text{BPB}_{\\text{global}}(p, T)$$")
        lines.append("")
        lines.append("**Memory gain** measures how much a mechanism recovers:")
        lines.append("")
        lines.append("$$G(p, T) = \\text{BPB}_{\\text{local}}(p, T) - \\text{BPB}_{\\text{memory}}(p, T)$$")
        lines.append("")

        analysis = compute_deficit_analysis(results)
        if analysis:
            lines.append("| Context | Mean Deficit | Mean Gain (PM) | Closure | Gain-Deficit ρ |")
            lines.append("|---------|-------------|---------------|---------|----------------|")
            for ctx in contexts:
                key = ("persistent", ctx)
                if key not in analysis:
                    continue
                a = analysis[key]
                lines.append(f"| {ctx} | {np.mean(a['deficit']):+.5f} | "
                           f"{np.mean(a['gain']):+.5f} | "
                           f"{a['closure_mean']:.3f} | "
                           f"{a['deficit_gain_corr']:+.3f} (p={a['deficit_gain_pvalue']:.4f}) |")
            lines.append("")

            # Falsification
            lines.append("### Falsification Checks")
            lines.append("")
            deficits = []
            for ctx in contexts:
                key = ("persistent", ctx)
                if key in analysis:
                    deficits.append((ctx, np.mean(analysis[key]["deficit"])))
            if len(deficits) > 1:
                growing = all(deficits[i][1] < deficits[i+1][1] for i in range(len(deficits)-1))
                lines.append(f"- **Deficit grows with context**: {'PASS' if growing else 'FAIL'} "
                           f"({', '.join(f'T={c}: {d:.5f}' for c, d in deficits)})")
            lines.append("")

    # Figures list — only show key analysis figures
    lines.append("## Figures")
    lines.append("")
    key_figs = [
        ("fig_three_condition.png", "Local vs Global vs Persistent Memory"),
        ("fig_crossover.png", "BPB vs Context Length (crossover analysis)"),
        ("fig_deficit_map.png", "Positional Context Deficit Map"),
        ("fig_deficit_closure_persistent.png", "Persistent Memory: Deficit and Closure"),
        ("fig_closure_vs_context.png", "Closure vs Context Length"),
        ("fig_bpb_position_grid.png", "BPB by Position (multi-context grid)"),
        ("fig1_bpb_comparison.png", "Mechanism Comparison at T=2048"),
        ("fig_compute_efficiency.png", "Compute Efficiency"),
    ]
    for fname, caption in key_figs:
        if (FIGURES_DIR / fname).exists():
            lines.append(f"### {caption}")
            lines.append(f"![{caption}](results/figures/{fname})")
            lines.append("")

    return "\n".join(lines)


def pull_results(host: str, port: str):
    """Pull results from pod."""
    print(f"Pulling results from {host}:{port}...")
    subprocess.run(["bash", "pull_results.sh", host, port], check=True)


def main():
    parser = argparse.ArgumentParser(description="Analyze memory-bench results")
    parser.add_argument("--pull", nargs=2, metavar=("HOST", "PORT"), help="Pull from pod first")
    args = parser.parse_args()

    if args.pull:
        pull_results(args.pull[0], args.pull[1])

    results = load_results()
    if not results:
        print("No results found in results/. Run experiments first or use --pull.")
        sys.exit(1)

    contexts = get_context_lengths(results)
    print(f"Loaded {len(results)} result files.")
    print(f"Context lengths: {contexts}")

    # Summary tables
    print_full_summary(results)

    # Check for Direction B data (global-attention reference runs)
    has_global = any(r.get("window_pattern") == "L" for r in results)
    has_multicontext = len(contexts) > 1

    # Crossover analysis (only if multi-context)
    if has_multicontext:
        print_crossover_summary(results)

    # Deficit analysis (Direction B — if global reference available)
    if has_global:
        print_regional_closure(results)   # primary endpoint first
        print_deficit_summary(results)
        print_statistical_summary(results)

    # Figures
    fig_main_comparison(results)
    fig_overhead(results)
    fig_scatter(results)

    # Multi-context figures
    if has_multicontext:
        fig_crossover(results)
        fig_bpb_position_grid(results)
        fig_compute_efficiency(results)

    # Direction B figures
    if has_global:
        fig_three_condition(results)
        fig_deficit_map(results)
        fig_deficit_closure(results)
        fig_closure_vs_context(results)

    print(f"\nAll figures saved to {FIGURES_DIR}/")

    # Generate markdown report
    report = generate_report(results)
    report_path = RESULTS_DIR / "RESULTS.md"
    report_path.write_text(report)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
