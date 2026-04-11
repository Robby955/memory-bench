"""
Plotting utilities for memory-bench results.

Generates comparison figures:
1. BPB bar chart across mechanisms
2. NIAH heatmap (accuracy vs context length and position)
3. Efficiency Pareto (BPB vs training time)
4. Parameter breakdown
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict


# Consistent colors for each mechanism
MECHANISM_COLORS = {
    "baseline": "#4A90D9",
    "none": "#4A90D9",
    "persistent": "#E8A838",
    "rmt": "#50C878",
    "ttt": "#E85050",
    "deltanet": "#9B59B6",
}

MECHANISM_LABELS = {
    "baseline": "Baseline",
    "none": "Baseline",
    "persistent": "Persistent Memory",
    "rmt": "RMT",
    "ttt": "TTT-Linear",
    "deltanet": "Gated DeltaNet",
}


def _normalize_mechanism(name: str) -> str:
    """Normalize mechanism name to base type (strip config suffixes)."""
    if name in ("none", "baseline"):
        return "baseline"
    # Strip config suffixes: persistent-32 -> persistent, rmt-m16-s512 -> rmt
    for base in ("persistent", "rmt", "ttt", "deltanet"):
        if name == base or name.startswith(base + "-"):
            return base
    return name


def _group_by_mechanism(results: list[dict]) -> dict:
    """Group results by mechanism name."""
    groups = defaultdict(list)
    for r in results:
        mech = _normalize_mechanism(r.get("mechanism", "baseline"))
        groups[mech].append(r)
    return dict(groups)


def plot_bpb_comparison(results: list[dict], output_path: str, depth: int = None):
    """Bar chart comparing BPB across mechanisms.

    Shows mean BPB with error bars (std across seeds).
    """
    groups = _group_by_mechanism(results)
    if depth:
        groups = {k: [r for r in v if r.get("depth") == depth] for k, v in groups.items()}
        groups = {k: v for k, v in groups.items() if v}

    mechanisms = []
    means = []
    stds = []
    colors = []

    for mech in ["baseline", "persistent", "rmt", "ttt", "deltanet"]:
        if mech not in groups:
            continue
        bpbs = [r["min_val_bpb"] for r in groups[mech] if r.get("min_val_bpb") is not None]
        if not bpbs:
            continue
        mechanisms.append(MECHANISM_LABELS.get(mech, mech))
        means.append(np.mean(bpbs))
        stds.append(np.std(bpbs) if len(bpbs) > 1 else 0)
        colors.append(MECHANISM_COLORS.get(mech, "#888888"))

    if not mechanisms:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(mechanisms))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Validation BPB (bits per byte)", fontsize=12)
    ax.set_title(f"Memory Mechanism Comparison - d{depth or 'all'}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(mechanisms, fontsize=11)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=9)

    # Make y-axis start near the minimum for better visual comparison
    if means:
        ymin = min(means) - max(max(stds, default=0) * 3, 0.01)
        ymax = max(means) + max(max(stds, default=0) * 3, 0.01)
        ax.set_ylim(max(0, ymin), ymax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_niah_heatmap(niah_results: dict, mechanism_name: str, output_path: str):
    """Heatmap of NIAH accuracy vs context length and passkey position."""
    accuracy = niah_results.get("accuracy", {})
    if not accuracy:
        return

    ctx_lengths = sorted([int(k) for k in accuracy.keys()])
    positions = sorted([float(p) for p in next(iter(accuracy.values())).keys()])

    data = np.zeros((len(ctx_lengths), len(positions)))
    for i, ctx in enumerate(ctx_lengths):
        for j, pos in enumerate(positions):
            data[i, j] = accuracy.get(str(ctx), {}).get(str(pos), 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"{p:.0%}" for p in positions])
    ax.set_yticks(range(len(ctx_lengths)))
    ax.set_yticklabels(ctx_lengths)

    ax.set_xlabel("Passkey Position (% through context)", fontsize=11)
    ax.set_ylabel("Context Length (tokens)", fontsize=11)
    ax.set_title(f"NIAH Retrieval Accuracy - {MECHANISM_LABELS.get(mechanism_name, mechanism_name)}", fontsize=13)

    # Add text annotations
    for i in range(len(ctx_lengths)):
        for j in range(len(positions)):
            color = "white" if data[i, j] < 0.5 else "black"
            ax.text(j, i, f"{data[i, j]:.0%}", ha="center", va="center", color=color, fontsize=10)

    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_efficiency_pareto(results: list[dict], output_path: str):
    """Scatter plot: BPB vs training time (efficiency Pareto frontier)."""
    groups = _group_by_mechanism(results)

    fig, ax = plt.subplots(figsize=(10, 6))

    for mech in ["baseline", "persistent", "rmt", "ttt", "deltanet"]:
        if mech not in groups:
            continue
        runs = groups[mech]
        bpbs = [r["min_val_bpb"] for r in runs if r.get("min_val_bpb") and r.get("total_time_min")]
        times = [r["total_time_min"] for r in runs if r.get("min_val_bpb") and r.get("total_time_min")]
        if not bpbs:
            continue
        color = MECHANISM_COLORS.get(mech, "#888888")
        label = MECHANISM_LABELS.get(mech, mech)
        ax.scatter(times, bpbs, c=color, label=label, s=80, edgecolors="black", linewidth=0.5, zorder=3)

    ax.set_xlabel("Training Time (minutes)", fontsize=12)
    ax.set_ylabel("Validation BPB", fontsize=12)
    ax.set_title("Efficiency-Quality Pareto", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_parameter_breakdown(results: list[dict], output_path: str):
    """Stacked bar chart showing base params vs memory params."""
    groups = _group_by_mechanism(results)

    mechanisms = []
    base_params = []
    mem_params = []
    colors_base = []
    colors_mem = []

    for mech in ["baseline", "persistent", "rmt", "ttt", "deltanet"]:
        if mech not in groups:
            continue
        r = groups[mech][0]  # take first run for param counts
        total = r.get("total_params", 0)
        memory = r.get("memory_params", 0)
        mechanisms.append(MECHANISM_LABELS.get(mech, mech))
        base_params.append((total - memory) / 1e6)
        mem_params.append(memory / 1e6)
        colors_base.append(MECHANISM_COLORS.get(mech, "#888888"))
        colors_mem.append("#FFD700")  # gold for memory params

    if not mechanisms:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(mechanisms))

    ax.bar(x, base_params, color=colors_base, edgecolor="black", linewidth=0.5, label="Base Parameters")
    ax.bar(x, mem_params, bottom=base_params, color="#FFD700", edgecolor="black", linewidth=0.5, label="Memory Parameters")

    ax.set_ylabel("Parameters (millions)", fontsize=12)
    ax.set_title("Parameter Breakdown", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(mechanisms, fontsize=11)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_bpb_by_position(results: list[dict], output_path: str, depth: int = None):
    """THE centerpiece figure: BPB vs context position for all mechanisms.

    Shows mean BPB curve with shaded std bands (across seeds) for each mechanism.
    Memory mechanisms should show steeper improvement at later positions.
    """
    groups = _group_by_mechanism(results)
    if depth:
        groups = {k: [r for r in v if r.get("depth") == depth] for k, v in groups.items()}
        groups = {k: v for k, v in groups.items() if v}

    fig, ax = plt.subplots(figsize=(12, 6))

    for mech in ["baseline", "persistent", "rmt", "ttt", "deltanet"]:
        if mech not in groups:
            continue

        # Collect BPB-by-position across seeds
        all_curves = []
        positions = None
        for r in groups[mech]:
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
        color = MECHANISM_COLORS.get(mech, "#888888")
        label = MECHANISM_LABELS.get(mech, mech)

        ax.plot(positions, mean_curve, color=color, label=f"{label} (n={len(all_curves)})",
                linewidth=2, marker="o", markersize=3)

        if len(all_curves) > 1:
            std_curve = np.std(curves, axis=0)
            ax.fill_between(positions, mean_curve - std_curve, mean_curve + std_curve,
                            color=color, alpha=0.15)

    ax.set_xlabel("Position in context (tokens)", fontsize=12)
    ax.set_ylabel("BPB (bits per byte)", fontsize=12)
    ax.set_title(f"BPB by Context Position - d{depth or 'all'}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_bpb_position_delta(results: list[dict], output_path: str, depth: int = None):
    """Delta plot: (mechanism BPB - baseline BPB) at each position bucket.

    Negative values = mechanism is better. Shows WHERE each mechanism helps.
    """
    groups = _group_by_mechanism(results)
    if depth:
        groups = {k: [r for r in v if r.get("depth") == depth] for k, v in groups.items()}
        groups = {k: v for k, v in groups.items() if v}

    # Get baseline mean curve
    if "baseline" not in groups:
        return
    baseline_curves = []
    positions = None
    for r in groups["baseline"]:
        buckets = r.get("bpb_by_position", {}).get("buckets", {})
        if not buckets:
            continue
        if positions is None:
            positions = [v["center_position"] for v in buckets.values()]
        baseline_curves.append([v["bpb"] for v in buckets.values()])

    if not baseline_curves or positions is None:
        return
    baseline_mean = np.mean(baseline_curves, axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    for mech in ["persistent", "rmt", "ttt", "deltanet"]:
        if mech not in groups:
            continue

        mech_curves = []
        for r in groups[mech]:
            buckets = r.get("bpb_by_position", {}).get("buckets", {})
            if not buckets:
                continue
            mech_curves.append([v["bpb"] for v in buckets.values()])

        if not mech_curves:
            continue

        mech_mean = np.mean(mech_curves, axis=0)
        delta = mech_mean - baseline_mean

        color = MECHANISM_COLORS.get(mech, "#888888")
        label = MECHANISM_LABELS.get(mech, mech)
        ax.plot(positions, delta, color=color, label=label,
                linewidth=2, marker="o", markersize=3)

        if len(mech_curves) > 1:
            mech_std = np.std(mech_curves, axis=0)
            ax.fill_between(positions, delta - mech_std, delta + mech_std,
                            color=color, alpha=0.15)

    ax.set_xlabel("Position in context (tokens)", fontsize=12)
    ax.set_ylabel("ΔBPB vs baseline (negative = better)", fontsize=12)
    ax.set_title(f"Memory Mechanism Improvement by Position - d{depth or 'all'}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_plots(results: list[dict], output_dir: str = "results/figures"):
    """Generate all standard comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Get unique depths
    depths = sorted(set(r.get("depth", 12) for r in results))

    for depth in depths:
        depth_results = [r for r in results if r.get("depth") == depth]
        plot_bpb_comparison(depth_results, os.path.join(output_dir, f"bpb_d{depth}.png"), depth=depth)
        plot_bpb_by_position(depth_results, os.path.join(output_dir, f"bpb_by_position_d{depth}.png"), depth=depth)
        plot_bpb_position_delta(depth_results, os.path.join(output_dir, f"bpb_position_delta_d{depth}.png"), depth=depth)

    plot_efficiency_pareto(results, os.path.join(output_dir, "pareto.png"))
    plot_parameter_breakdown(results, os.path.join(output_dir, "params.png"))

    # NIAH plots (if available)
    for r in results:
        if "niah" in r and r["niah"]:
            mech = r.get("mechanism", "baseline")
            seed = r.get("seed", 0)
            plot_niah_heatmap(
                r["niah"], mech,
                os.path.join(output_dir, f"niah_{mech}_s{seed}.png"),
            )
