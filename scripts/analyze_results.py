#!/usr/bin/env python3
"""Analyze memory-bench results and generate publication-ready figures.

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
COLORS = {
    "baseline": "#4A90D9",
    "persistent": "#E8A838",
    "rmt": "#50C878",
    "ttt": "#E85050",
    "deltanet": "#9B59B6",
}


def load_results() -> list[dict]:
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name.startswith("benchmark_"):
            continue
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def group_by_mechanism(results: list[dict]) -> dict[str, list[dict]]:
    groups = {}
    for r in results:
        mech = r.get("mechanism", "baseline")
        if mech == "none":
            mech = "baseline"
        # Normalize: "rmt-m16-s512" -> "rmt", "persistent-32" -> "persistent", etc.
        for key in MECHANISMS_ORDER:
            if mech.startswith(key):
                mech = key
                break
        groups.setdefault(mech, []).append(r)
    return groups


def print_summary(results: list[dict]):
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
        std_bpb = np.std(bpbs) if len(bpbs) > 1 else 0
        mean_time = np.mean(times) if times else 0

        if mech == "baseline":
            baseline_bpb = mean_bpb
            delta = "—"
        else:
            delta = f"{mean_bpb - baseline_bpb:+.4f}" if baseline_bpb else "?"

        print(f"{mech:<20} {len(bpbs):>5} {mean_bpb:>10.5f} {std_bpb:>8.5f} {delta:>8} {mean_time:>7.1f}m {mem_params:>12,}")

    print()


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

    print(f"Loaded {len(results)} result files.")
    print_summary(results)

    fig_main_comparison(results)
    fig_overhead(results)
    fig_scatter(results)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
