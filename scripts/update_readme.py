#!/usr/bin/env python3
"""Update README.md with latest benchmark results.

Reads result JSONs from results/, computes summary statistics,
and replaces the placeholder table and findings in README.md.

Usage:
    python update_readme.py
"""
import json
import re
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")
README_PATH = Path("README.md")

MECH_ORDER = [
    ("baseline", "Baseline"),
    ("persistent", "Persistent Memory (32 tokens)"),
    ("rmt", "RMT (16 tokens, seg=512)"),
    ("ttt", "TTT-Linear (chunk=64)"),
    ("deltanet", "Gated DeltaNet"),
]


def load_results():
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name.startswith("benchmark_"):
            continue
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def group_by_mechanism(results):
    groups = {}
    for r in results:
        mech = r.get("mechanism", "baseline")
        if mech == "none":
            mech = "baseline"
        # Normalize mechanism names that include config details
        base_mech = mech.split("-")[0] if mech not in ("baseline",) else mech
        for key in ("baseline", "persistent", "rmt", "ttt", "deltanet"):
            if mech.startswith(key):
                base_mech = key
                break
        groups.setdefault(base_mech, []).append(r)
    return groups


def build_table(results):
    groups = group_by_mechanism(results)
    baseline_bpb = None

    rows = []
    for mech_key, mech_label in MECH_ORDER:
        if mech_key not in groups:
            rows.append(f"| {mech_label} | — | — | — | — | — |")
            continue

        runs = groups[mech_key]
        bpbs = [r["min_val_bpb"] for r in runs if r.get("min_val_bpb") is not None]
        times = [r["total_time_min"] for r in runs if r.get("total_time_min") is not None]
        mem_params = runs[0].get("memory_params", 0)

        if not bpbs:
            rows.append(f"| {mech_label} | — | — | — | — | — |")
            continue

        mean_bpb = np.mean(bpbs)
        std_bpb = np.std(bpbs) if len(bpbs) > 1 else 0
        mean_time = np.mean(times) if times else 0

        if mech_key == "baseline":
            baseline_bpb = mean_bpb
            delta = "—"
        else:
            delta = f"{mean_bpb - baseline_bpb:+.4f}" if baseline_bpb else "?"

        if mem_params == 0:
            param_str = "0"
        elif mem_params < 1000:
            param_str = f"~{mem_params:,}"
        elif mem_params < 1_000_000:
            param_str = f"~{mem_params/1000:.0f}K"
        else:
            param_str = f"~{mem_params/1_000_000:.1f}M"

        bpb_str = f"**{mean_bpb:.4f}**" if (baseline_bpb and mean_bpb < baseline_bpb) else f"{mean_bpb:.4f}"
        delta_str = f"**{delta}**" if delta.startswith("-") else delta

        rows.append(
            f"| {mech_label} | {bpb_str} | {std_bpb:.5f} | {delta_str} | {param_str} | {mean_time:.1f} min |"
        )

    header = "| Mechanism | BPB (mean) | BPB (std) | vs Baseline | Extra Params | Train Time |"
    sep = "|-----------|-----------|-----------|-------------|--------------|------------|"
    return "\n".join([header, sep] + rows)


def build_findings(results):
    groups = group_by_mechanism(results)
    baseline_bpb = None
    if "baseline" in groups:
        bpbs = [r["min_val_bpb"] for r in groups["baseline"] if r.get("min_val_bpb")]
        baseline_bpb = np.mean(bpbs) if bpbs else None

    if not baseline_bpb:
        return "Results pending — benchmark running. Check back soon."

    lines = []
    better = []
    worse = []
    for mech_key, mech_label in MECH_ORDER:
        if mech_key == "baseline" or mech_key not in groups:
            continue
        bpbs = [r["min_val_bpb"] for r in groups[mech_key] if r.get("min_val_bpb")]
        if not bpbs:
            continue
        mean = np.mean(bpbs)
        delta = mean - baseline_bpb
        if delta < -0.001:
            better.append((mech_label, delta))
        elif delta > 0.001:
            worse.append((mech_label, delta))

    if worse:
        lines.append("**Mechanisms that hurt at this scale:**")
        for name, d in sorted(worse, key=lambda x: x[1], reverse=True):
            lines.append(f"- {name}: {d:+.4f} BPB vs baseline")
        lines.append("")

    if better:
        lines.append("**Mechanisms that help:**")
        for name, d in sorted(better, key=lambda x: x[1]):
            lines.append(f"- {name}: {d:+.4f} BPB vs baseline")
        lines.append("")

    if not better and not worse:
        lines.append("All mechanisms within noise of baseline at this scale.")

    n_complete = sum(1 for k, _ in MECH_ORDER if k in groups)
    n_total = len(MECH_ORDER)
    if n_complete < n_total:
        lines.append(f"*{n_complete}/{n_total} mechanisms complete. Results will update as experiments finish.*")

    return "\n".join(lines)


def update_readme(table: str, findings: str):
    readme = README_PATH.read_text()

    # Replace table
    readme = re.sub(
        r"<!-- RESULTS_TABLE_START -->.*?<!-- RESULTS_TABLE_END -->",
        f"<!-- RESULTS_TABLE_START -->\n{table}\n<!-- RESULTS_TABLE_END -->",
        readme, flags=re.DOTALL
    )

    # Replace findings
    readme = re.sub(
        r"<!-- FINDINGS_START -->.*?<!-- FINDINGS_END -->",
        f"<!-- FINDINGS_START -->\n{findings}\n<!-- FINDINGS_END -->",
        readme, flags=re.DOTALL
    )

    # Uncomment figure if it exists
    fig_path = RESULTS_DIR / "figures" / "fig1_bpb_comparison.png"
    if fig_path.exists():
        readme = re.sub(
            r"<!-- RESULTS_FIGURE -->.*?<!-- RESULTS_FIGURE_END -->",
            f"<!-- RESULTS_FIGURE -->\n![BPB Comparison](results/figures/fig1_bpb_comparison.png)\n<!-- RESULTS_FIGURE_END -->",
            readme, flags=re.DOTALL
        )

    README_PATH.write_text(readme)
    print(f"Updated {README_PATH}")


def main():
    results = load_results()
    if not results:
        print("No results found. Run experiments first.")
        return

    print(f"Found {len(results)} result files.")
    table = build_table(results)
    findings = build_findings(results)

    print("\nTable:")
    print(table)
    print("\nFindings:")
    print(findings)

    update_readme(table, findings)


if __name__ == "__main__":
    main()
