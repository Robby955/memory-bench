"""Benchmark runner: all mechanisms x seeds x depths."""

import os
import json
import subprocess
import sys
import argparse
import csv
from datetime import datetime
from pathlib import Path

from memory_bench.mechanisms import MECHANISMS


def run_experiment(
    depth: int,
    mechanism: str,
    seed: int,
    max_seq_len: int = 2048,
    window_pattern: str = "SSSL",
    device_batch_size: int = None,
    niah: bool = False,
    extra_args: list[str] = None,
    use_torchrun: bool = True,
    nproc: int = 8,
) -> dict:
    wp_tag = f"_w{window_pattern}" if window_pattern != "SSSL" else ""
    run_name = f"{mechanism}_d{depth}_t{max_seq_len}{wp_tag}_s{seed}"

    cmd = []
    if use_torchrun:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", f"--nproc_per_node={nproc}",
            "-m", "memory_bench.train",
        ]
    else:
        cmd = [sys.executable, "-m", "memory_bench.train"]

    cmd.extend([
        f"--depth={depth}",
        f"--mechanism={mechanism}",
        f"--seed={seed}",
        f"--max-seq-len={max_seq_len}",
        f"--window-pattern={window_pattern}",
        f"--exp-tag={run_name}",
    ])

    if device_batch_size is not None:
        cmd.append(f"--device-batch-size={device_batch_size}")

    if niah:
        cmd.append("--niah-at-end")

    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)

    # Read result file (check new format, fallback to old format for 2048/SSSL)
    mech_label = mechanism if mechanism != 'none' else 'baseline'
    result_file = f"results/{mech_label}_d{depth}_t{max_seq_len}{wp_tag}_s{seed}.json"
    result_file_old = f"results/{mech_label}_d{depth}_s{seed}.json"
    if os.path.exists(result_file):
        with open(result_file) as f:
            return json.load(f)
    elif max_seq_len == 2048 and os.path.exists(result_file_old):
        with open(result_file_old) as f:
            return json.load(f)
    else:
        print(f"WARNING: No result file found at {result_file}")
        return {
            "mechanism": mechanism,
            "depth": depth,
            "max_seq_len": max_seq_len,
            "seed": seed,
            "val_bpb": None,
            "error": f"exit code {result.returncode}",
        }


def collect_results(results_dir: str = "results") -> list[dict]:
    results = []
    for f in Path(results_dir).glob("*.json"):
        if f.name.startswith("benchmark_") or f.name.startswith("statistical_"):
            continue  # skip aggregate/stats files
        with open(f) as fp:
            data = json.load(fp)
        if data.get("min_val_bpb") is None:
            continue  # skip error/incomplete results
        results.append(data)
    return results


_CSV_FIELDS = [
    "mechanism", "depth", "max_seq_len", "seed", "val_bpb", "min_val_bpb",
    "total_params", "base_params", "memory_params", "param_overhead_pct",
    "total_time_min", "peak_vram_mib", "flops_per_step", "total_flops",
]


def write_csv(results: list[dict], output_path: str):
    if not results:
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def aggregate_results(results: list[dict]) -> dict:
    import numpy as np

    by_key = {}
    for r in results:
        mech = r["mechanism"]
        if mech == "none":
            mech = "baseline"
        for base in ("persistent", "rmt", "ttt", "deltanet"):
            if mech.startswith(base + "-"):
                mech = base
                break
        ctx = r.get("max_seq_len", 2048)
        key = (mech, r["depth"], ctx)
        if key not in by_key:
            by_key[key] = []
        by_key[key].append(r)

    aggregated = {}
    for (mech, depth, ctx), runs in by_key.items():
        bpbs = [r["min_val_bpb"] for r in runs if r.get("min_val_bpb") is not None]
        times = [r["total_time_min"] for r in runs if r.get("total_time_min") is not None]
        vrams = [r["peak_vram_mib"] for r in runs if r.get("peak_vram_mib") is not None]

        total_p = runs[0].get("total_params")
        mem_p = runs[0].get("memory_params", 0)
        overhead = runs[0].get("param_overhead_pct", 0.0)

        aggregated[f"{mech}_d{depth}_t{ctx}"] = {
            "mechanism": mech,
            "depth": depth,
            "max_seq_len": ctx,
            "n_seeds": len(bpbs),
            "mean_bpb": float(np.mean(bpbs)) if bpbs else None,
            "std_bpb": float(np.std(bpbs)) if len(bpbs) > 1 else 0,
            "mean_time_min": float(np.mean(times)) if times else None,
            "mean_vram_mib": float(np.mean(vrams)) if vrams else None,
            "total_params": total_p,
            "base_params": runs[0].get("base_params", total_p),
            "memory_params": mem_p,
            "param_overhead_pct": overhead,
        }

    return aggregated


def statistical_tests(results: list[dict], depth: int = 12, max_seq_len: int = None) -> dict:
    """Paired statistical tests: each mechanism vs baseline, paired by seed.

    If max_seq_len is None, tests are run per context length found in results.

    Returns a dict keyed by (mechanism, context_length) or mechanism with:
      - mean_delta: mean BPB difference (negative = mechanism is better)
      - paired_t_stat, paired_p_value: paired t-test
      - cohens_d: effect size (Cohen's d for paired samples)
      - seeds: list of (seed, baseline_bpb, mechanism_bpb, delta)
    """
    import numpy as np
    from scipy import stats

    # Determine context lengths to test
    if max_seq_len is not None:
        ctx_lengths = [max_seq_len]
    else:
        ctx_lengths = sorted(set(r.get("max_seq_len", 2048) for r in results))

    output = {}

    for ctx in ctx_lengths:
        # Group by (mechanism, seed) for this context length
        by_mech_seed = {}
        for r in results:
            if r.get("min_val_bpb") is None:
                continue
            mech = r["mechanism"]
            if mech == "none":
                mech = "baseline"
            for base in ("persistent", "rmt", "ttt", "deltanet"):
                if mech.startswith(base + "-"):
                    mech = base
                    break
            if r.get("depth", 12) != depth:
                continue
            if r.get("max_seq_len", 2048) != ctx:
                continue
            seed = r["seed"]
            by_mech_seed[(mech, seed)] = r["min_val_bpb"]

        # Get baseline BPBs by seed
        baseline_seeds = {seed: bpb for (mech, seed), bpb in by_mech_seed.items()
                          if mech == "baseline"}
        if not baseline_seeds:
            continue

        mechanisms = sorted(set(mech for mech, _ in by_mech_seed.keys()) - {"baseline"})

        for mech in mechanisms:
            paired = []
            for seed in sorted(baseline_seeds.keys()):
                if (mech, seed) in by_mech_seed:
                    b_bpb = baseline_seeds[seed]
                    m_bpb = by_mech_seed[(mech, seed)]
                    paired.append((seed, b_bpb, m_bpb, m_bpb - b_bpb))

            # Use (mech, ctx) key when multi-context, plain mech for single
            key = (mech, ctx) if len(ctx_lengths) > 1 else mech

            if len(paired) < 2:
                output[key] = {
                    "n_pairs": len(paired),
                    "max_seq_len": ctx,
                    "seeds": paired,
                    "mean_delta": paired[0][3] if paired else None,
                    "note": "insufficient pairs for t-test",
                }
                continue

            deltas = np.array([p[3] for p in paired])
            mean_delta = float(np.mean(deltas))
            std_delta = float(np.std(deltas, ddof=1))

            t_stat, p_value = stats.ttest_rel(
                [p[2] for p in paired],
                [p[1] for p in paired],
            )
            cohens_d = mean_delta / std_delta if std_delta > 0 else float("inf")

            output[key] = {
                "n_pairs": len(paired),
                "max_seq_len": ctx,
                "mean_delta": mean_delta,
                "std_delta": std_delta,
                "paired_t_stat": float(t_stat),
                "paired_p_value": float(p_value),
                "cohens_d": float(cohens_d),
                "seeds": paired,
            }

    return output


def print_statistical_summary(results: list[dict], depth: int = 12):
    """Print a formatted statistical comparison table."""
    stat_results = statistical_tests(results, depth=depth)
    if not stat_results:
        print("No statistical comparisons available (need baseline + mechanism results).")
        return

    # Check if multi-context (keys are tuples)
    sample_key = next(iter(stat_results))
    multi_ctx = isinstance(sample_key, tuple)

    print(f"\n{'='*90}")
    print("STATISTICAL COMPARISON vs BASELINE (paired by seed)")
    print(f"{'='*90}")
    if multi_ctx:
        print(f"{'Mechanism':<20} {'T':>6} {'N':>3} {'Mean Δ BPB':>12} {'t-stat':>8} {'p-value':>10} {'Cohen d':>9} {'Sig?':>6}")
    else:
        print(f"{'Mechanism':<20} {'N':>3} {'Mean Δ BPB':>12} {'t-stat':>8} {'p-value':>10} {'Cohen d':>9} {'Sig?':>6}")
    print("-" * 90)

    for key, s in sorted(stat_results.items()):
        n = s["n_pairs"]
        delta = s.get("mean_delta")
        if delta is None:
            continue

        if multi_ctx:
            mech, ctx = key
            ctx_str = str(ctx)
        else:
            mech = key
            ctx_str = None

        delta_str = f"{delta:+.6f}"

        if "paired_t_stat" in s:
            t_str = f"{s['paired_t_stat']:.3f}"
            p_str = f"{s['paired_p_value']:.4f}"
            d_str = f"{s['cohens_d']:.2f}"
            sig = "***" if s["paired_p_value"] < 0.01 else \
                  "**" if s["paired_p_value"] < 0.05 else \
                  "*" if s["paired_p_value"] < 0.10 else ""
        else:
            t_str = "N/A"
            p_str = "N/A"
            d_str = "N/A"
            sig = ""

        if multi_ctx:
            print(f"{mech:<20} {ctx_str:>6} {n:>3} {delta_str:>12} {t_str:>8} {p_str:>10} {d_str:>9} {sig:>6}")
        else:
            print(f"{mech:<20} {n:>3} {delta_str:>12} {t_str:>8} {p_str:>10} {d_str:>9} {sig:>6}")

        for seed, b_bpb, m_bpb, d in s["seeds"]:
            print(f"  seed {seed}: baseline={b_bpb:.6f}  {mech}={m_bpb:.6f}  Δ={d:+.6f}")

    print(f"\nSignificance: *** p<0.01, ** p<0.05, * p<0.10")
    print(f"Cohen's d: |d|>0.8 large, |d|>0.5 medium, |d|>0.2 small")


def _get_device_batch_size(max_seq_len: int) -> int:
    """Return device batch size that keeps total_batch_size ~ 524288."""
    return {2048: 32, 4096: 16, 8192: 8}.get(max_seq_len, 16)


def main():
    parser = argparse.ArgumentParser(description="memory-bench benchmark runner")
    parser.add_argument("--depths", type=str, default="12", help="comma-separated depths")
    parser.add_argument("--seeds", type=str, default="42,1337,3141", help="comma-separated seeds")
    parser.add_argument("--mechanisms", type=str, default="none,persistent,rmt,ttt,deltanet",
                        help="comma-separated mechanisms")
    parser.add_argument("--contexts", type=str, default="2048", help="comma-separated context lengths")
    parser.add_argument("--niah", action="store_true", help="run NIAH eval")
    parser.add_argument("--nproc", type=int, default=8, help="GPUs for torchrun")
    parser.add_argument("--no-torchrun", action="store_true", help="run single GPU")
    parser.add_argument("--extra-args", type=str, default="", help="extra args to pass through")
    args = parser.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    mechanisms = args.mechanisms.split(",")
    contexts = [int(c) for c in args.contexts.split(",")]
    extra = args.extra_args.split() if args.extra_args else []

    total = len(mechanisms) * len(depths) * len(seeds) * len(contexts)
    print(f"memory-bench: {len(mechanisms)} mechanisms x {len(depths)} depths x {len(contexts)} contexts x {len(seeds)} seeds")
    print(f"Mechanisms: {mechanisms}")
    print(f"Depths: {depths}")
    print(f"Contexts: {contexts}")
    print(f"Seeds: {seeds}")
    print(f"Total experiments: {total}")

    # Run all experiments
    all_results = []
    for depth in depths:
        for ctx in contexts:
            dbs = _get_device_batch_size(ctx)
            for mechanism in mechanisms:
                for seed in seeds:
                    result = run_experiment(
                        depth=depth,
                        mechanism=mechanism,
                        seed=seed,
                        max_seq_len=ctx,
                        device_batch_size=dbs,
                        niah=args.niah,
                        extra_args=extra,
                        use_torchrun=not args.no_torchrun,
                        nproc=args.nproc,
                    )
                    all_results.append(result)

    # Save aggregate results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"results/benchmark_{timestamp}.csv"
    write_csv(all_results, csv_path)
    print(f"\nResults written to {csv_path}")

    # Aggregate and print summary
    agg = aggregate_results(all_results)
    print(f"\n{'='*75}")
    print("SUMMARY")
    print(f"{'='*75}")
    print(f"{'Mechanism':<25} {'Depth':>5} {'Context':>7} {'Seeds':>5} {'BPB Mean':>10} {'BPB Std':>10} {'Time (min)':>10}")
    print("-" * 75)
    for key, data in sorted(agg.items()):
        bpb = f"{data['mean_bpb']:.5f}" if data['mean_bpb'] is not None else "N/A"
        std = f"{data['std_bpb']:.5f}" if data['std_bpb'] is not None else "N/A"
        t = f"{data['mean_time_min']:.1f}" if data['mean_time_min'] is not None else "N/A"
        ctx = data.get('max_seq_len', 2048)
        print(f"{data['mechanism']:<25} {data['depth']:>5} {ctx:>7} {data['n_seeds']:>5} {bpb:>10} {std:>10} {t:>10}")

    # Statistical tests
    print_statistical_summary(all_results)

    # Save statistical results to JSON
    stat_results = statistical_tests(all_results)
    if stat_results:
        stats_path = f"results/statistical_tests_{timestamp}.json"
        # Convert tuple keys and tuple values for JSON serialization
        stats_json = {}
        for key, s in stat_results.items():
            # Convert tuple key (mech, ctx) to string key
            json_key = f"{key[0]}_t{key[1]}" if isinstance(key, tuple) else key
            s_copy = dict(s)
            s_copy["seeds"] = [list(t) for t in s.get("seeds", [])]
            stats_json[json_key] = s_copy
        with open(stats_path, "w") as f:
            json.dump(stats_json, f, indent=2)
        print(f"Statistical tests saved to {stats_path}")

    # Generate plots
    print("\nGenerating plots...")
    from memory_bench.plot import generate_all_plots
    generate_all_plots(all_results, output_dir="results/figures")
    print("Done")


if __name__ == "__main__":
    main()
