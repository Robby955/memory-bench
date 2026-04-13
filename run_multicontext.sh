#!/bin/bash
# ==========================================================================
# memory-bench Direction B: Positional Context Deficit Experiment
#
# Tests whether persistent memory recovers performance specifically where
# restricted-context attention is most impaired, as measured by the gap
# between local-attention (SSSL) and global-attention (L) baselines.
#
# Phase 0: Environment recording + 2048 repro validation
# Phase 1: 2048 global-attention reference (3 seeds) — establish deficit at 2048
# Phase 2: 4096 local + global + persistent memory (9 runs)
# Phase 3: 8192 local + global + persistent memory (9 runs)
# Phase 4: Full deficit/gain/closure analysis + figures
#
# Each phase saves results to git and pauses for review.
#
# Usage:
#   bash run_multicontext.sh                    # Full phased run
#   bash run_multicontext.sh --skip-phase0      # Skip env validation
#   bash run_multicontext.sh --start-phase 2    # Resume from phase 2
# ==========================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
DEPTH=12
SEEDS="42,1337,3141"
NPROC=8
NIAH="--niah-at-end"
SKIP_PHASE0=false
START_PHASE=0

# Known baseline seed 42 BPB from original 2048 runs
KNOWN_BASELINE_S42_BPB="0.846660"
REPRO_TOLERANCE="0.005"

# Parse CLI
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-phase0) SKIP_PHASE0=true; shift ;;
        --start-phase) START_PHASE="$2"; shift 2 ;;
        --no-niah) NIAH=""; shift ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --nproc) NPROC="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Per-context batch sizes (total_batch_size=524288, 8 GPUs, grad_accum=1)
get_device_batch_size() {
    case $1 in
        2048) echo 32 ;;
        4096) echo 16 ;;
        8192) echo 8 ;;
        *) echo "ERROR: Unknown context $1" >&2; exit 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Environment
export NANOCHAT_BASE_DIR="/workspace/memory-bench-cache"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4
export WANDB_MODE=disabled

cd /workspace/memory-bench

IFS=',' read -ra SEED_ARR <<< "$SEEDS"

mkdir -p results results/figures results/logs results/environment

# ---------------------------------------------------------------------------
# Helper: run a single experiment
# Args: mechanism context_length seed [window_pattern]
run_one() {
    local mech="$1" ctx="$2" seed="$3"
    local wp="${4:-SSSL}"
    local dbs
    dbs=$(get_device_batch_size "$ctx")

    local wp_tag=""
    [[ "$wp" != "SSSL" ]] && wp_tag="_w${wp}"
    local run_name="${mech}_d${DEPTH}_t${ctx}${wp_tag}_s${seed}"
    local log_file="results/logs/${run_name}.log"

    # Result file check
    local result_base="${mech}"
    [[ "$mech" == "none" ]] && result_base="baseline"
    local result_file="results/${result_base}_d${DEPTH}_t${ctx}${wp_tag}_s${seed}.json"
    # Backward compat for 2048/SSSL
    local result_file_old="results/${result_base}_d${DEPTH}_s${seed}.json"

    if [[ -f "$result_file" ]]; then
        echo "    SKIP (exists: $result_file)"
        return 0
    fi
    if [[ -f "$result_file_old" && "$ctx" == "2048" && "$wp" == "SSSL" ]]; then
        echo "    SKIP (old format exists: $result_file_old)"
        return 0
    fi

    local cmd="python -m torch.distributed.run \
        --standalone --nproc_per_node=$NPROC \
        -m memory_bench.train \
        --depth=$DEPTH \
        --mechanism=$mech \
        --seed=$seed \
        --max-seq-len=$ctx \
        --device-batch-size=$dbs \
        --total-batch-size=524288 \
        --window-pattern=$wp \
        --exp-tag=$run_name \
        $NIAH"

    # Mechanism-specific args
    case $mech in
        persistent) cmd="$cmd --num-memory-tokens=32" ;;
        rmt)        cmd="$cmd --num-memory-tokens=16 --segment-length=512 --bptt-depth=2" ;;
        ttt)        cmd="$cmd --ttt-layer=-1 --ttt-chunk-size=64 --no-compile" ;;
        deltanet)   cmd="$cmd --no-compile" ;;
    esac

    echo "    RUN: $run_name (dbs=$dbs, wp=$wp, log=$log_file)"
    local t0
    t0=$(date +%s)
    if eval $cmd 2>&1 | tee "$log_file"; then
        local t1
        t1=$(date +%s)
        echo "    DONE in $(( (t1 - t0) / 60 ))m"
        return 0
    else
        echo "    FAILED (see $log_file)"
        return 1
    fi
}

# Helper: extract min_val_bpb from a result JSON
get_bpb() {
    python3 -c "import json; d=json.load(open('$1')); print(f'{d[\"min_val_bpb\"]:.6f}')" 2>/dev/null || echo "MISSING"
}

# Helper: phase gate
phase_gate() {
    local phase_name="$1"
    local message="$2"
    echo ""
    echo "================================================================"
    echo "  PHASE GATE: $phase_name"
    echo "  $message"
    echo "================================================================"
    echo "  Pausing 10 seconds — Ctrl-C to abort."
    echo ""
    sleep 10
}

# Helper: kill zombies
cleanup_gpus() {
    pkill -9 -f "memory_bench.train" 2>/dev/null || true
    pkill -9 -f "torchrun" 2>/dev/null || true
    sleep 15
}

# Helper: save results to private dev repo
save_checkpoint() {
    local phase_msg="$1"
    echo ""
    echo "  Saving results checkpoint: $phase_msg"
    if [[ -f "save_results.sh" ]]; then
        bash save_results.sh "$phase_msg" || echo "  WARNING: save_results.sh failed (non-fatal)"
    else
        echo "  WARNING: save_results.sh not found, skipping checkpoint"
    fi
}

# Helper: print results for a set of runs
print_results() {
    local pattern="$1"
    local label="$2"
    echo ""
    echo "  $label:"
    for f in $pattern; do
        if [[ -f "$f" ]]; then
            local bpb
            bpb=$(get_bpb "$f")
            local fname
            fname=$(basename "$f")
            echo "    $fname: $bpb"
        fi
    done
}

TOTAL_START=$(date +%s)

# =====================================================================
# PHASE 0: Environment recording + 2048 baseline reproduction
# =====================================================================
if [[ $START_PHASE -le 0 ]]; then
    echo ""
    echo "================================================================"
    echo "  PHASE 0: Environment Recording & Validation"
    echo "================================================================"
    echo ""

    echo "Recording environment..."
    date -u > results/environment/run_timestamp.txt
    git rev-parse HEAD > results/environment/git_hash.txt 2>/dev/null || echo "not a git repo" > results/environment/git_hash.txt
    pip freeze > results/environment/pip_freeze.txt
    nvidia-smi > results/environment/nvidia_smi.txt 2>/dev/null || echo "no GPU" > results/environment/nvidia_smi.txt
    python3 -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')" > results/environment/torch_version.txt
    echo "  Saved to results/environment/"
    cat results/environment/torch_version.txt

    if [[ "$SKIP_PHASE0" == "true" ]]; then
        echo "  --skip-phase0: Skipping baseline reproduction check."
    else
        EXISTING_REPRO=""
        if [[ -f "results/baseline_d${DEPTH}_t2048_s42.json" ]]; then
            EXISTING_REPRO="results/baseline_d${DEPTH}_t2048_s42.json"
        elif [[ -f "results/baseline_d${DEPTH}_s42.json" ]]; then
            EXISTING_REPRO="results/baseline_d${DEPTH}_s42.json"
        fi

        if [[ -n "$EXISTING_REPRO" ]]; then
            echo ""
            echo "  Existing 2048 baseline seed 42 found: $EXISTING_REPRO"
            echo "  Validating against known BPB (skipping re-run)..."
            REPRO_FILE="$EXISTING_REPRO"
        else
            echo ""
            echo "Reproducing 2048 baseline (seed 42) to validate environment..."
            repro_cmd="python -m torch.distributed.run \
                --standalone --nproc_per_node=$NPROC \
                -m memory_bench.train \
                --depth=$DEPTH \
                --mechanism=none \
                --seed=42 \
                --max-seq-len=2048 \
                --device-batch-size=32 \
                --total-batch-size=524288 \
                --exp-tag=repro_check_2048_s42"

            if eval $repro_cmd 2>&1 | tee results/logs/repro_check_2048_s42.log; then
                REPRO_FILE="results/baseline_d${DEPTH}_t2048_s42.json"
            else
                echo "  WARNING: Reproduction run failed. Proceeding with caution."
                REPRO_FILE=""
            fi
            cleanup_gpus
        fi

        if [[ -n "$REPRO_FILE" ]] && [[ -f "$REPRO_FILE" ]]; then
            REPRO_BPB=$(get_bpb "$REPRO_FILE")
            echo ""
            echo "  Reproduction BPB:  $REPRO_BPB"
            echo "  Known BPB:         $KNOWN_BASELINE_S42_BPB"

            if ! python3 -c "exit(0 if abs(float('$REPRO_BPB') - float('$KNOWN_BASELINE_S42_BPB')) <= float('$REPRO_TOLERANCE') else 1)"; then
                echo ""
                echo "  *** ENVIRONMENT VALIDATION FAILED ***"
                echo "  BPB deviates too much from stored result."
                echo "  Check: PyTorch version, CUDA version."
                echo "  Aborting."
                exit 1
            fi
            echo "  PASS: within tolerance $REPRO_TOLERANCE"
        else
            echo "  WARNING: No result file to validate. Proceeding with caution."
        fi
    fi

    save_checkpoint "Phase 0: environment + 2048 repro validation"
    phase_gate "Phase 0 complete" "Environment recorded. Baseline reproduction checked."
fi

# =====================================================================
# PHASE 1: 2048 global-attention reference (3 seeds)
# Establishes the positional context deficit at 2048.
# =====================================================================
if [[ $START_PHASE -le 1 ]]; then
    echo ""
    echo "================================================================"
    echo "  PHASE 1: 2048 Global-Attention Reference (3 seeds, ~15 min)"
    echo "  Purpose: Establish baseline deficit at 2048 context"
    echo "================================================================"
    echo ""

    PHASE1_FAILED=0
    for seed in "${SEED_ARR[@]}"; do
        echo "  [global / 2048 / seed $seed]"
        run_one none 2048 "$seed" L || PHASE1_FAILED=$((PHASE1_FAILED + 1))
        cleanup_gpus
    done

    # Report and compare
    echo ""
    echo "  2048 Results — Local (SSSL) vs Global (L):"
    python3 -c "
import json, glob
local_bpbs, global_bpbs = [], []
for f in glob.glob('results/baseline_d${DEPTH}_t2048_s*.json') + glob.glob('results/baseline_d${DEPTH}_s*.json'):
    d = json.load(open(f))
    if d.get('window_pattern', 'SSSL') == 'SSSL' and d.get('min_val_bpb'):
        local_bpbs.append(d['min_val_bpb'])
for f in glob.glob('results/baseline_d${DEPTH}_t2048_wL_s*.json'):
    d = json.load(open(f))
    if d.get('min_val_bpb'):
        global_bpbs.append(d['min_val_bpb'])

if local_bpbs and global_bpbs:
    ml = sum(local_bpbs)/len(local_bpbs)
    mg = sum(global_bpbs)/len(global_bpbs)
    print(f'  Local (SSSL) mean: {ml:.6f} ({len(local_bpbs)} seeds)')
    print(f'  Global (L) mean:   {mg:.6f} ({len(global_bpbs)} seeds)')
    print(f'  Aggregate deficit: {ml - mg:+.6f} BPB')
    if ml - mg < -0.001:
        print('  NOTE: Global is worse than local at 2048 — deficit may be negative.')
        print('  This is OK: at short context, local attention is already sufficient.')
else:
    print('  Missing data — need both local and global results.')
" 2>&1 || true

    save_checkpoint "Phase 1: 2048 global reference x 3 seeds"
    phase_gate "Phase 1 complete" "2048 deficit established. Review numbers above."
fi

# =====================================================================
# PHASE 2: 4096 — local baseline + global reference + persistent memory
# Kill gate: check that deficit grows vs 2048
# =====================================================================
if [[ $START_PHASE -le 2 ]]; then
    echo ""
    echo "================================================================"
    echo "  PHASE 2: 4096 Full Triad (9 runs, ~2.5h)"
    echo "  local baseline + global reference + persistent memory"
    echo "================================================================"
    echo ""

    PHASE2_FAILED=0

    echo "  --- 4096 Local Baseline (SSSL) ---"
    for seed in "${SEED_ARR[@]}"; do
        run_one none 4096 "$seed" SSSL || PHASE2_FAILED=$((PHASE2_FAILED + 1))
        cleanup_gpus
    done

    echo ""
    echo "  --- 4096 Global Reference (L) ---"
    for seed in "${SEED_ARR[@]}"; do
        run_one none 4096 "$seed" L || PHASE2_FAILED=$((PHASE2_FAILED + 1))
        cleanup_gpus
    done

    echo ""
    echo "  --- 4096 Persistent Memory (SSSL) ---"
    for seed in "${SEED_ARR[@]}"; do
        run_one persistent 4096 "$seed" SSSL || PHASE2_FAILED=$((PHASE2_FAILED + 1))
        cleanup_gpus
    done

    echo ""
    echo "  Phase 2 failures: $PHASE2_FAILED"

    # Deficit analysis at 4096 — this is the KILL GATE
    # If late-region deficit is not materially above zero, 8192 is lower priority.
    echo ""
    echo "  ================================================================"
    echo "  KILL GATE CHECK: Does deficit exist at 4096?"
    echo "  ================================================================"
    python3 -c "
import sys
sys.path.insert(0, '.')
from analyze_results import load_results, print_deficit_summary, print_regional_closure, get_context_lengths
results = load_results()
contexts = get_context_lengths(results)
has_global = any(r.get('window_pattern') == 'L' for r in results)
print(f'  Loaded {len(results)} results across contexts: {contexts}')
if has_global:
    print_regional_closure(results)
    print_deficit_summary(results)
else:
    print('  No global-attention results found — deficit analysis unavailable.')
print()
print('  *** REVIEW: Is late-region deficit > 0 at 4096? ***')
print('  If deficit is negligible, consider stopping (Ctrl-C in next 10s).')
" 2>&1 || echo "  (analysis failed, continuing anyway)"

    save_checkpoint "Phase 2: 4096 local + global + persistent x 3 seeds"
    phase_gate "Phase 2 complete" "4096 deficit + closure computed. Check if deficit grew vs 2048 before 8192."
fi

# =====================================================================
# PHASE 3: 8192 — local baseline + global reference + persistent memory
# =====================================================================
if [[ $START_PHASE -le 3 ]]; then
    echo ""
    echo "================================================================"
    echo "  PHASE 3: 8192 Full Triad (9 runs, ~4h)"
    echo "  local baseline + global reference + persistent memory"
    echo "================================================================"
    echo ""

    PHASE3_FAILED=0

    echo "  --- 8192 Local Baseline (SSSL) ---"
    for seed in "${SEED_ARR[@]}"; do
        run_one none 8192 "$seed" SSSL || PHASE3_FAILED=$((PHASE3_FAILED + 1))
        cleanup_gpus
    done

    echo ""
    echo "  --- 8192 Global Reference (L) ---"
    for seed in "${SEED_ARR[@]}"; do
        run_one none 8192 "$seed" L || PHASE3_FAILED=$((PHASE3_FAILED + 1))
        cleanup_gpus
    done

    echo ""
    echo "  --- 8192 Persistent Memory (SSSL) ---"
    for seed in "${SEED_ARR[@]}"; do
        run_one persistent 8192 "$seed" SSSL || PHASE3_FAILED=$((PHASE3_FAILED + 1))
        cleanup_gpus
    done

    echo ""
    echo "  Phase 3 failures: $PHASE3_FAILED"

    # Check for OOM
    if [[ $PHASE3_FAILED -gt 0 ]]; then
        echo ""
        echo "  *** $PHASE3_FAILED of 9 runs FAILED ***"
        echo "  Check logs for OOM. Global attention at 8192 uses more VRAM than SSSL."
        echo "  Continuing to analysis with available data."
    fi

    # Report VRAM for 8192 runs
    echo ""
    echo "  8192 VRAM Report:"
    for f in results/baseline_d${DEPTH}_t8192_s*.json results/baseline_d${DEPTH}_t8192_wL_s*.json results/persistent_d${DEPTH}_t8192_s*.json; do
        if [[ -f "$f" ]]; then
            python3 -c "
import json
d = json.load(open('$f'))
fname = '$f'.split('/')[-1]
vram = d.get('peak_vram_mib', 0)
bpb = d.get('min_val_bpb', 0)
time = d.get('total_time_min', 0)
print(f'    {fname}: BPB={bpb:.6f} VRAM={vram:.0f}MiB time={time:.1f}m')
" 2>/dev/null || true
        fi
    done

    save_checkpoint "Phase 3: 8192 local + global + persistent x 3 seeds"
    phase_gate "Phase 3 complete" "All training runs done. Proceeding to analysis."
fi

# =====================================================================
# PHASE 4: Full Analysis
# =====================================================================
echo ""
echo "================================================================"
echo "  PHASE 4: FULL ANALYSIS"
echo "================================================================"
echo ""

TOTAL_END=$(date +%s)
TOTAL_HOURS=$(( (TOTAL_END - TOTAL_START) / 3600 ))
TOTAL_MINS=$(( (TOTAL_END - TOTAL_START) / 60 ))
echo "Total wall time: ${TOTAL_HOURS}h ${TOTAL_MINS}m"

N_RESULTS=$(find results -maxdepth 1 -name '*.json' ! -name 'benchmark_*' ! -name 'statistical_*' 2>/dev/null | wc -l)
echo "Total result files: $N_RESULTS"

echo ""
echo "Running full analysis..."
python3 -c "
import sys
sys.path.insert(0, '.')
from analyze_results import (
    load_results, get_context_lengths, print_full_summary, print_deficit_summary,
    print_regional_closure, print_statistical_summary,
    fig_main_comparison, fig_overhead, fig_scatter,
    fig_deficit_map, fig_deficit_closure, fig_closure_vs_context,
    fig_crossover, fig_bpb_position_grid, fig_compute_efficiency,
)

results = load_results()
contexts = get_context_lengths(results)
has_global = any(r.get('window_pattern') == 'L' for r in results)
print(f'Loaded {len(results)} results across contexts: {contexts}')
print(f'Has global-attention reference: {has_global}')

# Summary
print_full_summary(results)

# Core Direction B analysis (primary endpoint first)
if has_global:
    print_regional_closure(results)
    print_deficit_summary(results)
    print_statistical_summary(results)

# Standard figures
fig_main_comparison(results)
fig_overhead(results)
fig_scatter(results)

# Multi-context figures
if len(contexts) > 1:
    fig_crossover(results)
    fig_bpb_position_grid(results)
    fig_compute_efficiency(results)

# Direction B figures
if has_global:
    fig_deficit_map(results)
    fig_deficit_closure(results)
    fig_closure_vs_context(results)

print('All figures saved to results/figures/')
" 2>&1

save_checkpoint "FINAL: all experiments complete + analysis + figures"

echo ""
echo "================================================================"
echo "  COMPLETE"
echo "================================================================"
echo "  Results:  results/*.json"
echo "  Figures:  results/figures/"
echo "  Logs:     results/logs/"
echo "  Env:      results/environment/"
echo "  Git:      pushed to origin (memory-bench-dev)"
echo ""
echo "  Key figures:"
echo "    fig_deficit_map.png         — positional context deficit"
echo "    fig_deficit_closure_*.png   — gain/closure by position"
echo "    fig_closure_vs_context.png  — closure trend across contexts"
echo "================================================================"
