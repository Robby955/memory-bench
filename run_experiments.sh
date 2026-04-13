#!/bin/bash
# Run the full memory-bench experiment suite on 8xH100 pod.
#
# Usage:
#   bash run_experiments.sh              # Full suite (5 mechanisms x 3 seeds x 3 contexts)
#   bash run_experiments.sh --quick      # Quick smoke test (baseline + persistent, 1 seed, 2048 only)
#   bash run_experiments.sh --mechanism ttt --seeds 42  # Single mechanism
#   bash run_experiments.sh --context 4096,8192         # Specific context lengths only
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
DEPTH=12
SEEDS="42,1337,3141"
MECHANISMS="none,persistent,rmt,ttt,deltanet"
CONTEXTS="2048,4096,8192"
NPROC=8
NIAH="--niah-at-end"  # enabled by default (critical for benchmark)
EXTRA_ARGS=""

# Parse CLI overrides
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            SEEDS="42"
            MECHANISMS="none,persistent"
            CONTEXTS="2048"
            shift ;;
        --mechanism)
            MECHANISMS="$2"
            shift 2 ;;
        --seeds)
            SEEDS="$2"
            shift 2 ;;
        --depth)
            DEPTH="$2"
            shift 2 ;;
        --context)
            CONTEXTS="$2"
            shift 2 ;;
        --niah)
            NIAH="--niah-at-end"
            shift ;;
        --no-niah)
            NIAH=""
            shift ;;
        --nproc)
            NPROC="$2"
            shift 2 ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Per-context batch size (keeps total_batch_size = 524288 constant)
get_device_batch_size() {
    case $1 in
        2048) echo 32 ;;
        4096) echo 16 ;;
        8192) echo 8 ;;
        *) echo 16 ;;
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

# ---------------------------------------------------------------------------
# Preflight checks
echo "============================================================"
echo "  memory-bench experiment suite"
echo "============================================================"
echo "Depth:       $DEPTH"
echo "Seeds:       $SEEDS"
echo "Mechanisms:  $MECHANISMS"
echo "Contexts:    $CONTEXTS"
echo "GPUs:        $NPROC"
echo "NIAH:        ${NIAH:-disabled}"
echo "Cache dir:   $NANOCHAT_BASE_DIR"
echo "============================================================"
echo ""

# GPU check
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "$NUM_GPUS" -lt "$NPROC" ]]; then
    echo "ERROR: Found $NUM_GPUS GPUs but need $NPROC"
    exit 1
fi
echo "GPUs: $NUM_GPUS x $(nvidia-smi -L | head -1 | sed 's/GPU 0: //' | sed 's/ (.*//')"

# Kill any zombie processes from previous runs
echo "Killing any zombie training processes..."
pkill -9 -f "memory_bench.train" 2>/dev/null || true
pkill -9 -f "torchrun" 2>/dev/null || true
sleep 3

# Create results directory
mkdir -p results results/figures results/logs

# Ensure data will download (first shard test)
echo "Checking data availability..."
python -c "
import os, sys
os.environ['NANOCHAT_BASE_DIR'] = '$NANOCHAT_BASE_DIR'
sys.path.insert(0, os.path.join(os.getcwd(), 'nanochat'))
from nanochat.dataset import list_parquet_files
try:
    files = list_parquet_files()
    print(f'Data shards available: {len(files)}')
except Exception as e:
    print(f'Data will auto-download on first run: {e}')
"

# ---------------------------------------------------------------------------
# Run experiments
echo ""
echo "Starting experiments..."
echo ""

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
IFS=',' read -ra MECH_ARR <<< "$MECHANISMS"
IFS=',' read -ra CTX_ARR <<< "$CONTEXTS"

TOTAL=$(( ${#CTX_ARR[@]} * ${#MECH_ARR[@]} * ${#SEED_ARR[@]} ))
CURRENT=0
FAILED=0
START_TIME=$(date +%s)

for ctx in "${CTX_ARR[@]}"; do
    DBS=$(get_device_batch_size $ctx)
    echo "============================================================"
    echo "  Context length: $ctx (device_batch_size=$DBS)"
    echo "============================================================"

    for mech in "${MECH_ARR[@]}"; do
        for seed in "${SEED_ARR[@]}"; do
            CURRENT=$((CURRENT + 1))
            RUN_NAME="${mech}_d${DEPTH}_t${ctx}_s${seed}"
            LOG_FILE="results/logs/${RUN_NAME}.log"

            echo "============================================================"
            echo "  [$CURRENT/$TOTAL] $RUN_NAME"
            echo "  Log: $LOG_FILE"
            echo "============================================================"

            # Skip if result already exists (resume capability)
            RESULT_BASE="${mech}"
            [[ "$mech" == "none" ]] && RESULT_BASE="baseline"
            RESULT_FILE="results/${RESULT_BASE}_d${DEPTH}_t${ctx}_s${seed}.json"
            # Also check old filename format (without _t{ctx}) for backward compat
            RESULT_FILE_OLD="results/${RESULT_BASE}_d${DEPTH}_s${seed}.json"
            if [[ -f "$RESULT_FILE" ]]; then
                echo "  SKIPPED (result exists: $RESULT_FILE)"
                continue
            fi
            if [[ -f "$RESULT_FILE_OLD" && "$ctx" == "2048" ]]; then
                echo "  SKIPPED (old-format result exists: $RESULT_FILE_OLD)"
                continue
            fi

            # Build command
            CMD="python -m torch.distributed.run \
                --standalone --nproc_per_node=$NPROC \
                -m memory_bench.train \
                --depth=$DEPTH \
                --mechanism=$mech \
                --seed=$seed \
                --max-seq-len=$ctx \
                --device-batch-size=$DBS \
                --total-batch-size=524288 \
                --exp-tag=$RUN_NAME \
                $NIAH $EXTRA_ARGS"

            # Add mechanism-specific defaults
            case $mech in
                persistent)
                    CMD="$CMD --num-memory-tokens=32" ;;
                rmt)
                    CMD="$CMD --num-memory-tokens=16 --segment-length=512 --bptt-depth=2" ;;
                ttt)
                    CMD="$CMD --ttt-layer=-1 --ttt-chunk-size=64 --no-compile" ;;
                deltanet)
                    CMD="$CMD --no-compile" ;;
            esac

            # Run with logging
            RUN_START=$(date +%s)
            if eval $CMD 2>&1 | tee "$LOG_FILE"; then
                RUN_END=$(date +%s)
                RUN_MINS=$(( (RUN_END - RUN_START) / 60 ))
                echo "  DONE in ${RUN_MINS}m"
            else
                FAILED=$((FAILED + 1))
                echo "  FAILED (see $LOG_FILE)"
            fi

            # Kill any zombie GPU processes and wait for VRAM to clear
            pkill -9 -f "memory_bench.train" 2>/dev/null || true
            pkill -9 -f "torchrun" 2>/dev/null || true
            sleep 15
        done
    done
done

# ---------------------------------------------------------------------------
# Summary
END_TIME=$(date +%s)
TOTAL_MINS=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "============================================================"
echo "  EXPERIMENT SUITE COMPLETE"
echo "============================================================"
echo "  Total time:  ${TOTAL_MINS}m"
echo "  Succeeded:   $((TOTAL - FAILED))/$TOTAL"
echo "  Failed:      $FAILED"
echo ""

# Run the bench aggregator to generate summary + plots
echo "Aggregating results..."
python -c "
from memory_bench.bench import collect_results, aggregate_results, write_csv, print_statistical_summary
from memory_bench.plot import generate_all_plots
from datetime import datetime

results = collect_results('results')
if results:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    write_csv(results, f'results/benchmark_{ts}.csv')

    agg = aggregate_results(results)
    print()
    print(f\"{'Mechanism':<25} {'Context':>7} {'Seeds':>5} {'BPB Mean':>10} {'BPB Std':>10} {'Time (min)':>10}\")
    print('-' * 75)
    for key, data in sorted(agg.items()):
        bpb = f\"{data['mean_bpb']:.5f}\" if data['mean_bpb'] is not None else 'N/A'
        std = f\"{data['std_bpb']:.5f}\" if data['std_bpb'] is not None else 'N/A'
        t = f\"{data['mean_time_min']:.1f}\" if data['mean_time_min'] is not None else 'N/A'
        print(f\"{data['mechanism']:<25} {data['max_seq_len']:>7} {data['n_seeds']:>5} {bpb:>10} {std:>10} {t:>10}\")

    generate_all_plots(results, output_dir='results/figures')
    print(f'Figures saved to results/figures/')
    print_statistical_summary(results)
else:
    print('No result files found.')
"

echo ""
echo "Results in: results/"
echo "Figures in: results/figures/"
echo "Logs in:    results/logs/"
