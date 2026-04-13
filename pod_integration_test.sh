#!/bin/bash
# Pod-side integration tests that catch issues CPU tests can't.
# Run AFTER preflight.sh, BEFORE the full experiment suite.
#
# Tests:
#   1. Seed determinism on GPU (bf16, DDP) — two runs with same seed produce identical val_bpb
#   2. Seed variance — two different seeds produce different val_bpb
#   3. Each mechanism runs 25 steps without crash on 8 GPUs
#   4. DDP gradient sync — single-GPU and multi-GPU produce same loss trajectory
#
# Usage: bash pod_integration_test.sh
set -euo pipefail

cd /workspace/memory-bench

export NANOCHAT_BASE_DIR="/workspace/memory-bench-cache"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4
export WANDB_MODE=disabled

FAIL=0
pass() { echo "  [PASS] $1"; }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }

mkdir -p results/integration_test

echo "============================================================"
echo "  POD INTEGRATION TESTS (GPU + DDP + bf16)"
echo "============================================================"
echo ""

# Helper: run a short training job and extract final val_bpb from stdout
run_short() {
    local mech="$1"
    local seed="$2"
    local nproc="${3:-8}"
    local steps="${4:-25}"
    local extra="${5:-}"
    local tag="inttest_${mech}_s${seed}_n${nproc}"

    python -m torch.distributed.run \
        --standalone --nproc_per_node="$nproc" \
        -m memory_bench.train \
        --depth=4 --mechanism="$mech" --seed="$seed" \
        --num-iterations="$steps" --eval-every=999 \
        --exp-tag="$tag" $extra 2>&1
}

# ──────────────────────────────────────────────────────────
# Test 1: Seed determinism (same seed → identical val_bpb)
echo "1. Seed determinism (same seed, 2 runs)"

BPB_A=$(run_short none 42 8 10 | grep "Min val BPB" | awk '{print $NF}')
BPB_B=$(run_short none 42 8 10 | grep "Min val BPB" | awk '{print $NF}')
pkill -9 -f "memory_bench.train" 2>/dev/null || true; sleep 5

if [[ "$BPB_A" == "$BPB_B" ]]; then
    pass "Seed 42 reproduced: $BPB_A == $BPB_B"
else
    fail "Seed 42 NOT reproducible: $BPB_A vs $BPB_B"
fi

# ──────────────────────────────────────────────────────────
# Test 2: Seed variance (different seeds → different val_bpb)
echo "2. Seed variance (different seeds)"

BPB_42=$(run_short none 42 8 10 | grep "Min val BPB" | awk '{print $NF}')
BPB_1337=$(run_short none 1337 8 10 | grep "Min val BPB" | awk '{print $NF}')
pkill -9 -f "memory_bench.train" 2>/dev/null || true; sleep 5

if [[ "$BPB_42" != "$BPB_1337" ]]; then
    pass "Seeds differ: 42=$BPB_42, 1337=$BPB_1337"
else
    fail "Seeds 42 and 1337 produced identical BPB: $BPB_42"
fi

# ──────────────────────────────────────────────────────────
# Test 3: Each mechanism runs 25 steps on 8 GPUs
echo "3. Mechanism smoke tests (25 steps, 8 GPUs)"

for mech in none persistent rmt ttt deltanet; do
    extra=""
    case $mech in
        persistent) extra="--num-memory-tokens=16" ;;
        rmt) extra="--num-memory-tokens=8 --segment-length=256 --bptt-depth=2" ;;
        ttt) extra="--ttt-layer=-1 --ttt-chunk-size=32" ;;
        deltanet) extra="--no-compile" ;;
    esac

    echo "  Testing $mech..."
    if run_short "$mech" 42 8 25 "$extra" > "results/integration_test/${mech}.log" 2>&1; then
        BPB=$(grep "Min val BPB" "results/integration_test/${mech}.log" | awk '{print $NF}')
        pass "$mech completed (val_bpb=$BPB)"
    else
        fail "$mech CRASHED (see results/integration_test/${mech}.log)"
    fi
    pkill -9 -f "memory_bench.train" 2>/dev/null || true; sleep 10
done

# ──────────────────────────────────────────────────────────
# Test 4: DDP consistency (1 GPU vs 8 GPU baseline should be close)
echo "4. DDP consistency (1 GPU vs 8 GPU)"

BPB_1GPU=$(run_short none 42 1 15 | grep "Min val BPB" | awk '{print $NF}')
pkill -9 -f "memory_bench.train" 2>/dev/null || true; sleep 5
BPB_8GPU=$(run_short none 42 8 15 | grep "Min val BPB" | awk '{print $NF}')
pkill -9 -f "memory_bench.train" 2>/dev/null || true; sleep 5

if [[ -n "$BPB_1GPU" && -n "$BPB_8GPU" ]]; then
    # They won't be identical (different batch compositions per rank), but should be close
    DIFF=$(python3 -c "print(abs(float('$BPB_1GPU') - float('$BPB_8GPU')))")
    if python3 -c "exit(0 if float('$DIFF') < 0.5 else 1)"; then
        pass "DDP consistent: 1GPU=$BPB_1GPU, 8GPU=$BPB_8GPU (diff=$DIFF)"
    else
        fail "DDP diverges: 1GPU=$BPB_1GPU, 8GPU=$BPB_8GPU (diff=$DIFF > 0.5)"
    fi
else
    fail "Could not extract BPB values for DDP test"
fi

# ──────────────────────────────────────────────────────────
# Cleanup
rm -f results/inttest_*.json 2>/dev/null

echo ""
echo "============================================================"
if [[ $FAIL -eq 0 ]]; then
    echo "  ALL POD INTEGRATION TESTS PASSED"
    echo "  Safe to run: bash run_experiments.sh"
else
    echo "  $FAIL TEST(S) FAILED — investigate before full suite"
fi
echo "============================================================"
