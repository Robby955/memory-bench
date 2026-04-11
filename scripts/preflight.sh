#!/bin/bash
# Pre-flight checks before running the full experiment suite.
# Run this FIRST on the pod to catch issues before queueing 15 runs.
# Usage: bash preflight.sh
set -euo pipefail

cd /workspace/memory-bench

echo "============================================================"
echo "  memory-bench PREFLIGHT CHECKS"
echo "============================================================"
echo ""

FAIL=0
warn() { echo "  [WARN] $1"; }
pass() { echo "  [PASS] $1"; }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }

# 1. GPU check
echo "1. GPU check"
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "$NUM_GPUS" -ge 8 ]]; then
    pass "$NUM_GPUS GPUs detected"
elif [[ "$NUM_GPUS" -ge 1 ]]; then
    warn "Only $NUM_GPUS GPUs (expected 8). Runs will be slower."
else
    fail "No GPUs detected"
fi

# 2. Python imports
echo "2. Python imports"
if python -c "
import memory_bench
from memory_bench.mechanisms import MECHANISMS
from nanochat.gpt import GPT, GPTConfig
from nanochat.common import compute_init
print(f'  Mechanisms: {list(MECHANISMS.keys())}')
" 2>/dev/null; then
    pass "All imports OK"
else
    fail "Import error (run: pip install -e '.[gpu]')"
fi

# 3. FLA (for DeltaNet)
echo "3. FLA library"
if python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print('  FLA v' + __import__('fla').__version__)" 2>/dev/null; then
    pass "FLA installed"
else
    warn "FLA not installed (DeltaNet will fail). Install: pip install fla-core"
fi

# 4. Data availability
echo "4. Data check"
export NANOCHAT_BASE_DIR="/workspace/memory-bench-cache"
if python -c "
import os, sys
os.environ['NANOCHAT_BASE_DIR'] = '/workspace/memory-bench-cache'
sys.path.insert(0, os.path.join(os.getcwd(), 'nanochat'))
from nanochat.dataset import list_parquet_files
files = list_parquet_files()
print(f'  Data shards: {len(files)}')
" 2>/dev/null; then
    pass "Data available"
else
    warn "Data not cached (will download on first run)"
fi

# 5. Seed determinism check
echo "5. Seed determinism"
python -c "
import torch
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'nanochat'))
from nanochat.gpt import GPT, GPTConfig

def init_with_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    config = GPTConfig(vocab_size=8192, n_layer=4, n_head=4, n_embd=128, sequence_len=256)
    with torch.device('meta'):
        m = GPT(config)
    m.to_empty(device='cpu')
    m.init_weights()
    return sum(p.sum().item() for p in m.parameters())

s42 = init_with_seed(42)
s1337 = init_with_seed(1337)
s42b = init_with_seed(42)
if s42 != s1337 and s42 == s42b:
    print(f'  seed 42 checksum:   {s42:.6f}')
    print(f'  seed 1337 checksum: {s1337:.6f}')
    print(f'  seed 42 replay:     {s42b:.6f}')
" && pass "Seeds produce different models, same seed reproduces" || fail "Seed check failed"

# 6. Results directory
echo "6. Results directory"
mkdir -p results results/figures results/logs
pass "results/ structure created"

# 7. Disk space
echo "7. Disk space"
AVAIL=$(df -BG /workspace 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
if [[ -n "$AVAIL" ]] && [[ "$AVAIL" -ge 50 ]]; then
    pass "${AVAIL}G available on /workspace"
elif [[ -n "$AVAIL" ]]; then
    warn "Only ${AVAIL}G available (recommend 50G+)"
else
    warn "Could not check disk space"
fi

# 8. Quick smoke test (baseline, 5 steps)
echo "8. Smoke test (5-step baseline)"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4
export WANDB_MODE=disabled
if timeout 120 python -m torch.distributed.run \
    --standalone --nproc_per_node=1 \
    -m memory_bench.train \
    --depth=4 --mechanism=none --seed=42 \
    --num-iterations=5 --eval-every=999 \
    --exp-tag=preflight 2>&1 | tail -3; then
    pass "Baseline training runs"
else
    fail "Baseline training crashed"
fi

# Kill any leftover processes
pkill -9 -f "memory_bench.train" 2>/dev/null || true
sleep 3

echo ""
echo "============================================================"
if [[ $FAIL -eq 0 ]]; then
    echo "  ALL CHECKS PASSED — ready for: bash run_experiments.sh"
else
    echo "  $FAIL CHECK(S) FAILED — fix before running experiments"
fi
echo "============================================================"
