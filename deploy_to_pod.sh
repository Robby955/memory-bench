#!/bin/bash
# Deploy memory-bench to a remote pod.
# Usage: ./deploy_to_pod.sh <host> <port>
# Example: ./deploy_to_pod.sh root@my-pod.example.com 22
set -euo pipefail

# Pod config — set these for your environment
POD_HOST="${1:?Usage: $0 <host> <port>}"
POD_PORT="${2:-22}"
POD_KEY="$HOME/.ssh/id_ed25519"
POD_DIR="/workspace/memory-bench"

SSH="ssh -p $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=no $POD_HOST"
RSYNC="rsync -avz --delete -e 'ssh -p $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=no'"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== memory-bench deploy ==="
echo "Pod: $POD_HOST:$POD_PORT"
echo "Source: $SCRIPT_DIR"
echo "Target: $POD_DIR"
echo ""

# Test connection
echo "Testing SSH connection..."
if ! $SSH "echo 'Connected'" 2>/dev/null; then
    echo "ERROR: Cannot connect to pod. Check host and port."
    exit 1
fi

# Create target directory
$SSH "mkdir -p $POD_DIR"

# Rsync memory-bench (exclude heavy/unnecessary files)
echo "Syncing memory-bench..."
eval $RSYNC \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='results/' \
    --exclude='architecture.pdf' \
    --exclude='*.egg-info' \
    "$SCRIPT_DIR/" "$POD_HOST:$POD_DIR/"

# Rsync nanochat submodule
echo "Syncing nanochat submodule..."
eval $RSYNC \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    "$SCRIPT_DIR/nanochat/" "$POD_HOST:$POD_DIR/nanochat/"

# Install on pod
echo "Installing memory-bench on pod..."
$SSH << 'INSTALL'
cd /workspace/memory-bench
pip install --break-system-packages -e ".[gpu]" 2>&1 | tail -5
pip install --break-system-packages tokenizers sentencepiece rustbpe datasets tiktoken 2>&1 | tail -3
# Verify imports work
python -c "
import memory_bench
from memory_bench.mechanisms import MECHANISMS
print(f'memory-bench installed. Mechanisms: {list(MECHANISMS.keys())}')
from nanochat.gpt import GPT, GPTConfig
print('nanochat imports OK')
"
INSTALL

# Set up git on pod for result saving
echo "Setting up git for result saving..."
$SSH << 'GITSETUP'
cd /workspace/memory-bench
git init 2>/dev/null || true
git config user.name "Robby Sneiderman"
git config user.email "robbysneiderman@gmail.com"
# Add dev repo as remote if not already set
if ! git remote get-url origin >/dev/null 2>&1; then
    git remote add origin https://github.com/Robby955/memory-bench-dev.git
fi
echo "Git configured. Remote: $(git remote get-url origin)"
GITSETUP

echo ""
echo "=== Deploy complete ==="
echo ""
echo "IMPORTANT: For automatic result saving, set up git auth on the pod:"
echo "  ssh $POD_HOST -p $POD_PORT"
echo "  gh auth login   # or set GITHUB_TOKEN"
echo ""
echo "Run experiments with:"
echo "  cd /workspace/memory-bench && bash run_multicontext.sh"
