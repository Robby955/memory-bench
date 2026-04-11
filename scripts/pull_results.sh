#!/bin/bash
# Pull experiment results from a remote pod back to local.
# Usage: ./pull_results.sh <host> <port>
# Example: ./pull_results.sh root@my-pod.example.com 22
set -euo pipefail

POD_HOST="${1:?Usage: $0 <host> <port>}"
POD_PORT="${2:-22}"
POD_KEY="$HOME/.ssh/id_ed25519"
POD_DIR="/workspace/memory-bench/results"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DIR="$SCRIPT_DIR/results"

RSYNC="rsync -avz -e 'ssh -p $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=no'"

echo "Pulling results from pod..."
mkdir -p "$LOCAL_DIR"
eval $RSYNC "$POD_HOST:$POD_DIR/" "$LOCAL_DIR/"

echo ""
echo "Results pulled to: $LOCAL_DIR"
ls -la "$LOCAL_DIR"/*.json 2>/dev/null || echo "(no JSON results yet)"
ls -la "$LOCAL_DIR"/*.csv 2>/dev/null || echo "(no CSV summaries yet)"
ls -la "$LOCAL_DIR"/figures/ 2>/dev/null || echo "(no figures yet)"
