#!/bin/bash
# Pull ALL experiment artifacts from a remote pod back to local.
# This is the manual backup path — run_multicontext.sh also auto-pushes
# to the dev repo after each phase, but this gives you a local copy.
#
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

echo "=== Pull Results ==="
echo "Pod: $POD_HOST:$POD_PORT"
echo "Remote: $POD_DIR"
echo "Local:  $LOCAL_DIR"
echo ""

# Create all subdirectories
mkdir -p "$LOCAL_DIR" "$LOCAL_DIR/figures" "$LOCAL_DIR/logs" "$LOCAL_DIR/environment"

# Pull everything under results/
echo "Pulling results..."
eval $RSYNC "$POD_HOST:$POD_DIR/" "$LOCAL_DIR/"

echo ""
echo "=== Summary ==="
echo "JSON results:  $(find "$LOCAL_DIR" -maxdepth 1 -name '*.json' | wc -l)"
echo "CSV summaries: $(find "$LOCAL_DIR" -maxdepth 1 -name '*.csv' | wc -l)"
echo "Figures:       $(find "$LOCAL_DIR/figures" -type f 2>/dev/null | wc -l)"
echo "Logs:          $(find "$LOCAL_DIR/logs" -name '*.log' 2>/dev/null | wc -l)"
echo "Environment:   $(find "$LOCAL_DIR/environment" -type f 2>/dev/null | wc -l)"
echo ""
echo "Results saved to: $LOCAL_DIR"
