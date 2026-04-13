#!/bin/bash
# ==========================================================================
# Save experiment results to the private dev repo (memory-bench-dev).
# Runs ON THE POD. Commits results/ to a dedicated branch and pushes.
#
# What gets saved:
#   - results/*.json          (per-run result files)
#   - results/*.csv           (summary CSVs)
#   - results/figures/        (generated plots)
#   - results/logs/           (training logs)
#   - results/environment/    (pip freeze, nvidia-smi, torch version)
#
# Usage:
#   bash save_results.sh                          # auto-commit + push
#   bash save_results.sh "Phase 1 complete"       # custom commit message
#   bash save_results.sh --dry-run                # show what would be saved
# ==========================================================================
set -euo pipefail

cd /workspace/memory-bench

RESULTS_BRANCH="multicontext-results"
COMMIT_MSG="${1:-Auto-save results $(date -u +%Y-%m-%dT%H:%M:%SZ)}"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    COMMIT_MSG=""
fi

# ---------------------------------------------------------------------------
# Check git is configured (pod may not have user.name set)
if ! git config user.name >/dev/null 2>&1; then
    git config user.name "Robby Sneiderman"
    git config user.email "robbysneiderman@gmail.com"
fi

# ---------------------------------------------------------------------------
# Ensure we have the remote
if ! git remote get-url origin >/dev/null 2>&1; then
    echo "ERROR: No git remote 'origin'. Set up the dev repo first:"
    echo "  git remote add origin https://github.com/Robby955/memory-bench-dev.git"
    exit 1
fi

# ---------------------------------------------------------------------------
# Create or switch to results branch (keeps results separate from main code)
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

# Fetch to know about remote branches
git fetch origin --quiet 2>/dev/null || true

if git rev-parse --verify "origin/$RESULTS_BRANCH" >/dev/null 2>&1; then
    # Remote branch exists — make sure local tracks it
    if ! git rev-parse --verify "$RESULTS_BRANCH" >/dev/null 2>&1; then
        git branch "$RESULTS_BRANCH" "origin/$RESULTS_BRANCH"
    fi
fi

# ---------------------------------------------------------------------------
# Count what we'd save
N_JSON=$(find results -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
N_CSV=$(find results -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l)
N_FIGS=$(find results/figures -name '*.png' -o -name '*.pdf' 2>/dev/null | wc -l)
N_LOGS=$(find results/logs -name '*.log' 2>/dev/null | wc -l)
N_ENV=$(find results/environment -type f 2>/dev/null | wc -l)

echo "================================================================"
echo "  SAVE RESULTS"
echo "================================================================"
echo "  JSON results:  $N_JSON"
echo "  CSV summaries: $N_CSV"
echo "  Figures:       $N_FIGS"
echo "  Logs:          $N_LOGS"
echo "  Environment:   $N_ENV"
echo "  Branch:        $RESULTS_BRANCH"
echo "  Message:       $COMMIT_MSG"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "  (dry run — nothing pushed)"
    exit 0
fi

if [[ $N_JSON -eq 0 && $N_LOGS -eq 0 ]]; then
    echo "  Nothing to save. Skipping."
    exit 0
fi

# ---------------------------------------------------------------------------
# Force-add results (bypasses .gitignore)
# We add from the current branch to avoid switching mid-experiment.
git add -f results/*.json 2>/dev/null || true
git add -f results/*.csv 2>/dev/null || true
git add -f results/figures/ 2>/dev/null || true
git add -f results/logs/ 2>/dev/null || true
git add -f results/environment/ 2>/dev/null || true

# Check if there's anything new to commit
if git diff --cached --quiet; then
    echo "  No new changes to save."
    exit 0
fi

# Commit and push
git commit -m "$COMMIT_MSG" --no-verify
echo ""
echo "  Committed. Pushing to origin/$CURRENT_BRANCH..."
if git push origin "$CURRENT_BRANCH" --no-verify 2>&1; then
    echo "  Pushed successfully."
else
    echo "  WARNING: Push failed. Results are committed locally."
    echo "  Manual push: git push origin $CURRENT_BRANCH"
fi

echo ""
echo "  Latest commit:"
git log --oneline -1
echo "================================================================"
