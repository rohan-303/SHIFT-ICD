#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="${TASK_ROOT:?}"
cd "$TASK_ROOT/work"
source .venv/bin/activate
export PYTHONPATH=src:.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?}"
export DENSE_TEST_LOCK_HASH="bf706bd320f13b6cda921123a24040e33efd171a5049b7659d8941c673cc94cd"
export DENSE_TARGET_ROOT="$TASK_ROOT/work/results"
export DENSE_TEST_OUTPUT="$TASK_ROOT/work/test_results"
DENSE_TEST_LOCK_HASH="bf706bd320f13b6cda921123a24040e33efd171a5049b7659d8941c673cc94cd" DENSE_TARGET_ROOT="$TASK_ROOT/work/results" DENSE_TEST_OUTPUT="$TASK_ROOT/results/test_results" python scripts/evaluate_dense_locked_v2.py
printf '%s\n' TEST_DENSE_COMPLETE > "$TASK_ROOT/results/test_results/TEST_DENSE_COMPLETE"
