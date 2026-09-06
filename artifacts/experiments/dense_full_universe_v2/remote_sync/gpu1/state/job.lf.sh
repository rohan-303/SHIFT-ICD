#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="${TASK_ROOT:?remote-run did not provide TASK_ROOT}"
GPU="${CUDA_VISIBLE_DEVICES:?remote-run did not provide CUDA_VISIBLE_DEVICES}"
cd "$TASK_ROOT/work"
source .venv/bin/activate
export PYTHONPATH=src:.
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="$TASK_ROOT/work/hf_cache_gpu${GPU}"
export TRANSFORMERS_CACHE="$HF_HOME"
export DENSE_GIT_COMMIT='59d408e26f78adad7dcbe12529be316e9d0fd6ab'
mkdir -p "$TASK_ROOT/results" "$TASK_ROOT/logs" "$HF_HOME"
if [ "$GPU" = "0" ]; then
  DENSE_MODELS="SapBERT BioLORD-2023"
else
  DENSE_MODELS="MedCPT Qwen3-Embedding-0.6B"
fi
for model in ${DENSE_MODELS}; do
  slug=$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
  export DENSE_MODEL="$model"
  export DENSE_OUTPUT="$TASK_ROOT/results/dense_full_universe_v2/$slug"
  mkdir -p "$DENSE_OUTPUT"
  date -u +%FT%TZ > "$TASK_ROOT/logs/${slug}.start"
  python scripts/run_dense_full_universe_v2.py 2>&1 | tee "$TASK_ROOT/logs/${slug}.log"
  date -u +%FT%TZ > "$TASK_ROOT/logs/${slug}.done"
done
printf '%s\n' '{"status":"DEV_DENSE_COMPLETE","split":"forward_stratified_dev","test_evaluation":"NOT_RUN","shift_map":"NOT_RUN","cross_encoder":"NOT_RUN","gpu":"'"$GPU"'"}' > "$TASK_ROOT/results/dev_complete_gpu${GPU}.json"
