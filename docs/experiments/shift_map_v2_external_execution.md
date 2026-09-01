# SHIFT-MAP v2 external compute handoff (Step 8.3)

## Frozen scope

This handoff is engineering-only. Scientific Step 8 remains version `2.0`; the handoff is `8.3`. It preserves MedCPT `ncbi/MedCPT-Cross-Encoder` at revision `71caf65d4927987813984f54c284405a13fcca49`, candidate K `100`, source/target description-only input, dynamic padding, and max length `96`. TEST is forbidden until a separately approved `test_lock.json` exists.

Operational guidance is conservative: CUDA GPU with at least 8 GB VRAM; 12 GB preferred. This is not a scientific requirement. Do not select a provider in Step 8.3.

## Setup on external compute

```bash
git clone <repository-url>
cd SHIFT-ICD
python -m venv .venv
# activate the environment for your shell
python -m pip install --upgrade pip
# install PyTorch from the host-compatible CUDA channel, then:
python -m pip install -r requirements-step8.txt
```

Copy these files separately, preserving their repository-relative paths, because they are large and/or ignored by Git:

- `artifacts/candidates/shift_map_v2/manifest.json`
- `artifacts/candidates/shift_map_v2/forward_stratified_train_k100.jsonl.gz`
- `artifacts/candidates/shift_map_v2/forward_stratified_dev_k100.jsonl.gz`
- `artifacts/candidates/shift_map_v2/forward_stratified_test_k100.jsonl.gz`
- `data/benchmarks/cms_track_a/v1.0/forward/stratified/train.jsonl`
- `data/benchmarks/cms_track_a/v1.0/forward/stratified/dev.jsonl`
- `data/benchmarks/cms_track_a/v1.0/forward/stratified/test.jsonl`

## Gates and model

```bash
python scripts/external_step8/run_step8.py verify-data
python scripts/external_step8/run_step8.py preflight
# local handoff-only validation (does not require CUDA):
python scripts/external_step8/run_step8.py preflight --cpu-check
```

Download only the exact revision through Hugging Face's normal mechanism, with no quantization or CPU offload. Record the resolved commit and snapshot file hashes. Pass the resulting local snapshot path to the runner; the runner uses `local_files_only=True` and the frozen revision.

## Smoke and DEV

```bash
python scripts/external_step8/run_step8.py smoke --model-path /path/to/snapshot
python scripts/external_step8/run_step8.py zero-shot-dev --model-path /path/to/snapshot
python scripts/external_step8/run_step8.py status
```

Smoke uses exactly 10 DEV sources / 1,000 pairs. It checks 100 candidates per source, scalar finite logits, deterministic ordering, and writes to `artifacts/experiments/shift_map_v2_external_runs/<run_id>/`. Full external DEV starts from source 0 and never merges the old local 10,000-pair provenance chunks.

Scoring is FP32 initially. Chunks contain 50 sources / 5,000 pairs, are written atomically, and are recorded with source range, pair count, SHA-256, revision, precision, device, and timestamp/run identity. An interruption is resumed by rerunning the same command with the same output root; completed hashed chunks are retained. Do not delete or merge the old local chunks.

## Later scientific stages

Only after complete external DEV and the required gates are independently accepted:

1. training-probe: BCE and listwise forward/loss/backward, candidate list 8, micro-batches 1/2/4, no optimizer step;
2. objective and negative-strategy DEV ablations;
3. learning-rate and optional initialization controls;
4. final seeds 17/42/2026;
5. create the approved TEST lock;
6. TEST only after the lock and all prior gates pass.

The current handoff does not create the final TEST lock. The future lock must include model revision, candidate hashes, K, max length, objective, negative strategy, learning rate, epochs, seeds, canonical seed, alpha if selected, environment hash, GPU identity, Git commit, and prior TEST-exposure disclosure.

## Sync-back

```bash
python scripts/external_step8/export_results.py artifacts/experiments/shift_map_v2_external_runs/<run_id> ./step8_external_export
```

Transfer only the compact export: manifests, metrics, configs, training curves, table-ready CSVs, and hashes. Do not export weights, giant logits, token caches, credentials, or raw unnecessary data. Verify the export hash and originating Git/model revision before import-back.

## Current local decision

The laptop remains `CF4`: two guarded AC-loss stops. Do not run another long local GPU job in Step 8.3. The next milestone is **Step 8.4 — external compute validation and zero-shot DEV completion**.
