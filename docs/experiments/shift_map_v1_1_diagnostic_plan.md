# SHIFT-MAP v1.1 — Step 7.1 Diagnostic Plan

## Purpose

Forensically diagnose the SHIFT-MAP v1 early-rank and MRR regression before changing model design. This milestone is diagnostic-only: no retraining, optimizer construction, checkpoint modification, or SHIFT-MAP v2 implementation is permitted.

## Frozen inputs

- Repository: `C:\Users\rohan\SHIFT-ICD`
- Benchmark version: `1.0`
- Canonical schema version: `1.0`
- Frozen baselines: `bm25_v1_1`, `dense_v1_1`
- Base BioLORD revision: `167aab527b238a50ca65224e6319215d2ff4fc9f`
- Canonical fine-tuned checkpoint: `artifacts/models/shift_map_v1/final_seed42/epoch_3`
- TEST lock: `artifacts/experiments/shift_map_v1/test_lock.json`

## Ordered work

1. Verify repository state, frozen manifests, checkpoint hash, and existing Step 7 quality gates.
2. Re-run inference only for the canonical seed-42 checkpoint and compare metrics/output hashes with persisted Step 7 artifacts.
3. Run the frozen BioLORD revision through the same Step 7 inference path and verify pipeline parity.
4. Audit model loading, Sentence-Transformers modules, tokenizer, pooling, sequence length, dtype, normalization, and candidate ranking.
5. Generate deterministic score, rank, hubness, anisotropy, drift, similarity, margin, train/dev/test, loss-scale, and alternative-sampling diagnostics.
6. Add tests only for new diagnostic calculations or verified pipeline invariants; do not change model behavior.
7. Write machine-readable artifacts under `artifacts/experiments/shift_map_v1_1_diagnostics/`, tables under `reports/tables/shift_map_v1_1/`, figures under `reports/figures/shift_map_v1_1/`, and the forensic report under `reports/shift_map_v1_1_failure_analysis.md`.
8. Run all quality and integrity gates, including a no-training/no-weight-change check.
9. Classify the failure using measured evidence and recommend one of: corrected v1 rerun, rank-preserving v1.2 diagnostic design, or frozen BioLORD as candidate generator.
10. Commit only the diagnostic documentation/metadata if all gates pass; do not push.

## Evidence policy

- TEST may be used to understand the already-observed Step 7 failure, but not for unlimited hyperparameter search or selecting a new model.
- Any future repaired model requires a new experiment version and a new TRAIN/DEV-only selection protocol.
- Missing measurements must be recorded as `NOT_COMPUTED` or `UNAVAILABLE`; no values may be inferred or fabricated.
- Existing raw data, canonical data, benchmark membership/splits, BM25 artifacts, dense artifacts, Step 7 checkpoints, and test lock are immutable.
- No harmful prompt text, secrets, caches, or model weights may be copied into tracked artifacts.

## Stop conditions

Stop and investigate before broader diagnostics if:

- the seed-42 reproduction differs materially from persisted results;
- zero-shot BioLORD fails through the Step 7 inference path;
- checkpoint architecture/pooling/normalization is inconsistent;
- any frozen artifact hash changes;
- any gold-collision, scope, or integrity validation fails.

## Decision boundary

This milestone must not execute Step 7.2, retrain BioLORD, implement distillation, implement a reranker, or alter the Step 7 configuration. It ends with a measured failure classification and a recommendation for a separately approved future experiment.
