# STEP 8-R1 — Corrected preflight and runner validation

Status: **STEP8_CORRECTED_RUNNER_READY**

R1 outcome: the corrected production runner now exposes candidate audit, model snapshot, DEV scoring, bounded TRAIN objective/checkpoint smoke, resume validation, and manifest emission through one CLI. The runner is ready for `STEP 8-R2 — CORRECTED MEDCPT DEV ABLATIONS + CROSS-ENCODER CONFIGURATION FREEZE`. No full DEV ablation/search or final Step 8 claim was performed.

## Frozen inputs

- Retrieval status: `SHIFT_MAP_FULL_UNIVERSE_FROZEN`
- Candidate contract SHA-256: `20ebe99b2fbf2a82773a3cd1e5778cdcb68c92b9c830ad689f4afd06e36a07e9`
- Candidate files: corrected `shift_map_full_universe_v2`, K=100; all three hashes/counts pass.
- Canonical checkpoint SHA-256: `9e9739e45ad041a4027cdfd611fb83e398debb50658ede710482660e16efe302`

## MedCPT snapshot

- ID: `ncbi/MedCPT-Cross-Encoder`
- Revision: `71caf65d4927987813984f54c284405a13fcca49`
- Class: `BertForSequenceClassification`; architecture BERT sequence classification; num_labels=1; parameters 109,483,009; max positions 512.
- Tokenizer: `BertTokenizerFast`, revision pinned identically, vocab 30,522.
- Scalar score: validated `output.logits[:, 0]` only because the pinned config is exactly `[batch, 1]`; no softmax.
- Input: `(source description, target description)` in that order; no codes, ranks, retriever scores, labels, mapping metadata, or scenario metadata.

## Tokenization and precision

- Maximum length: 96; tokenizer truncation is right-side/default model truncation with pair special-token construction and dynamic longest padding per batch.
- Full TRAIN/DEV pre-truncation distributions are in `token_length_distribution.csv`.
- Scientific reference precision: FP32.
- Deterministic DEV smoke: 100 sources / 10,000 pairs; max score delta 0.0; ranking disagreement 0; top-1 disagreement 0.
- Throughput batches 1–64 all passed; operational default frozen to batch 32 for headroom.

## Training contracts

- Ordinary TRAIN eligible sources: 9,434; gold-present: 9,224; gold-missing: 210 (excluded).
- Eligible kinds: SINGLE_EXACT, SINGLE_APPROXIMATE, ALTERNATIVE. Structural/NO_MAP kinds excluded.
- Alternative: every valid candidate-contained alternative is positive; none is sampled as negative.
- Source-balanced list size: 8. Negatives are sampled deterministically within each source’s frozen Top-100 after excluding all positives.
- Objectives frozen to BCE and SET_POSITIVE_LISTWISE; listwise numerator includes every positive and uses stable logsumexp.
- DEV selection is frozen lexicographic Hit@1, MRR, Hit@10, NDCG@10, then CompleteScenarioRetrieval@10/frozen structural criterion; no TEST metrics participate.

## Lifecycle and safety

The corrected production runner is `scripts/step8_full_universe/runner.py`; `smoke.py` remains its bounded operational primitive, not an independent scientific path. The integrated CLI was remotely validated for candidate audit, fail-closed TEST command, DEV scoring (100 sources / 10,000 pairs), and TRAIN objective/checkpoint smoke (128 TRAIN sources). FP32 scoring, BCE/listwise finite-loss and gradient checks, parameter update, checkpoint save/reload, and resume-counter validation all passed. Smoke artifacts are explicitly `SMOKE_ONLY_NOT_SCIENTIFIC_RESULT`. The run manifests record frozen candidate hashes, contract hashes, objective, seed, precision, evaluator version, and TEST counts. Evaluator regression tests prove Hit@100 and structural coverage@100 are invariant under reorder-only reranking. TEST access defaults closed and requires a future explicit lock. Historical candidate-dependent namespaces are quarantined and rejected by policy.

Remote namespace: `/home/gra_rohan/Rohan/tasks/step8_full_universe_20260914T202500Z_6f3a9c2d`; environment: PyTorch 2.7.1+cu126, Transformers 4.52.4, CUDA 12.6, RTX 4090.

Corrected Step 8 TEST scoring count: **0**. Corrected Step 8 TEST training count: **0**.
