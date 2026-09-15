# STEP 8-R2 — Corrected MedCPT DEV configuration selection

Status: **BLOCKED_STEP8_R2_TRAINING_CONTRACT_INCONSISTENCY**

## Provenance preflight

- Starting HEAD: `d2818991fee17ed1a581f83a462611e6127f23b8` — `analysis: complete corrected Step 8-R1 runner preflight`
- R2 runner/protocol implementation checkpoint: `cdc74c9f4772653cf217d7ffd25ac03273240b99`
- Worktree was clean at R2 start.
- R1 model snapshot, runner contract, objective search space, DEV selection rule, training population manifest, candidate hashes, candidate contract, and TEST quarantine all passed preflight.
- Fresh local gates before remote launch: 131 tests passed; Ruff passed; mypy passed for 35 source files; package import passed; diff check passed.

## Frozen R2 protocol

`artifacts/experiments/step8_full_universe/r2_search_protocol.json` was written before any R2 outcome. SHA-256: `999119c1c9ea810dc05eb234a346f79e0f642433087faf93497c9b10812bc719`.

- Objectives: `BCE`, `SET_POSITIVE_LISTWISE`
- Negative strategies: `TOP_RANK_HARD`, `MIXED_RANK`, `RANDOM_WITHIN_CANDIDATE`
- Learning-rate grid: `1e-5`, `2e-5`, `3e-5`
- Development seed: `17`
- Optimizer: `AdamW`
- Weight decay: `0.01`
- Warmup ratio: `0.1`
- Gradient clipping: `1.0`
- Source/effective batch size: `32`
- Maximum epochs: `3`
- Max length: `96`
- Precision: `FP32`
- Pair order: source description, target description
- Selection population: corrected forward DEV only
- TEST scoring/training: prohibited, counts fixed at zero

## Epoch and DEV selection rules

`epoch_selection_rule.json` was written before outcomes. SHA-256: `29c5d89f710068b78ca46797b82e2d3fdc3fc8239a3e695bba876402198e519d`.

- Epoch rule: `E2_BEST_DEV_CHECKPOINT_WITHIN_MAX_EPOCH_BUDGET`
- Maximum epochs: `3`
- Metric key: Hit@1 descending, MRR descending, Hit@10 descending, NDCG@10 descending, CompleteScenarioRetrieval@10 descending
- Tie-break: earliest epoch
- Historical evidence: the prior corrected SHIFT-MAP configuration freeze used this E2-within-budget rule; it was frozen before R2 outcomes.

## Exact blocker

The frozen R1 training contract specifies both:

1. source-balanced list size `8`; and
2. all candidate-contained valid alternatives retained as positives and excluded from negative selection.

Recomputation from the frozen TRAIN candidate file and authoritative benchmark gives:

- Ordinary TRAIN sources: `9,434`
- Gold-present ordinary TRAIN sources: `9,224`
- Gold-missing ordinary TRAIN sources: `210`
- Gold-present sources with more than 8 candidate-contained positives: `55`
- Maximum candidate-contained positive count: `49`
- Affected mapping kind: `ALTERNATIVE`

Therefore the current frozen contract cannot construct valid list-8 examples for all eligible sources without either dropping positives or expanding list size. Both changes are prohibited during R2. The exact blocker is recorded in `artifacts/experiments/step8_full_universe/r2_blocker.json`.

## Execution boundary

A new remote namespace was created and GPU preflight passed on RTX 4090 GPU 0. The first attempt failed before model loading because `runner.py` was missing from the transferred namespace. The dependency was synchronized and the one permitted corrected retry was run. It then failed closed at TRAIN list construction on the positive-count conflict above.

- Remote namespace retained: `/home/gra_rohan/Rohan/tasks/step8_r2_20260915T000000Z_8b7c1a4e`
- Remote retry exit code: `1`
- Objective/strategy/LR outcomes: **NOT COMPUTED**
- DEV reranking metrics: **NOT COMPUTED**
- Selected objective/strategy/LR: **NOT COMPUTED**
- Corrected TEST scoring: `0`
- Corrected TEST training: `0`
- No corrected TEST rows were scored or trained on.
- No final seed was trained, no TEST lock was created, and no publication claim was produced.

## Required resolution

R2 must remain blocked until R1 is formally amended by the user/protocol owner to resolve the contradiction, for example by explicitly defining a variable-length list policy or a positive-preserving list construction that is compatible with list size 8. No such amendment was authorized or applied in the blocked R2 attempt.

## TRAINING CONTRACT AMENDMENT (R2A)

The blocker is **CLOSED** for contract purposes, before DEV scientific evaluation and without using outcome information. The amended v2 contract preserves all candidate-contained valid positives, requires `N_i = max(1, 8 - P_i)` negatives, and permits only the affected source lists to exceed the minimum width of 8. It uses source-normalized BCE and source-level listwise loss, preserves frozen candidate membership, and keeps corrected TEST access at zero. The original blocked record remains preserved in `artifacts/experiments/step8_full_universe/r2_blocker.json`; this section does not rewrite that historical state. The preliminary blocker record reported 55 sources over the width; the exact R2A TRAIN-only audit established the corrected count as 48, with 27 sources at exactly 8 and 75 at least 8.

- Contract v2: `artifacts/experiments/step8_full_universe/training_list_contract_v2.json`
- Protocol v2: `artifacts/experiments/step8_full_universe/r2_search_protocol_v2.json`
- Interface: `docs/interfaces/step8_training_list_contract_v2.md`
- TRAIN audit: `artifacts/experiments/step8_full_universe/training_list_audit_v2.json`
- Amendment timing: before any scientific DEV metric
- Outcome information used: none
- Corrected TEST scoring/training: 0/0

The next permitted milestone is STEP 8-R2B — CORRECTED MEDCPT DEV ABLATIONS + CONFIGURATION FREEZE UNDER CONTRACT V2.
