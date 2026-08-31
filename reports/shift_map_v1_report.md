# SHIFT-MAP v1 — Step 7 Report

## Evidence status

**Executed and reproducible locally; not a successful improvement over zero-shot BioLORD.** The final model, analyses, and audit artifacts were regenerated after correcting two aggregation issues: TEST rows now preserve `valid_target_codes`, and seed-slice tables report unique benchmark-example counts rather than raw seed-row counts.

## Objective and frozen boundary

SHIFT-MAP v1 tests BioLORD-initialized supervised source-to-target contrastive adaptation for ICD-9-CM → ICD-10-CM retrieval. Training used only the permitted forward stratified TRAIN population. DEV selected the configuration. TEST was evaluated only after the immutable lock at `artifacts/experiments/shift_map_v1/test_lock.json`.

Frozen base model: `FremyCompany/BioLORD-2023`, revision `167aab527b238a50ca65224e6319215d2ff4fc9f`. Frozen comparators remain `dense_v1_1`/BioLORD and `bm25_v1_1`.

## Training design

- Positive policy: `P2` (`SINGLE_EXACT`, `SINGLE_APPROXIMATE`, and `ALTERNATIVE`).
- Objective: `L1` masked single-positive InfoNCE.
- Negatives: `N1_RANDOM` after DEV comparison against lexical, dense, same-family, and mixed alternatives.
- Learning rate: `2e-5`, selected on DEV from the preregistered candidates.
- Optimizer: AdamW; weight decay `0.01`; temperature `0.05`; three epochs; maximum length 64; effective source batch 32.
- Seeds: `17`, `42`, and `2026`; canonical seed: `42`.
- Combinations and `NO_MAP` examples were excluded from pairwise training.
- Valid alternatives were retained as positive sets and excluded from sampled negatives.
- No symmetric target-to-source objective, reranker, calibration, conformal layer, routing, GNN, or cardinality model was implemented.

## Training population and contamination audit

The eligible forward TRAIN population contains `9,434` source concepts. The negative-mining manifest records `0` gold collisions, `0` remaining collisions, and no DEV or TEST data use.

The final validator reports:

```text
DEV IDs in training: 0
TEST IDs in training: 0
backward TEST IDs in training: 0
complex sources in training: 0
NO_MAP sources in training: 0
remaining gold collisions: 0
```

## Final forward TEST results

The primary ordinary, answerable, non-combination population contains `2,695` unique benchmark examples.

Three-seed results:

| Seed | Hit@1 | Hit@5 | Hit@10 | Hit@25 | Hit@50 | Hit@100 | MRR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.018553 | 0.738033 | 0.844527 | 0.929499 | 0.966605 | 0.983673 | 0.352772 |
| 42 | 0.019666 | 0.735807 | 0.847124 | 0.932468 | 0.966976 | 0.983673 | 0.353663 |
| 2026 | 0.018182 | 0.740260 | 0.854545 | 0.934323 | 0.969202 | 0.984045 | 0.356356 |
| **Mean** | **0.018800** | **0.738033** | **0.848732** | **0.932096** | **0.967594** | **0.983797** | **0.354264** |

Canonical seed 42 is the locked reporting seed. The corresponding per-example output is stored locally in `artifacts/experiments/shift_map_v1/test_metrics.json` and is intentionally not committed because it is a large generated file.

## Frozen BioLORD comparison

Frozen zero-shot BioLORD forward TEST values are:

```text
Hit@1   0.717996
Hit@5   0.882004
Hit@10  0.921336
Hit@25  0.957328
Hit@50  0.974397
Hit@100 0.984045
MRR     0.792142
```

Paired bootstrap comparison uses the same `2,695` benchmark IDs, `5,000` resamples, and bootstrap seed `2026`:

| Metric | SHIFT-MAP − BioLORD | 95% CI |
|---|---:|---:|
| Hit@10 | −0.074212 | [−0.085343, −0.063822] |
| Hit@100 | −0.000371 | [−0.004453, +0.003340] |
| MRR | −0.438479 | [−0.447806, −0.429189] |

The supported conclusion is a clear early-rank and MRR regression, with near-parity only at Hit@100. This is a negative adaptation result, not evidence that supervised adaptation improved retrieval.

Frozen BM25 remains below SHIFT-MAP on the available forward Hit@10/Hit@100 comparison, but that does not offset the regression against the stronger zero-shot BioLORD comparator.

## Structural and transfer analyses

Machine-readable tables cover lexical difficulty, mapping kind, alternative-set size, family-held-out splits, reverse-direction transfer, combination transfer, `NO_MAP` diagnostics, three-seed results, and paired comparisons.

- `final_backward_transfer.csv` contains only `ICD10CM_TO_ICD9CM`; it is transfer from a forward-trained model, not backward training.
- `final_combination_transfer.csv` contains only `COMBINATION` and `COMBINATION_WITH_ALTERNATIVES` rows.
- Alternative-size bins are computed from the serialized `valid_target_codes`, not from a per-row placeholder.
- `NO_MAP` outputs are post-hoc similarity diagnostics only; no threshold, calibration, abstention, or routing was selected.
- CMS GEM-derived retrieval is not clinical validation or proof of clinical equivalence.

## Artifact and quality gates

The committed Step 7 namespace contains the protocol, source modules, training/evaluation/aggregation scripts, manifests, compact tables, figures, and report. Checkpoints remain ignored and local.

Verified gates:

```text
pytest: 56 passed
Ruff: All checks passed
mypy: Success: no issues found in 27 source files
training contamination validator: passed
artifact validator: passed
raw/canonical/benchmark/BM25/dense frozen validators: passed
```

The final artifact manifest records `51,762` TEST rows across three seeds and preserves the TEST-lock SHA-256:

```text
e4258a0bc85faf9f85b01d5281550ffa2ddd885da8fb0e421970eecb91def58a
```

## Limitations

Representation-drift statistics and a fully serialized mechanistic success/regression sample remain follow-up diagnostics. The committed figures are navigation artifacts pointing readers to machine-readable tables; they are not independent evidence beyond those tables. No claim of clinical validation is made.

## Decision

**BLOCKED for SHIFT-MAP v2 cross-encoder reranking.** First investigate why training preserves broad Hit@100 while substantially damaging Hit@10 and MRR. Any redesign must use a separately approved DEV-only protocol and must not use these TEST results to tune the next configuration.
