# SHIFT-MAP v1.1 — Step 7.1 Failure Analysis

## 1. Decision status

**Step 7.1 is blocked at the pipeline-parity gate.** The canonical seed-42 fine-tuned reproduction is exact, but the zero-shot BioLORD control fails when passed through the original Step 7 evaluator. A direction-specific corpus probe restores the frozen BioLORD metrics exactly. Therefore the original Step 7 evaluator contains a target-corpus construction bug, and no representation, hubness, anisotropy, objective, or genuine-specialization conclusion is valid until a corrected v1 rerun is completed.

## 2. Scope and safety

This milestone was diagnostic-only. No BioLORD retraining occurred, no optimizer was created, no checkpoint weights were changed, and SHIFT-MAP v2 was not implemented. Frozen benchmark data, canonical data, BM25 artifacts, dense artifacts, and Step 7 checkpoints were not overwritten.

## 3. Repository and frozen state

Repository: `C:\Users\rohan\SHIFT-ICD`

Branch: `main`

Pre-diagnostic Git history included:

```text
1c59266 docs: record corrected SHIFT-MAP v1 gates
b8d2f55 fix: audit SHIFT-MAP v1 evaluation aggregation
0350036 feat: train SHIFT-MAP v1 bi-encoder
```

Frozen base revision:

```text
FremyCompany/BioLORD-2023
167aab527b238a50ca65224e6319215d2ff4fc9f
```

## 4. Baseline gates

Before diagnostics:

```text
pytest: 56 passed
Ruff: All checks passed
mypy: Success: no issues found in 27 source files
package import: passed
canonical validation: passed
Track A benchmark validation: passed
BM25 v1.1 validation: passed
Dense v1/v1.1 validation: passed
SHIFT-MAP training contamination validation: passed
SHIFT-MAP artifact validation: passed
```

## 5. Fine-tuned reproduction

The canonical checkpoint was evaluated without retraining:

```text
artifacts/models/shift_map_v1/final_seed42/epoch_3
```

The reproduction produced exactly the persisted Step 7 output:

```text
Rows: 17,254
SHA-256: 3090ba9e290304d83bf5af0c3a20378f43fbf2d553000de9a85a70fa7cc563fa
```

Ordinary forward TEST metrics:

```text
Hit@1   = 0.019666
Hit@5   = 0.735807
Hit@10  = 0.847124
Hit@100 = 0.983673
MRR     = 0.353663
```

This confirms that the persisted fine-tuned evaluation is reproducible, but does not establish that its comparison baseline was valid.

## 6. Zero-shot pipeline parity failure

The original Step 7 evaluator constructs forward target text with an unfiltered dictionary over all canonical rows. The frozen dense pipeline instead constructs a direction-specific corpus.

Original Step 7 evaluator behavior:

```python
target_texts = {
    str(c): str(g)
    for c, g in zip(df.target_code, df.target_label, strict=True)
    if pd.notna(c) and pd.notna(g)
}
```

Frozen dense behavior:

```python
frame = rows[(rows.direction == direction) & rows.target_code.notna()]
```

Because the ICD-9-CM and ICD-10-CM namespaces contain overlapping code strings, the unfiltered Step 7 dictionary can overwrite forward target descriptions with reverse-direction descriptions.

Zero-shot BioLORD through the original Step 7 target construction:

```text
Hit@1   = 0.024119
Hit@5   = 0.725788
Hit@10  = 0.830798
Hit@100 = 0.971429
MRR     = 0.352961
```

This fails to reproduce the frozen dense baseline:

```text
Hit@1   = 0.717996
Hit@5   = 0.882004
Hit@10  = 0.921336
Hit@100 = 0.984045
MRR     = 0.792142
```

## 7. Corrected directional-corpus control

The same Step 7 metric/ranking function was then supplied with the frozen direction-specific forward corpus, without changing the model or metric code.

Target corpus size:

```text
17,513 ICD-10-CM targets
```

Recovered metrics:

```text
Hit@1   = 0.717996
Hit@5   = 0.882004
Hit@10  = 0.921336
Hit@100 = 0.984045
MRR     = 0.792142
```

These match the frozen BioLORD baseline exactly.

## 8. Root-cause classification

```text
F1 — EVALUATION BUG: PRIMARY, CONFIRMED
F2 — CHECKPOINT/INFERENCE PIPELINE BUG: SECONDARY, specifically target-corpus construction
F3 — EMBEDDING COLLAPSE: UNSUPPORTED
F4 — SCORE COMPRESSION: UNSUPPORTED
F5 — TARGET HUBNESS: UNSUPPORTED
F6 — OBJECTIVE MISALIGNMENT: UNSUPPORTED
F7 — EASY-NEGATIVE SATURATION: UNSUPPORTED
F8 — POSITIVE-SAMPLING UNDERCOVERAGE: UNSUPPORTED
F9 — DEV SELECTION-CRITERION MISALIGNMENT: NOT YET ISOLATED
F10 — OVERFITTING: UNSUPPORTED
F11 — DIRECTIONAL SPECIALIZATION: UNSUPPORTED
F12 — GENUINE FINE-TUNING DEGRADATION: UNSUPPORTED until corrected parity
```

## 9. Diagnostics intentionally not interpreted

The following were not treated as scientific findings because the highest-priority zero-shot parity gate failed:

- score distributions;
- rank histograms;
- rank displacement;
- target hubness;
- anisotropy;
- representation drift;
- source-versus-target drift;
- positive/negative similarity changes;
- gold margins;
- TRAIN/DEV/TEST model comparisons;
- loss saturation;
- explicit-negative contribution;
- alternative-sampling coverage;
- lexical regression;
- family-held-out paired deltas;
- backward-transfer paired deltas.

These require a corrected evaluation path first.

## 10. Tests added

Added a regression test for direction-scoped dense target corpora:

```text
test_dense_target_corpus_is_direction_scoped
```

Focused result:

```text
8 passed in 12.09s
```

## 11. Quality status after diagnostic change

The existing Step 7 tests passed before the diagnostic change. The new regression test passed. Full post-change gates remain to be run before any commit.

## 12. Recommended Step 7.2

The correct next milestone is:

```text
Step 7.2 = corrected SHIFT-MAP v1 rerun
```

It must:

1. use direction-specific target corpora in the evaluator;
2. run zero-shot BioLORD and fine-tuned BioLORD through the same corrected path;
3. reproduce the frozen zero-shot baseline before any interpretation;
4. rerun final fine-tuned evaluation with the corrected target corpus;
5. use a new experiment version and new TEST lock because the previous TEST result was generated through a defective evaluator;
6. preserve the original Step 7 result as an invalid-comparison artifact rather than deleting it;
7. avoid changing the training objective, learning rate, negative strategy, or model design simultaneously.

No retraining or Step 7.2 implementation was performed here.

## 13. Test-set reuse disclosure

The original Step 7 TEST result has already been observed and was generated with a defective evaluator. A corrected rerun cannot be presented as a pristine first-look TEST experiment. Future configuration decisions must be TRAIN/DEV-only, and the corrected evaluation must explicitly disclose that the redesign/rerun was motivated by the previously observed and invalid-comparison TEST result.

## 14. Final decision

```text
STEP_7_1_STATUS: BLOCKED_AT_PIPELINE_PARITY
PRIMARY_FAILURE: F1_EVALUATION_BUG
CORRECTED_V1_RERUN_REQUIRED: YES
SHIFT-MAP_V2: NOT_READY
```

The observed Step 7 Hit@1/MRR collapse is **not yet a valid scientific finding**. It was reproduced internally, but its zero-shot comparator was invalid under the original Step 7 evaluator. The next action is a corrected v1 rerun, not cross-encoder implementation or objective redesign.
