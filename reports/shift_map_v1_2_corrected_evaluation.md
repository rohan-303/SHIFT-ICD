# SHIFT-MAP v1.2 corrected evaluation and replay

## 1. Evaluator bug summary

Step 7 used an invalid evaluator: target descriptions were aggregated from all canonical mapping rows, without filtering by direction. Overlapping code strings allowed reverse-direction descriptions to overwrite forward descriptions. Step 7.1 confirmed this as F1/F2.

The corrected evaluator is version **2.0**. Benchmark semantics remain v1.0 and canonical schema remains v1.0.

## 2. Why original Step 7 comparison was invalid

Frozen BioLORD failed the original evaluator path (`Hit@1=0.024119`, `MRR=0.352961`) but exactly reproduced dense_v1.1 after direction-specific corpus correction. Therefore the original apparent fine-tuning collapse was an evaluator artifact and is invalid for model comparison. Original Step 7 outputs remain preserved and are labeled invalid-comparison historical artifacts.

## 3. Direction-specific corpus fix

`shift_icd.dense.corpus.build_target_corpus(rows, direction)` is now authoritative. It filters before description aggregation, validates code-family syntax, sorts deterministically, records CMS FY2018, and hashes the ordered code/description payload.

| direction | target terminology | concepts | corpus hash |
|---|---:|---:|---|
| ICD9CM_TO_ICD10CM | ICD-10-CM | 17,513 | `a394fe6a0055b13e2df4eaeb4ac4b739e7d8d54277adb02f9bdecff67ad80a3a` |
| ICD10CM_TO_ICD9CM | ICD-9-CM | 11,690 | `f04b166e3ffbb04cfd47c7dba7c934d22a4482c8cc20e7f786ff7eb002e66a17` |

Both hashes and code orders match the frozen dense caches.

## 4. Training-pipeline direction audit

The training population, target cache, negative miner, and training batch lookup are forward-scoped. The immutable manifests record 9,434 TRAIN sources, 15,588 positive relations, no DEV/TEST mining, and zero persisted gold collisions. The corrected audit reports all five negative strategies as forward-only, with zero non-ICD-10 codes and zero missing descriptions.

Historical batch-level literal target strings were not persisted; positive/negative text integrity is therefore a reconstructed audit from immutable codes, manifests, frozen cache identity, and the training lookup implementation. No evidence indicates training was affected by the evaluation bug. No model was retrained and no optimizer was created in Step 7.2.

## 5. Positive and negative description integrity

Positive relations audited: **15,588**; reconstructed correct matches: **15,588**; mismatches: **0**; missing descriptions: **0**; cross-direction collisions: **0**. RANDOM, LEXICAL, DENSE, SAME_FAMILY, and MIXED negative sets each report zero per-source gold collisions, zero missing descriptions, and zero non-ICD-10 codes. Historical negative text files were not separately persisted; descriptions are reconstructed from the authoritative forward corpus.

## 6. Zero-shot parity

Corrected forward TEST BioLORD exactly reproduced dense_v1.1:

```text
Hit@1   0.717996
Hit@5   0.882004
Hit@10  0.921336
Hit@25  0.957328
Hit@50  0.974397
Hit@100 0.984045
MRR     0.792142
```

Corrected forward DEV also reproduced the frozen forward pipeline within normal deterministic evaluation tolerance. Corrected backward evaluation used the 11,690-concept ICD-9-CM corpus and produced 14,341 rows; corpus identity and direction gates passed. The historical dense backward reference is preserved; no new backward model was trained.

## 7–12. Corrected DEV replay and selection

All **36 retained requested checkpoints** were replayed through the corrected evaluator: P0/P1/P2 policy runs, L1/L2, N1/N2/N3/N5, 5e-6/1e-5/2e-5, and final seeds 17/42/2026 across epochs 1–3. TEST was not used.

| decision | corrected winner |
|---|---|
| positive policy | P2 |
| loss | **L2** |
| negative strategy | N1_RANDOM |
| learning rate | 2e-5 |
| final-seed epoch | epoch 3 for seeds 17, 42, 2026 |
| corrected canonical seed | seed 42 |

The loss result is the critical change: the best retained L2 DEV checkpoint had a higher first selection key (`Hit@100`) than the best retained L1 checkpoint. Thus the original P2/L1/2e-5 final-seed configuration is **not fully validated** by corrected DEV replay.

## 13–15. TEST status and exposure

No new Step 7.2 TEST lock was created. TEST had already been observed through the defective evaluator. The existing final seed-17/42/2026 epoch-3 checkpoints were replayed diagnostically through the corrected evaluator, but their results are not definitive because corrected DEV selected L2 while these checkpoints use L1. This is historical BUG-FIX RE-EVALUATION, not an untouched first-look test. No hyperparameter or training change was made from the defective TEST result before correction.

## 16. Corrected forward TEST diagnostic results

Each seed has 2,695 ordinary answerable examples. Results below are diagnostic-only:

| seed | Hit@1 | Hit@10 | Hit@100 | MRR |
|---:|---:|---:|---:|---:|
| 17 | 0.747310 | 0.944341 | 0.993692 | 0.817762 |
| 42 | 0.738404 | 0.944712 | 0.992950 | 0.812901 |
| 2026 | 0.746568 | 0.944712 | 0.992950 | 0.817766 |
| mean | 0.744094 | 0.944589 | 0.993197 | 0.816143 |
| SD | 0.004941 | 0.000214 | 0.000428 | 0.002808 |

Canonical diagnostic seed 42 is the DEV-selected seed, but not a definitive final experiment under the corrected configuration.

## 17. Zero-shot comparison and paired bootstrap

Canonical seed 42 versus corrected zero-shot BioLORD, paired on 2,695 ordinary answerable examples, 5,000 bootstrap samples, seed 2026:

| metric | zero-shot | fine-tuned | delta | 95% CI |
|---|---:|---:|---:|---:|
| Hit@1 | 0.717996 | 0.738404 | +0.020408 | [+0.010761, +0.030427] |
| Hit@10 | 0.921336 | 0.944712 | +0.023377 | [+0.016698, +0.030065] |
| Hit@100 | 0.984045 | 0.992950 | +0.008905 | [+0.005566, +0.012616] |
| MRR | 0.792142 | 0.812901 | +0.020759 | [+0.014676, +0.026974] |

These positive diagnostic deltas do not override the corrected DEV loss-selection invalidation.

## 18. BM25 and Qwen comparison

Frozen BM25 and Qwen3-Embedding-0.6B artifacts were not rerun or retuned. Their persisted forward TEST references are available in `dense_v1_1/frozen_metrics.json`; corrected SHIFT-MAP diagnostic values are in `forward_test_metrics.json`. Direct definitive ranking claims are deferred to Step 7.3 because the selected loss differs.

## 19–22. Slices, mapping kinds, alternatives, and combinations

Corrected diagnostic lexical and mapping-kind summaries are written to `lexical_slices.json`, `mapping_kind.json`, `alternative_size.json`, and `combination_metrics.json`, with CSV exports where available. Combination cases remain evaluation-only because complex mappings were excluded from gradient supervision. Per-slice paired confidence intervals were not promoted to a definitive claim because the final configuration is invalidated.

## 23. Family-held-out generalization

A corrected fine-tuned family-held-out replay was not executed because the retained final replay rows expose stratified TEST membership and the corrected DEV decision invalidated the original final configuration. Frozen family-held-out references remain available in the baseline artifacts. This is explicitly **NOT_AVAILABLE**, not a negative result.

## 24. Backward transfer

The forward-trained checkpoints were evaluated diagnostically on the corrected ICD-9-CM backward corpus and are labeled **FORWARD-TRAINED → BACKWARD TRANSFER**. No backward supervision occurred. Family-held-out backward transfer is not available from the retained corrected replay namespace.

## 25. NO_MAP diagnostics

The corrected rows retain max/top score and mean Top-5 similarity fields. No threshold, classifier, calibration, or clinical interpretation was introduced. The diagnostic artifact records the available NO_MAP population and explicitly avoids an AUROC claim.

## 26. Rank distribution and complementarity

Hit@1/10/100 complementarity is generated for the paired canonical population. Exact target ranks above 100 and top-1 target identity were not persisted in the historical row schema, so exact rank buckets and hubness concentration are marked unavailable rather than reconstructed as facts.

## 27. Representation drift

Embedding matrices were not persisted by the original Step 7 replay outputs. A new encoding run would be required for exact target/source drift statistics. `representation_drift.json` records this as unavailable; no unsupported drift claim is made.

## 28. Final scientific interpretation

Outcome classification: **R7 — CONFIGURATION SELECTION INVALIDATED**. The evaluation bug is fixed and the training path appears direction-correct, but corrected DEV replay selects L2 while the existing final seeds were trained with L1. The corrected diagnostic TEST results suggest the retained L1 model can improve the frozen baseline on this population, but they cannot establish the intended final SHIFT-MAP v1 result.

The original catastrophic-degradation claim is invalidated. The corrected evidence supports neither v2 nor abandonment of the retriever yet.

## 29. Recommended next milestone

**PATH C — Step 7.3 controlled retraining.** Rerun only the corrected preregistered design selected by corrected DEV (P2/L2/N1_RANDOM/2e-5, subject to a fresh explicit protocol decision), then select on corrected DEV and reserve a newly declared evaluation procedure. Do not implement SHIFT-MAP v2 in Step 7.2.

## Artifacts, tests, and integrity

Step 7.2 artifacts are under `artifacts/experiments/shift_map_v1_2/`; tables are under `reports/tables/shift_map_v1_2/`. The original `shift_map_v1` artifacts and test lock were not overwritten. No model binaries were staged. The complete post-change quality gates and commit are the remaining finalization steps.
