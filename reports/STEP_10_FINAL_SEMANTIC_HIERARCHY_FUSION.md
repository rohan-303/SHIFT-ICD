# STEP 10 FINAL — Semantic + Hierarchy Fusion

**Status:** `STEP10_FUSION_CLOSED` (scientific closeout; GitHub publication pending verification)

## 1. Motivation

Step 10 tested whether frozen SHIFT-MAP semantic rankings could be improved by a bounded H3 hierarchy residual without replacing the semantic anchor.

## 2. Nested-development design

R2 used only the frozen FUSION_TRAIN/FUSION_VAL source split for configuration selection. R3 refit normalization on full original TRAIN, trained independent seeds 17, 42, and 2026 for the frozen two epochs, froze checkpoints, then created the official-DEV lock before one-shot scoring. No TEST feature extraction, training, scoring, or lock occurred.

## 3. Bounded residual architecture

The frozen model was 4 → 16 → 1 with ReLU hidden activation, tanh output, H3-only features, and `s_fused = z_sem + 0.20*r_h`. Candidate membership remained frozen.

## 4. Inner split and R2 search

The selected configuration was λ=0.20, learning rate 3e-4, weight decay 1e-4, epoch 2. The corrected provenance record shows full-precision MRR, not NDCG@10, was the first differing decisive criterion after Hit@1 tied: selected MRR 0.7677976091295469 versus runner-up 0.7677849981529529.

## 5. Microscopic inner improvement

Against B0, the selected inner result changed Hit@1 by +0.0007062147, MRR by +0.0006203202, NDCG@10 by +0.0000932353, and neither primary structural @10 metric. The movement was 2 Top-1 improvements, 1 worsening, and 1,413 unchanged sources. The retained R2 package does not contain those three source-level rows, so their exact margins/residuals are **NOT COMPUTED**.

## 6. One-shot official DEV protocol

The lock preceded feature extraction and scoring. Official DEV ordinary population was 1,348 and P_COMPLEX was 67. Bootstrap used 10,000 paired source-level repetitions with seed 20260917.

## 7. Final seeds and canonical result

Seed 17 remained canonical by preregistration. Its official DEV deltas versus B0 were Hit@1 0.0000000000, MRR -0.0000715519, and NDCG@10 -0.0002173419. P_COMPLEX CSR@10 and ChoiceListRecall@10 were unchanged. The paired 95% CIs were [-0.0004438569, 0.0003014056] for MRR and [-0.0007608319, 0.0001442012] for NDCG@10.

## 8. Failure to replicate

The frozen classification is `OFFICIAL_DEV_FAILS_TO_REPLICATE`. The canonical model made no Top-1 changes on official DEV: 0 improved, 0 worsened, 1,348 unchanged. Its MRR movement was 14 improved, 25 worsened, and 1,309 unchanged.

## 9. Posthoc residual and rank diagnostics

From frozen ranked outputs and candidate scores, the canonical official-DEV mean absolute rank displacement was 0.846839763, median 0.000000000, p95 3.000000000, maximum 19; 0.551602374 of candidate positions were unchanged and the Top-1 unchanged fraction was 1.0.

Per-row residual distributions and actual score-delta distributions were not retained: only aggregate maxima were available (max absolute residual 0.3189316094; maximum absolute fusion delta 0.0637863219). Mean, SD, median, p5/p95 are therefore **NOT RECORDED**, not estimated.

The B0 semantic Top-1/Top-2 margin was computed posthoc from frozen retriever scores. MRR-improved and MRR-worsened source-group summaries are in `r3c_mrr_movement_characteristics.csv`; these are descriptive and not selection evidence.

## 10. Split-distribution diagnostics

The frozen R3 package does not retain the per-row FUSION_VAL or official-DEV H3 feature matrix, so H3 mean/SD/median/p5/p95 and standardized mean differences are **NOT COMPUTED**. No normalization adaptation or retraining was performed. Semantic-margin distribution is available for official DEV; an exact FUSION_VAL-versus-DEV standardized comparison is **NOT COMPUTED** because the required per-row R2 table was not retained.

## 11. Seed variability

Official DEV Hit@1 was 0.6965875371 for seed 17 and 0.6973293769 for both seeds 42 and 2026. Noncanonical improvements do not supersede the preregistered canonical seed-17 outcome. Seed variability is descriptive and does not reopen selection.

## 12. Evidence versus interpretation

**Observed:** the inner improvement was extremely small; canonical official DEV had zero Top-1 movement, negative MRR/NDCG deltas, unchanged structural @10 metrics, and a larger number of MRR-worsened than MRR-improved sources. The bounded correction preserved most candidate ordering.

**Plausible interpretation, not causal proof:** the H3 residual may contain little incremental information after semantic ranking, and the inner gain may have been concentrated in too few near-tie cases to be stable. The seed spread is also larger than the canonical effect. These interpretations are consistent with, but not proven by, the frozen evidence.

## 13. Limitations

This is a negative external confirmation on a benchmark with prior exposure in earlier project stages. Missing retained per-row H3/residual and R2 changed-source artifacts limit posthoc mechanism analysis. No additional scoring was used to fill those gaps.

## 14. TEST quarantine

TEST feature count: 0. TEST scoring count: 0. TEST training count: 0. TEST lock: ABSENT. The fusion model was not promoted.

## 15. Final Step 10 conclusion

Bounded H3 hierarchy residual fusion produced a microscopic improvement on a TRAIN-internal validation split, but the canonical preregistered model failed to reproduce the gain on one-shot official DEV. The method was therefore not promoted to TEST. This is a valid negative result.

## 16. Next research direction

Freeze `STEP 11-R1 — MAPPING CARDINALITY + STRUCTURED SET DECODER DESIGN AND PREREGISTRATION`. The next stage should model how many targets and what structured mapping form should be returned across SINGLE, ALTERNATIVE, COMBINATION, COMBINATION_WITH_ALTERNATIVES, and NO_MAP, rather than perturbing rank positions.
