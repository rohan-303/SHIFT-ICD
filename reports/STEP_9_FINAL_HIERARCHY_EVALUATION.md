# STEP 9-R3 — Final Hierarchy-Aware Reranking Evaluation

**Status:** `STEP9_HIERARCHY_EVALUATION_FROZEN`

## 1. Step 8 negative motivation

Step 8 established that corrected SHIFT-MAP outperformed MedCPT on the primary ranking metrics while Top-100 membership remained invariant. Step 9 therefore tested a separate hypothesis: whether ontology-derived structural information could improve ranking within the frozen corrected SHIFT-MAP candidate set.

## 2. R1 hierarchy protocol

R1 froze the terminology-only ICD-9/ICD-10 prefix hierarchy, the eight-feature contract, the corrected SHIFT-MAP Top-100 candidate sets, TRAIN-only normalization, the source-balanced variable-length multi-positive objective, and the Step 9 TEST lockout. GEM mappings, labels, raw learned code text, retriever scores, and semantic ancestor features were excluded.

## 3. R2 configuration search

R2 executed the preregistered DEV-only grid and froze `H3_CANDIDATE_SET_STRUCTURAL_CONTEXT`, mask `00001111`, learning rate `3e-4`, weight decay `1e-4`, epoch 3, and model family `SHALLOW_MLP_8_TO_32_TO_1_RELU`. R2 classified the hierarchy-only result as `HIERARCHY_DEGRADES_RERANKING`.

## 4. Hierarchy-only DEV degradation

B0 DEV Hit@1 was `0.696588`; the selected H3 DEV Hit@1 was `0.264095`. DEV Hit@100 remained `0.977745`, confirming membership invariance rather than candidate loss.

## 5. R2C seed preregistration

The project-wide convention `[17, 42, 2026]` with canonical seed `17` was frozen before Step 9 TEST and before final Step 9 seed training. Seeds were not chosen using Step 9 DEV performance. Policy SHA: `f2da29bec862a821820c0974cee467fac37fb9db961c809d1739e56c5154c2e3`.

## 6. B1 provenance correction

The earlier B1 DEV value came from a zero-shot dense full-universe artifact with the wrong candidate set. It was classified `B1D_WRONG_CANDIDATE_SET`. The corrected Step 8 final-seed B1 artifact uses the frozen candidate set and restores Hit@100 invariance. This correction was descriptive only and did not affect R2 selection. Provenance SHA: `02131885cd61f172fc82e27dbf3227c4ba412d6d93042a6324b2910b264e3085`.

## 7. Final-seed training

Three seeds were initialized independently from the frozen H3 model initialization policy and trained for three epochs using TRAIN only. Each selected epoch-3 checkpoint was readable, finite, seed-isolated, and had a distinct SHA. Final-seed manifest SHA: `343a03d32ce39f77de4aa326869fcf01d31662997417c11c4e97da84f82bb4e8`.

## 8. TEST lock

The Step 9 TEST lock was created only after all three final checkpoints and the final-seed manifest were frozen. Lock SHA: `bde5cc2251be9644c691ef4e18df1569e6e46ffe783172557bfc387759a131ec`. Timestamp: `2026-09-16T17:40:37.384382Z`.

The first TEST feature extraction occurred after the lock, followed by TEST scoring. TEST training count was zero.

## 9. Canonical TEST result

The canonical seed is 17 by preregistration, not by performance.

| Metric | B0 SHIFT-MAP | Hierarchy seed 17 | Delta |
|---|---:|---:|---:|
| Hit@1 | 0.703154 | 0.273469 | -0.429685 |
| MRR | 0.776287 | 0.330882 | -0.445405 |
| P_COMPLEX CompleteScenarioRetrieval@10 | 0.679487 | 0.448718 | -0.230769 |
| P_COMPLEX ChoiceListRecall@10 | 0.738034 | 0.548077 | -0.189957 |
| NDCG@10 | 0.775580 | 0.321901 | -0.453679 |
| Hit@100 | 0.978108 | 0.978108 | 0.000000 |

## 10. Paired bootstrap versus B0

The preregistered plan used 10,000 paired source-level bootstrap repetitions with seed `20260915` and percentile 95% confidence intervals.

| Metric | Delta | 95% CI |
|---|---:|---:|
| Hit@1 | -0.429685 | [-0.449351, -0.409647] |
| MRR | -0.445188 | [-0.463266, -0.427610] |
| CSR@10 | -0.230769 | [-0.333333, -0.128205] |
| ChoiceListRecall@10 | -0.189957 | [-0.282906, -0.103632] |
| NDCG@10 | -0.453679 | [-0.471468, -0.436306] |

The bootstrap table is machine-readable at `reports/tables/step9_hierarchy/b0_vs_hierarchy_bootstrap.csv`.

## 11. Structural TEST results

All three hierarchy seeds preserved the frozen candidate membership. Hit@100 was `0.978108` for B0 and every hierarchy seed. P_COMPLEX structural @100 was invariant at ChoiceListRecall `0.921795` and CompleteScenarioRetrieval `0.871795`.

## 12. Seed variability

Selected-seed TEST means and population standard deviations:

| Metric | Mean | SD |
|---|---:|---:|
| Hit@1 | 0.271985 | 0.001092 |
| MRR | 0.329184 | 0.001343 |
| NDCG@10 | 0.319761 | 0.001654 |
| P_COMPLEX CSR@10 | 0.448718 | 0.000000 |
| P_COMPLEX ChoiceListRecall@10 | 0.548077 | 0.000000 |

## 13. Candidate invariance

The TEST population contained 2,913 sources and 291,300 candidate rows. Each source had 100 candidates. Candidate mutation count was zero for seeds 17, 42, and 2026. All hierarchy Hit@100 values equaled B0 membership coverage.

## 14. Movement and failure analysis

For canonical seed 17 versus B0:

- Top-1 improved: `42`
- Top-1 worsened: `1,210`
- Top-1 unchanged: `1,661`
- MRR improved: `164`
- MRR worsened: `1,720`
- MRR unchanged: `1,029`

Failure categories over ordinary TEST sources:

- A — B0 correct, hierarchy wrong: `1,210`
- B — B0 wrong, hierarchy correct: `42`
- C — both correct: `728`
- D — both wrong, gold inside Top-100: `788`
- E — gold absent from Top-100: `145`

For P_COMPLEX earliest complete-scenario rank, hierarchy improved `144`, worsened `1,726`, and was unchanged for `1,043` sources. Full tables are under `reports/tables/step9_hierarchy/`.

## 15. Feature diagnostics and distribution shift

The largest first-layer absolute mean weight was for `candidate_same_parent_fraction` (`1.267082`), followed by `candidate_same_family_fraction` (`0.229345`), `candidate_same_root_fraction` (`0.213613`), and `candidate_shared_ancestor_fraction` (`0.166808`). These are descriptive weight magnitudes and are not causal importance claims.

TRAIN-to-TEST standardized mean differences were:

- candidate_same_parent_fraction: `-0.0382`
- candidate_same_family_fraction: `0.1024`
- candidate_shared_ancestor_fraction: `0.0617`
- candidate_same_root_fraction: `0.0617`

No adaptation or tuning was performed. The TRAIN correlation matrix is recorded at `reports/tables/step9_hierarchy/feature_correlation_train.csv`.

## 16. Limitations

This result evaluates a hierarchy-only replacement reranker over a fixed lexical candidate set. It does not test hierarchy as an auxiliary signal combined with semantic retrieval. The negative result does not establish that ontology structure is universally unhelpful; it establishes that this frozen H3-only replacement ranking configuration degraded the preregistered TEST comparison.

Step 8 had prior TEST exposure, which is disclosed in the R1/R2C artifacts. Step 9 TEST was accessed only after the Step 9 lock. No post-TEST tuning, feature changes, seed changes, or retraining occurred.

## 17. Final scientific classification

`HIERARCHY_DEGRADES_RERANKING`

The experiment is valid: all candidate and structural invariants passed, all primary paired-bootstrap intervals were strictly negative, and the negative result was preserved without tuning it away.

### Frozen artifact index

- Final run protocol: `artifacts/experiments/step9_hierarchy/final_seed_run_protocol.json`
- Final-seed manifest: `artifacts/experiments/step9_hierarchy/final_seed_manifest.json`
- TEST lock: `artifacts/experiments/step9_hierarchy/test_lock.json`
- TEST result summary: `artifacts/experiments/step9_hierarchy/r3_test_results.json`
- Next-stage decision: `artifacts/experiments/step9_hierarchy/next_stage_decision.json`
- Canonical scored artifacts: `artifacts/experiments/step9_hierarchy/canonical_{train,dev,test}_scored.jsonl.gz`

**Final status:** `STEP9_HIERARCHY_EVALUATION_FROZEN`

## POST-FREEZE STRUCTURAL POPULATION AUDIT

R3A audited the frozen evaluation without retraining, model selection, TEST tuning, checkpoint changes, candidate changes, or new model inference.

### Detected mismatch and root cause

The authoritative forward TEST evaluator defines `P_COMPLEX` as the union of `P_COMBINATION` and `P_COMBINATION_WITH_ALTERNATIVES` (with `MULTI_SCENARIO` included by contract but absent in this split). The canonical population is 133 sources: 64 combinations plus 69 combinations-with-alternatives.

The original R3 structural evaluator instead selected sources containing the `HIGH_MAPPING_COMPLEXITY` difficulty slice. That produced 78 sources: 52 alternatives, 22 combination-with-alternatives, and 4 combinations. The original R3 set intersects the canonical P_COMPLEX set in 26 sources, leaving 107 canonical-only and 52 R3-only sources.

The issue is classified as `SP4_MAPPING_SUBTYPE_FILTER` and `SP5_EVALUATOR_IMPLEMENTATION_DRIFT`. It was not conditioning on gold presence or complete-scenario retrievability.

### Superseded and authoritative metrics

The original R3 `n=78` structural table and structural bootstrap are preserved and superseded for canonical P_COMPLEX claims. The corrected canonical B0 @100 coverage is:

- ChoiceListRecall@100: `0.7323308271`
- CompleteScenarioRetrieval@100: `0.4887218045`

These reproduce the frozen retrieval-stage coverage. Reorder-only hierarchy scoring preserves both values at @100 for seed 17; candidate mutation count remains zero.

Corrected canonical P_COMPLEX @10 results:

| System | ChoiceListRecall@10 | CompleteScenarioRetrieval@10 |
|---|---:|---:|
| B0 | 0.459148 | 0.157895 |
| Hierarchy seed 17 | 0.306391 | 0.022556 |

Corrected paired bootstrap over the canonical `n=133` population, using the unchanged 10,000 repetitions and seed `20260915`:

- ChoiceListRecall@10 delta: `-0.152757`, 95% CI `[-0.193988, -0.111779]`
- CompleteScenarioRetrieval@10 delta: `-0.135338`, 95% CI `[-0.195489, -0.075188]`

### Ordinary metric correction

The authoritative ordinary evaluator assigns MRR `0.0` when no gold target is retrieved. R3’s local helper used `1/101` for that case. Hit@K and NDCG were unchanged; ordinary MRR is corrected from B0 `0.776287` to `0.776070` and from hierarchy seed 17 `0.330882` to `0.330882` (the hierarchy value happened not to change at displayed precision). Corrected ordinary bootstrap MRR delta is `-0.445188`, 95% CI `[-0.463266, -0.427610]`.

### Classification and provenance

After reapplying interpretation-contract-v2 to the corrected primary metrics, the classification remains `HIERARCHY_DEGRADES_RERANKING`. The original classification did not change. The original R3 freeze manifest and outputs remain immutable; correction evidence is chained through:

- `artifacts/experiments/step9_hierarchy/structural_population_audit.json`
- `artifacts/experiments/step9_hierarchy/r3_evaluation_correction_manifest.json`
- `reports/tables/step9_hierarchy/r3a_*`

No model retraining or model selection occurred during R3A.

**R3A status:** `STEP9_R3_STRUCTURAL_AUDIT_CLOSED`
