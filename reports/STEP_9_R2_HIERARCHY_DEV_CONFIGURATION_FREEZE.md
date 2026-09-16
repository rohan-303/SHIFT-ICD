# STEP 9-R2 — Hierarchy-Aware DEV Ablations + Configuration Freeze

**Status:** `STEP9_CONFIG_FROZEN`

## 1. Scope and quarantine
R2 executed only the frozen corrected SHIFT-MAP DEV Top-100 candidate set. No Step 9 TEST features, scores, labels, training, or TEST lock were accessed. Final publication seeds were not trained.

## 2. H0 semantics resolution
H0 was resolved before DEV as `H0_BASELINE_ONLY`: the frozen B0 candidate ordering, with no hierarchy feature extraction or MLP training. Resolution SHA: `d8cbff63afaec36e953371c9e31efaac4393cfc1940b97a157cc848ed4d7b306`.

## 3. Interpretation-contract-v2
The original R1 interpretation artifact was preserved. Version 2 separates scientific outcomes from invalid experiments. SHA: `b18536b4c989cf6b04f3665915fc912e484cc9b7d07886561cdd6fbf55a426d9`.

## 4. Frozen hashes
- Feature contract: `f3e306a071bb5126a5b833f8f6f4ab4428a00749cadbbcdc2acd248ea0c4df06`
- Feature cache: `a73adbc94b36053fbcd9f73478731d37f294575f082f4a70c1c37f0e335a6718`
- Hierarchy manifest: `4a9e99149da5af5f77063a794f4cd8eb55e19c267831b92f9ed53e06e3b37a47`
- H0 resolution: `d8cbff63afaec36e953371c9e31efaac4393cfc1940b97a157cc848ed4d7b306`
- Interpretation v2: `b18536b4c989cf6b04f3665915fc912e484cc9b7d07886561cdd6fbf55a426d9`
- R2 run manifest: `6d4b2833011c9c7933dd0e40589569497ea7c0a8d9bec6969fcc05c230a1fd31`
- Configuration freeze: `4eb057b3f7badfc7c56062f4d921999449ccdb0e3481c8d4cff7e66a76398659`
- TRAIN scaler: `e5eb4b27b7debd1172e0db272018bb6a0049b08aa98679a9dedc7117185e4a24`
- Search space: `bb28e8415a7da78e2c96e98c4ee2526ac45f0eefdaa8b8669b6ce23b2ff96c10`
- Search protocol: `c0b70178458863a76ff94ed3ef5bd8e7744de4e574e3d2e8aec46d9f352e19c2`
- DEV selection rule: `c1e9ba21b385d6126f1df981839d23b55b2837c736161a4e009a76f9e461acf8`

## 5. Candidate and hierarchy verification
- TRAIN candidate SHA: matched `d90dcb937748f1c0b173416d6b8f23f1c97ca8e8caf8934efdf8120993c1cb10`
- DEV candidate SHA: matched `c7240c1f38dd87f54d63c97999cba71d91f6d0887d89bc2d4094f9ba38ad6a6f`
- TEST candidate SHA: matched `6ef38443a201b1d630148b30717d845b0785b1a4ed433eeab4b03d893b6bf1ce` (hash verification only)
- Hierarchy integrity: source coverage 14,567/14,567; target coverage 71,704/71,704; all recorded integrity error counts are zero.

## 6. Baselines

### B0 — frozen corrected SHIFT-MAP ordering

| Metric | DEV value |
|---|---:|
| Hit@1 | 0.696587537 |
| Hit@5 | 0.862017804 |
| Hit@10 | 0.905044510 |
| Hit@25 | 0.942136499 |
| Hit@50 | 0.965133531 |
| Hit@100 | 0.977744807 |
| MRR | 0.772350533 |
| NDCG@10 | 0.773866252 |

### B1 — frozen Step 8 MedCPT seed 17

| Metric | DEV value |
|---|---:|
| Hit@1 | 0.496290801 |
| Hit@5 | 0.694362018 |
| Hit@10 | 0.768545994 |
| Hit@25 | 0.836053412 |
| Hit@50 | 0.870919881 |
| Hit@100 | 0.901335312 |
| MRR | 0.586306407 |
| NDCG@10 | NOT_RECORDED |
`NDCG@10` is `NOT_RECORDED` in the frozen B1 DEV artifact and was not reconstructed.

## 7. Exact executed search design
The frozen protocol was executed as a full Cartesian grid over three trainable variants (H1, H3, H1+H3), two learning rates, and two weight decays. H0 was the non-trainable B0 control. Every trainable configuration used seed 17, source batch size 4, three epochs, gradient clipping 1.0, the frozen 8-dimensional input, TRAIN-only scaler, independent initialization, and SET_POSITIVE_LISTWISE.
Scientific configurations: **12 trainable configurations**, **36 epoch rows**, plus H0/B0.

## 8. Variant masks

| Variant | Active mask |
|---|---|
| H0_NO_NEW_HIERARCHY_FEATURES | `00000000` |
| H1_BASIC_ONTOLOGY_STRUCTURE | `11110000` |
| H3_CANDIDATE_SET_STRUCTURAL_CONTEXT | `00001111` |
| H1_PLUS_H3 | `11111111` |

## 9. Selected configuration
- Feature variant: `H3_CANDIDATE_SET_STRUCTURAL_CONTEXT`
- Active feature mask: `00001111`
- Learning rate: `0.0003`
- Weight decay: `0.0001`
- Selected epoch: `3`
- Selected checkpoint SHA-256: `acbf14dc9c6a45347004e95894bb9df5b1682d371708f3ff09f6b5ab3fb608df`
- Decisive criterion: ordinary Hit@1 under the frozen DEV lexicographic rule.

## 10. Selected DEV metrics and B0 deltas
- Hit@1: `0.264094955`
- Hit@5: `0.334569733`
- Hit@10: `0.436201780`
- Hit@25: `0.591988131`
- Hit@50: `0.785608309`
- Hit@100: `0.977744807`
- MRR: `0.323335309`
- NDCG@10: `0.317597241`
- P_COMPLEX_ChoiceListRecall@10: `0.542763158`
- P_COMPLEX_CompleteScenarioRetrieval@10: `0.447368421`
- Delta Hit@1 vs B0: `-0.432492582`
- Delta MRR vs B0: `-0.449015224`
- Delta P_COMPLEX_CompleteScenarioRetrieval@10 vs B0: `-0.368421053`
- Delta P_COMPLEX_ChoiceListRecall@10 vs B0: `-0.315789474`
- Delta NDCG@10 vs B0: `-0.456269011`
- DEV classification: `HIERARCHY_DEGRADES_RERANKING`.

## 11. Integrity and leakage
- Candidate mutation count: `0` for every row.
- Hit@100 invariance: PASS; selected `0.9777448071216617` equals B0 `0.9777448071216617`.
- Structural @100 invariance: PASS; selected ChoiceListRecall `0.9638157894736842` and CompleteScenarioRetrieval `0.9473684210526315` equal B0.
- Gold leakage regression: PASS.
- GEM leakage regression: PASS.
- Metadata leakage regression: PASS for all seven preregistered metadata mutations.
- TRAIN-only scaler audit: PASS.
- Checkpoint collision audit: PASS; 36 distinct nonempty epoch checkpoints.
- Training stability: PASS; all 36 rows valid with finite losses/gradients.

## 12. Selection reconstruction
Independent reconstruction reproduced `h3_candidate_set_structural_context_lr_0p0003_wd_0p0001` epoch `3` without a hard-coded winner. Artifact SHA: `b99d7dee29c60cd5d64cf491f5240e9b3250f5df337a723cf32bfcaad9c1ed5c`.

## 13. TEST quarantine
- Step 9 TEST feature-extraction count: `0`
- Step 9 TEST scoring count: `0`
- Step 9 TEST training count: `0`
- No Step 9 TEST lock was created.

**NO STEP 9 TEST FEATURES OR SCORES WERE ACCESSED DURING CONFIGURATION SELECTION.**

## 14. Final-seed policy and next milestone
R1/R2 preregistered only development seed 17; no final publication seeds were specified. Status: `BLOCKED_STEP9_FINAL_SEED_POLICY_UNSPECIFIED`. No final seeds were trained.

Next milestone: `STEP 9-R3 — FINAL HIERARCHY SEEDS + TEST LOCK + CONFIRMATORY STRUCTURAL RERANKING EVALUATION`, gated on resolving the final-seed policy.

## 15. POST-FREEZE PROVENANCE CORRECTION — R2C

This administrative/provenance correction did not rerun hierarchy training, rerun the R2 DEV grid, alter the selected H3 configuration, reopen Step 8 tuning, or access Step 9 TEST.

### Final-seed policy

A pre-existing project-wide outcome-independent convention was found and frozen as `S1_EXISTING_PROJECT_WIDE_CONVENTION`: final seeds `[17, 42, 2026]`, canonical seed `17`. Evidence includes the Step 8 final-seed protocol (`artifacts/experiments/step8_full_universe/final_seed_run_protocol.json`, SHA `ff1a60caef9c6d600a90a47918ab07a5e0ee14338e0e5ed11c2ca4388db3d158`, commit `fe927241e1a5ed79b63fb5b12bea05ff3a3558a9`), the Step 8 configuration freeze (SHA `b82c0cba4b672726f0c3911fac00d5abe36263b0c20a36e0d261f3c14f5b4064`), and the corrected SHIFT-MAP seed artifacts.

The frozen policy is recorded at `artifacts/experiments/step9_hierarchy/final_seed_policy.json`, SHA `f2da29bec862a821820c0974cee467fac37fb9db961c809d1739e56c5154c2e3`. It was frozen before Step 9 TEST and before final Step 9 seed training. **NO STEP 9 TEST RESULT WAS AVAILABLE WHEN THIS POLICY WAS FROZEN.** **SEEDS WERE NOT CHOSEN USING STEP 9 DEV PERFORMANCE.**

### B1 provenance correction

The original B1 value (`Hit@100 = 0.901335312`) came from `artifacts/experiments/dense_full_universe_v2/remote_sync_final/gpu1/results/dense_full_universe_v2/medcpt/dev_metrics.json`, SHA `1cf8ec359036e1bda2b54dadd927d2fcb7d5505bdfe5cb909bb494739149cfd5`. It contained 1,457 source rows with 100 ranked codes per row, but no frozen candidate hash, checkpoint SHA, or scoring timestamp. Direct comparison against `artifacts/candidates/shift_map_full_universe_v2/forward_dev_k100.jsonl.gz` found zero exact candidate-membership matches across all 1,457 sources; the first audited source overlapped by only 42/100 codes. The artifact was zero-shot MedCPT Query Encoder output, not the final Step 8 MedCPT Cross-Encoder reranking artifact.

The issue is classified exactly as `B1D_WRONG_CANDIDATE_SET`. The authoritative corrected Step 8 final-seed DEV artifact is `artifacts/remote/step8_r3_20260915T203248Z_a2333bc9/final_seed_run/final_seed_dev.csv`, SHA `54337b03914e1984b0b1516ba74c666e0b20bacf4108edde0b996e49f342710a`, with canonical seed-17 checkpoint SHA `1614dfb9377048e2b77b46fd2b82433f842ca063420fe0ad72f2c3c818ece66b` and execution-manifest SHA `db9ba5b65647594e7fdf7eaec3afbc2676b55e6e041c3ea4fe3891709437292a`. It uses the frozen DEV candidate SHA `c7240c1f38dd87f54d63c97999cba71d91f6d0887d89bc2d4094f9ba38ad6a6f`, 1,457 sources, 145,700 rows, and 100 candidates per source.

Corrected canonical B1 seed-17 epoch-3 metrics are Hit@1 `0.652077151`, MRR `0.738264110`, Hit@10 `0.888724036`, and Hit@100 `0.977744807`. Candidate mutation count is `0`; exact membership matches are 1,457/1,457; structural @100 remains invariant. This correction is descriptive only and **does not affect R2 hierarchy selection**. Full machine-readable details are in `artifacts/experiments/step9_hierarchy/b1_baseline_provenance.json`, SHA `02131885cd61f172fc82e27dbf3227c4ba412d6d93042a6324b2910b264e3085`.

### R3 preregistration boundary

The primary future TEST comparison is the selected hierarchy-aware H3 model versus B0 canonical frozen corrected SHIFT-MAP ordering. B1 MedCPT seed 17 is secondary/descriptive only. The primary metric family is ordinary Hit@1, ordinary MRR, P_COMPLEX CompleteScenarioRetrieval@10, P_COMPLEX ChoiceListRecall@10, and ordinary NDCG@10.

The future paired bootstrap plan is frozen at 10,000 source-level paired-with-replacement repetitions, seed `20260915`, percentile 95% confidence intervals, on the identical TEST population. It is not executed now. The R3 addendum is `artifacts/experiments/step9_hierarchy/r3_preregistration_addendum.json`, SHA `997fac62612b1482e68a1bd354754a32011ca94f74ac3b6382ce8dbcf3b0e519`.

Step 9 counts during R2C remain: new hierarchy DEV training `0`; Step 9 TEST feature extraction `0`; Step 9 TEST scoring `0`; Step 9 TEST training `0`. No Step 9 TEST lock was created.
