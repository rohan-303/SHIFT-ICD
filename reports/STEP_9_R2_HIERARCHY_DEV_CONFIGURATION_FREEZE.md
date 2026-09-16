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
