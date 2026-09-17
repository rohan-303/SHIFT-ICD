# STEP 10-R3 — Final Fusion Seeds + One-Shot Official DEV Confirmation

**Status:** `STEP10_OFFICIAL_DEV_CONFIRMATION_FROZEN`

## 1. R2 provenance correction

R2 selected configuration and chronology were preserved. Full-precision comparison showed:

- Selected MRR: `0.7677976091295469`
- Runner-up MRR: `0.7677849981529529`
- First differing criterion: `MRR`
- NDCG@10 was not reached by the lexicographic selector.
- Selected configuration unchanged: `TRUE`
- Scientific selection affected: `FALSE`

Correction artifact: `artifacts/experiments/step10_fusion/r2_selection_provenance_correction.json`.

## 2. Final scaler and training population

The R1 policy is interpreted from the committed inner-scaler procedure: the inner scaler fit every source/candidate row in FUSION_TRAIN, while supervision separately filtered ordinary gold-present sources. The final analogue therefore fits the new scaler over every source/candidate row in the full original TRAIN candidate artifact.

- Full TRAIN scaler population: all frozen original TRAIN source groups, all mapping kinds, all 100 frozen candidates/source.
- Source count: `10,197`
- Candidate-row count: `1,019,700`
- Feature order: the four frozen H3 features.
- DEV rows used: `0`
- TEST rows used: `0`
- Final scaler SHA-256: `7556bce74060e501476276fd128fa1b6bfa9f5fd673b3294ed52d5c9822f7ae7`

Final ordinary TRAIN population:

- Ordinary eligible: `9,434`
- Gold-present supervised: `9,224`
- Gold-missing: `210`
- Included kinds: `SINGLE_EXACT`, `SINGLE_APPROXIMATE`, `ALTERNATIVE`
- Excluded from supervision: gold-missing ordinary, `NO_MAP`, `COMBINATION`, and `COMBINATION_WITH_ALTERNATIVES`.

## 3. Frozen final training

- Final-run protocol SHA-256: `fe38ebc98f05a6ddce2fce0789a3759cac20d41a79190c4f250093bd463d82a4`
- Configuration: λ `0.20`, LR `3e-4`, WD `1e-4`
- Architecture: `4 → 16 → 1`, ReLU then tanh
- Objective: `FULL_TOP100_SET_POSITIVE_LISTWISE`
- Batch size: `32`
- Epochs: exactly `2`
- Final seed manifest SHA-256: `ce7ef492bdb77af502b7879b2b98811475a0991935221279ffacc66bee585d45`

| Seed | Initialization SHA | Epoch 1 loss | Epoch 2 loss | Epoch-2 checkpoint SHA |
|---:|---|---:|---:|---|
| 17 | `4812895b491cee7b78e0861519b8c33205ae1c0008b861d1018b8c5068640109` | 4.4048010420 | 4.3888884020 | `312f11536156935f91b798cb8074072bc65ef3742f9f4ea8ea3429ae7c28c84a` |
| 42 | `7a22d07db2fbebb5e3a8ccf31fb1b974e162273aab543f1eb160412299f07266` | 4.3973513557 | 4.3835814783 | `c81ab94f05df0a768de358ac943588bf74f1b8417ad26cdcece084319bc52958` |
| 2026 | `dc5e69496f67dda4a3fdce4edd248e3c9d857fe762551c297a8f79ea041ba1f3` | 4.3943649477 | 4.3792525872 | `b20139e054cf8a044be547297055b29ba1b2b2b7d96e68576a689fdcc1471385` |

All seeds used independent initialization, exactly two epochs, finite gradients, candidate mutation count `0`, and bounded residuals. No R2 checkpoint was used to initialize final training.

## 4. Official DEV lock and chronology

All three final checkpoints and the final-seed manifest were frozen before official DEV access.

- Official-DEV lock SHA-256: `1d3851b73830ea398e3cdc9d34c930a55e5fc88a25c0e5eae9280d11e3cbcf9d`
- Lock timestamp: `2026-09-17T15:15:26.327002Z`
- First official DEV feature timestamp: `2026-09-17T15:15:26.336011Z`
- First official DEV score timestamp: `2026-09-17T15:15:41.330626Z`
- Lock-before-feature: `PASS`
- Lock-before-score: `PASS`
- DEV candidate SHA: `c7240c1f38dd87f54d63c97999cba71d91f6d0887d89bc2d4094f9ba38ad6a6f`
- DEV source count: `1,457`
- Feature-cache SHA: `4e0de46c8877bca03e7698d1da6be018d776c746c507a9a2811efabfcbda0bec`
- Official DEV was scored exactly once per frozen seed.

## 5. Official DEV populations

- Ordinary population count: `1,348`
- Ordinary source-set SHA: `1c052095f0db330e7a313c8db3b45e663fd96574967d0ece8f298bd0b68d97e4`
- Canonical P_COMPLEX count: `67`
- Canonical P_COMPLEX source-set SHA: `d5d29efa7cb00795b8ce9a2beb2567e6a54f42d7f40c1edd3856f76de83a4f23`
- Canonical structural definitions were used; no `HIGH_MAPPING_COMPLEXITY` substitution occurred.

## 6. Official DEV ordinary metrics

| Model | Hit@1 | Hit@5 | Hit@10 | Hit@25 | Hit@50 | Hit@100 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.6965875371 | 0.8620178042 | 0.9050445104 | 0.9421364985 | 0.9651335312 | 0.9777448071 | 0.7721301845 | 0.7738662523 |
| Seed 17 | 0.6965875371 | 0.8627596439 | 0.9043026706 | 0.9421364985 | 0.9643916914 | 0.9777448071 | 0.7720586326 | 0.7736489104 |
| Seed 42 | 0.6973293769 | 0.8627596439 | 0.9043026706 | 0.9421364985 | 0.9643916914 | 0.9777448071 | 0.7725553407 | 0.7741337651 |
| Seed 2026 | 0.6973293769 | 0.8627596439 | 0.9043026706 | 0.9421364985 | 0.9643916914 | 0.9777448071 | 0.7724474292 | 0.7741401129 |

## 7. Official DEV structural metrics

The three fusion seeds and B0 had identical primary structural metrics:

| Model | P_COMPLEX CSR@10 | P_COMPLEX ChoiceListRecall@10 | CSR@100 | Choice@100 |
|---|---:|---:|---:|---:|
| B0 | 0.1343283582 | 0.4720149254 | 0.4925373134 | 0.7406716418 |
| Seed 17 | 0.1343283582 | 0.4720149254 | 0.4925373134 | 0.7406716418 |
| Seed 42 | 0.1343283582 | 0.4720149254 | 0.4925373134 | 0.7406716418 |
| Seed 2026 | 0.1343283582 | 0.4720149254 | 0.4925373134 | 0.7406716418 |

Complete structural metrics for all populations and cutoffs are in `official_dev_structural.csv`.

## 8. Seed variability

| Metric | Mean | Population SD |
|---|---:|---:|
| Hit@1 | 0.6970820969 | 0.0003497066 |
| MRR | 0.7723538008 | 0.0002133142 |
| P_COMPLEX CSR@10 | 0.1343283582 | 0.0000000000 |
| P_COMPLEX ChoiceListRecall@10 | 0.4720149254 | 0.0000000000 |
| NDCG@10 | 0.7739742628 | 0.0002300735 |

Canonical seed remains `17`; seed selection was not changed based on DEV.

## 9. Canonical seed17 versus B0 bootstrap

Paired source-level bootstrap: 10,000 repetitions, seed `20260917`, percentile 95% CIs.

| Metric | Delta | 95% CI |
|---|---:|---:|
| Hit@1 | 0.0000000000 | [0.0000000000, 0.0000000000] |
| MRR | -0.0000715519 | [-0.0004438569, 0.0003014056] |
| P_COMPLEX CSR@10 | 0.0000000000 | [0.0000000000, 0.0000000000] |
| P_COMPLEX ChoiceListRecall@10 | 0.0000000000 | [0.0000000000, 0.0000000000] |
| NDCG@10 | -0.0002173419 | [-0.0007608319, 0.0001442012] |

## 10. Integrity and movement

- Candidate mutation count: `0`
- Hit@100 invariance: `PASS`
- Structural @100 invariance: `PASS`
- Seed17 Top-1 movement: improved `0`, worsened `0`, unchanged `1,348`
- Seed17 MRR movement: improved `14`, worsened `25`, unchanged `1,309`
- Missing-gold MRR contribution: `0.0`

## 11. Replication gate

The frozen gate required strict Hit@1 improvement and non-degradation in MRR, P_COMPLEX CSR@10, P_COMPLEX ChoiceListRecall@10, and NDCG@10. Seed17 had no Hit@1 improvement, lower MRR, and lower NDCG@10.

**Classification:** `OFFICIAL_DEV_FAILS_TO_REPLICATE`

The inner-to-external comparison was:

| Metric | Inner delta | Official DEV delta | Direction replicated |
|---|---:|---:|---|
| Hit@1 | +0.0007062147 | 0.0000000000 | YES |
| MRR | +0.0006203202 | -0.0000715519 | NO |
| P_COMPLEX CSR@10 | 0.0000000000 | 0.0000000000 | YES |
| P_COMPLEX ChoiceListRecall@10 | 0.0000000000 | 0.0000000000 | YES |
| NDCG@10 | +0.0000932353 | -0.0002173419 | NO |

The TRAIN-internal improvement was a microscopic, non-robust signal and did not replicate cleanly on official DEV.

## 12. TEST quarantine and next stage

- TEST feature extraction count: `0`
- TEST scoring count: `0`
- TEST training count: `0`
- No TEST lock was created.

Confirmation manifest: `artifacts/experiments/step10_fusion/official_dev_confirmation_manifest.json`.

Confirmation-manifest SHA-256: `015c07f4181f7ad20889dfbc596b4ce0d8f934ac2b76a5c090bb48f86871b7d0`.

Because the external confirmation failed to replicate the frozen gate, fusion is not promoted to TEST.

Conditional next milestone:

`STEP 10-R3C — FUSION CLOSEOUT + FAILURE ANALYSIS`
