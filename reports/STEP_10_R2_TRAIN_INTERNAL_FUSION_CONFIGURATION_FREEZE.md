# STEP 10-R2 — Train-Internal Fusion Ablations + Configuration Freeze

**Status:** `STEP10_CONFIG_FROZEN`

## Protocol and provenance

- Starting HEAD: `cd57fe297bdbf2623832b52c0a55fc16967d4039`
- R2 run manifest: `artifacts/experiments/step10_fusion/r2_run_manifest.json`
- R2 run-manifest SHA-256: `00e4f2fdf8843c6ba816d2c88ea1ff8e95128e8edccf3141f65769ada777eee2`
- Configuration grid SHA-256: `9be491e45655e5a7182c92c90d4617f3d1cb62ebcfeb287a91926ca3eaf4f514`
- R1 artifact hash manifest verified against the frozen R1 SHA: `45a6e70e05ddf78ac7e5ade680889a1ee959d49979fe4755a7ccc76e7e7002c8`
- Official DEV and TEST candidate files were not opened.

The frozen semantic anchor was the SHIFT-MAP `retriever_score`, normalized per source with population standard deviation (`ddof=0`, `epsilon=1e-8`). H3 features remained exactly the four R1 features. The residual model was `4 → 16 → 1`, ReLU hidden activation and tanh output. The objective was full-Top-100 source-level set-positive listwise loss.

## Inner populations

| Population | FUSION_TRAIN | FUSION_VAL |
|---|---:|---:|
| Total sources | 8,666 | 1,531 |
| Ordinary answerable | 8,018 | 1,416 |
| Ordinary supervised gold-present | 7,840 | 1,384 |
| Ordinary gold-missing | 178 | 32 |
| P_COMBINATION | 190 | 35 |
| P_COMBINATION_WITH_ALTERNATIVES | 207 | 36 |
| Canonical P_COMPLEX | 397 | 71 |

- FUSION_VAL ordinary source-set SHA-256: `e40bd1d1985fe9e888596bc2322da7e19add0f75403478dbded1e153fbc39aa6`
- FUSION_VAL P_COMPLEX source-set SHA-256: `c7bb59ea4e8077019b4cf0dfa215f7e78fe37314ef339cbbe000a48a9b96b27b`
- Canonical P_COMPLEX used the committed mapping contract: `COMBINATION ∪ COMBINATION_WITH_ALTERNATIVES ∪ MULTI_SCENARIO`; it was not conditioned on gold presence, retrievability, or the historical `HIGH_MAPPING_COMPLEXITY` slice.
- FUSION_TRAIN/FUSION_VAL overlap: `0`.

## B0 FUSION_VAL baseline

B0 was the non-trainable λ=0 semantic-only control and participated in global selection.

| Metric | B0 |
|---|---:|
| Hit@1 | 0.6899717514 |
| Hit@5 | 0.8644067797 |
| Hit@10 | 0.9004237288 |
| Hit@25 | 0.9442090395 |
| Hit@50 | 0.9661016949 |
| Hit@100 | 0.9774011299 |
| MRR | 0.7671772890 |
| NDCG@10 | 0.7693795745 |
| P_COMPLEX CSR@10 | 0.1690140845 |
| P_COMPLEX ChoiceListRecall@10 | 0.4348591549 |

The corrected MRR contract contributed `0.0` when no valid gold candidate appeared; no `1/101` contribution was used.

Normalized semantic ordering disagreements: `0`; Top-1 disagreements: `0`.

## Complete ablation

- Trainable configurations: `12`
- Trainable scientific epoch rows: `36`
- Total master rows including B0: `37`
- Every trainable configuration completed epochs 1–3.
- Every row was valid.
- Training losses and gradients were finite.
- Initialization was independently created for each configuration using development seed `17`.
- Residual bound passed for every configuration/epoch.
- Maximum selected-model residual: `0.2874308825`.
- Maximum selected-model fusion perturbation: `0.0574861765 ≤ λ=0.20`.
- Candidate mutation count: `0`.
- Hit@100 invariance: `PASS`.
- Structural @100 invariance: `PASS`.

## Global selection

The deterministic selector compared B0 plus one representative epoch from each trainable configuration using the frozen lexicographic rule:

`Hit@1 → MRR → P_COMPLEX CSR@10 → P_COMPLEX ChoiceListRecall@10 → NDCG@10 → smaller λ → configuration ID → earlier epoch`.

### Selected configuration

- Model type: `TRAINABLE_FUSION`
- Configuration: `fusion_lambda_0p20_lr_3e-04_wd_1e-04`
- λ: `0.20`
- Learning rate: `3e-4`
- Weight decay: `1e-4`
- Selected epoch: `2`
- Selected checkpoint SHA-256: `d057e7c0653b633b225b8251a1a9b02aad59410416b84a5cc7f4b2c1bc7314a2`
- Runner-up: `fusion_lambda_0p20_lr_3e-04_wd_1e-03`, epoch 2
- Decisive criterion: MRR after Hit@1 tied; lower-priority structural metrics tied; NDCG@10 selected the lower-weight-decay configuration.

| Metric | Selected | Δ vs B0 |
|---|---:|---:|
| Hit@1 | 0.6906779661 | +0.0007062147 |
| MRR | 0.7677976091 | +0.0006203202 |
| P_COMPLEX CSR@10 | 0.1690140845 | 0.0000000000 |
| P_COMPLEX ChoiceListRecall@10 | 0.4348591549 | 0.0000000000 |
| NDCG@10 | 0.7694728098 | +0.0000932353 |
| Hit@100 | 0.9774011299 | 0.0000000000 |

- Interpretation classification: `FUSION_MIXED_RESULT`
- B0 did not win, so the selected residual checkpoint is preserved for provenance.
- This is TRAIN-internal evidence only; it is not official DEV or TEST confirmation.

## Movement and perturbation diagnostics

On the 1,416 ordinary FUSION_VAL sources:

- Top-1 improved: `2`
- Top-1 worsened: `1`
- Top-1 unchanged: `1,413`
- MRR improved: `17`
- MRR worsened: `15`
- MRR unchanged: `1,384`

Across all 1,531 FUSION_VAL sources:

- Mean absolute rank displacement: `0.6545787067`
- Median absolute rank displacement: `0`
- 95th-percentile displacement: `3`
- Maximum displacement: `12`
- Top-1 change fraction: `0.0026126715`

### Representative λ summary

| λ | Mean Hit@1 | Mean MRR | Mean NDCG@10 | Mean Top-1 change fraction |
|---:|---:|---:|---:|---:|
| 0.05 | 0.6903248588 | 0.7673622511 | 0.7694366200 | 0.0003265839 |
| 0.10 | 0.6903248588 | 0.7674516408 | 0.7694488284 | 0.0003265839 |
| 0.20 | 0.6906779661 | 0.7676610337 | 0.7695004046 | 0.0016329197 |

These summaries are descriptive and did not alter the frozen grid or selection rule.

## Quarantine and freeze

- Step 10 official DEV feature extraction: `0`
- Step 10 official DEV scoring: `0`
- Step 10 official DEV training: `0`
- Step 10 TEST feature extraction: `0`
- Step 10 TEST scoring: `0`
- Step 10 TEST training: `0`
- No TEST lock was created.

Configuration freeze:

- Path: `artifacts/experiments/step10_fusion/config_freeze.json`
- SHA-256: `f1686da002e748a3018c9fb0d77e6439e38c85444daf1f37ddaa113c6c47cb43`
- Deterministic selection reconstruction: `PASS`
- Reconstruction artifact: `artifacts/experiments/step10_fusion/selection_reconstruction.json`
- Local sync: `STEP10_CONFIG_LOCAL_SYNC_VERIFIED = TRUE`

R2's original report wording stated that NDCG@10 selected the lower-weight-decay configuration. The full-precision representative table shows that the first differing criterion was MRR: `0.7677976091295469` versus `0.7677849981529529`. NDCG@10 was not reached by the lexicographic selector. The selected configuration is unchanged and the scientific selection is unaffected. This correction is recorded in `artifacts/experiments/step10_fusion/r2_selection_provenance_correction.json`.
