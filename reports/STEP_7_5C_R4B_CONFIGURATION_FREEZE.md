# STEP 7.5C-R4B — Configuration Freeze Report

## Status

`SHIFT_MAP_CONFIG_FROZEN`

This report closes DEV-only configuration selection for the corrected full-universe SHIFT-MAP retraining. The freeze uses only corrected FORWARD STRATIFIED DEV and `P_ORDINARY_ANSWERABLE`.

## Frozen configuration

| Parameter | Frozen value |
|---|---|
| Terminology universe | `terminology_universe_v2` |
| Forward target count | 71,704 |
| Forward target order hash | `8ca0d0cf0213581645fe64d9c1e599552a2fb207eb740f32ca09ffeaaf9f8e26` |
| Retrieval corpus hash | `32f572abeb2ff4cb003145216fa83bfd9984fab96579205f432993aa9c550464` |
| Base model | `FremyCompany/BioLORD-2023` |
| Base revision | `167aab527b238a50ca65224e6319215d2ff4fc9f` |
| Positive policy | `P1` |
| Loss | `L1` |
| Negative strategy | `lexical_hard` |
| Learning rate | `1e-5` |
| Epoch rule | `E2_BEST_DEV_CHECKPOINT_WITHIN_MAX_EPOCH_BUDGET` |
| Maximum epochs | 3 |
| Final training seeds | 17, 42, 2026 (authorized only after this freeze) |

## Selection results

- **Positive policy:** P1, epoch 3; Hit@100 = 0.9740356083.
- **Loss:** L1, epoch 3; tied with L2 on Hit@100, CompleteScenarioRetrieval@100, Hit@10, and Hit@1; L1 won MRR: 0.7340214374 vs 0.7339838171.
- **Negative strategy:** `lexical_hard`, epoch 2; tied with `mixed` on Hit@100, then won CompleteScenarioRetrieval@100: 0.5223880597 vs 0.4925373134.
- **Learning rate:** `1e-5`, epoch 2; tied with `2e-5` on Hit@100, then won CompleteScenarioRetrieval@100: 0.5223880597 vs 0.4925373134.

Selected LR DEV metrics: Hit@100 = 0.9792284866; CompleteScenarioRetrieval@100 = 0.5223880597; Hit@10 = 0.9057863501; MRR = 0.7738648850; Hit@1 = 0.6988130564.

## Integrity and provenance

- Negative mining manifest SHA-256: `68e820c170b9f8f3be9e87f679d239f6495db7c5bce898cc95b71d0208c289cf`.
- Frozen negative-set SHA-256: `2f75fb8e954d4f85e35316fd67b598282931852108de0a7c62b6160fa3e2a7ce`.
- All ablations initialized independently from the pinned original BioLORD checkpoint.
- All metadata records report `test_data_used: false`.
- Corrected TEST invocation count: 0.
- No legacy 17,513-target universe references detected in the accepted selection records.
- No final-seed training, TEST lock/evaluation, candidate generation, MedCPT, or Step 8 execution occurred in R4B.

## Evidence artifacts

- `artifacts/experiments/shift_map_full_universe/config_freeze.json`
- `artifacts/experiments/shift_map_full_universe/selected_loss.json`
- `artifacts/experiments/shift_map_full_universe/selected_negative_strategy.json`
- `artifacts/experiments/shift_map_full_universe/selected_learning_rate.json`
- `artifacts/experiments/shift_map_full_universe/lr_search_space.json`
- `artifacts/experiments/shift_map_full_universe/selection_reconstruction.json`
- `reports/tables/shift_map_full_universe/ablation_master.csv`
- `reports/tables/shift_map_full_universe/positive_policy_dev.csv`
- `reports/tables/shift_map_full_universe/loss_ablation_dev.csv`
- `reports/tables/shift_map_full_universe/negative_strategy_dev.csv`
- `reports/tables/shift_map_full_universe/learning_rate_dev.csv`
- `reports/tables/shift_map_full_universe/training_runtime_ablation.csv`
- `reports/tables/shift_map_full_universe/config_selection_summary.csv`

## Stop gate

`CONFIG_SELECTION_LOCAL_SYNC_VERIFIED`

R4B stops here. Final-seed training and corrected TEST evaluation require the subsequent authorized milestone.
