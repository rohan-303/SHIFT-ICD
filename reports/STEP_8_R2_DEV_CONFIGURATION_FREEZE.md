# STEP 8-R2B — Corrected MedCPT DEV Configuration Freeze

## 1. Original fixed-list blocker

The original fixed list-size-8 representation could not preserve every candidate-contained valid positive for sources with more than eight positives. The historical blocker is preserved at `artifacts/experiments/step8_full_universe/r2_blocker.json`; no positives were truncated and no affected sources were excluded.

## 2. Count discrepancy reconciliation

The preliminary blocker reported 55 sources with more than eight positives. The authoritative TRAIN-only contract-v2 audit reports 48 sources with `P>8`, 27 with `P=8`, and 75 with `P>=8`. The exact explanation is an intermediate reporting/count-label error: `55` is the frequency of `P=5` in the authoritative distribution, not the count of sources with `P>8`. The exact `P>8` count is the sum of frequencies for 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 22, 23, 24, 31, 47, and 49, equal to 48. No contract-v2 count is affected.

## 3. Contract-v2 amendment

Contract v2 remains frozen before scientific DEV outcomes:

- `P_i` = all valid candidate-contained positives;
- `N_i = max(1, 8 - P_i)`;
- `L_i = P_i + N_i`;
- minimum one true negative per supervised source;
- source-normalized BCE and source-level listwise loss;
- effective batches are measured in sources;
- 9,224 gold-present ordinary TRAIN sources retained;
- 210 gold-missing sources excluded, never converted to all-negative examples.

Contract SHA-256: `17c3455e2ca2414f32fc11bfbbfb186dbc710b420d8b940042e5a5b390612d46`.

## 4. Candidate invariance

Frozen candidate membership was unchanged. TRAIN SHA is `d90dcb937748f1c0b173416d6b8f23f1c97ca8e8caf8934efdf8120993c1cb10`; DEV SHA is `c7240c1f38dd87f54d63c97999cba71d91f6d0887d89bc2d4094f9ba38ad6a6f`; TEST SHA was hash-verified only as `6ef38443a201b1d630148b30717d845b0785b1a4ed433eeab4b03d893b6bf1ce`. Every accepted row reports candidate mutation count zero.

## 5. R2B protocol

Protocol v2 SHA-256: `5f1a9d34249acf15b451ac8434eccb57e41ff4702027359fcbf036bd156d600d`. The search space was exactly the frozen space: two objectives, three negative strategies, and learning rates 1e-5, 2e-5, and 3e-5. Model input was description pair only, with max length 96, right truncation, longest padding, FP32 inference, and source batch size 32.

The pre-training R2B run-manifest SHA-256 is `74a9d77d4859ae6c532f3195f706e3c43e51a3d2a302e8f63641832f822033b2`, tied to source commit `59cfcabbc8ca55f02ae034f294a133f6ccf7e37b`.

## 6. Pre-rerank DEV baseline

The descriptive frozen-order DEV baseline contains 1,457 sources and all 100 frozen candidates per source:

| Metric | Baseline |
|---|---:|
| Hit@1 | 0.6965875371 |
| Hit@5 | 0.8620178042 |
| Hit@10 | 0.9050445104 |
| Hit@25 | 0.9421364985 |
| Hit@50 | 0.9651335312 |
| Hit@100 | 0.9777448071 |
| MRR | 0.7723505330 |
| NDCG@10 | 0.7738662523 |
| P_COMPLEX ChoiceListRecall@1 | 0.5142543860 |
| P_COMPLEX ChoiceListRecall@10 | 0.8585526316 |
| P_COMPLEX ChoiceListRecall@100 | 0.9638157895 |
| P_COMPLEX CompleteScenarioRetrieval@1 | 0.4736842105 |
| P_COMPLEX CompleteScenarioRetrieval@10 | 0.8157894737 |
| P_COMPLEX CompleteScenarioRetrieval@100 | 0.9473684211 |

## 7. Objective results

Both objectives were trained independently from the pinned MedCPT checkpoint for three epochs. Full per-epoch training losses, runtimes, checkpoint hashes, and all ordinary/structural metrics are in `reports/tables/step8_full_universe/ablation_master.csv` and `objective_ablation_dev.csv`.

| Objective | Selected epoch | Hit@1 | MRR | Hit@10 | NDCG@10 | P_COMPLEX Choice@1 | P_COMPLEX Complete@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BCE | 3 | 0.4629080119 | 0.5865481341 | 0.8249258160 | 0.6094879086 | 0.4451754386 | 0.7894736842 |
| SET_POSITIVE_LISTWISE | 3 | 0.5934718101 | 0.6891086710 | 0.8635014837 | 0.6955084314 | 0.4682017544 | 0.7631578947 |

## 8. Selected objective

Selected objective: `SET_POSITIVE_LISTWISE`.

Runner-up: `BCE`.

Decisive criterion: `Hit@1` under the frozen descending lexicographic rule, after within-run epoch selection. Selected checkpoint SHA: `6005a250c4c265e88ed652e393bf2ff55d825533c2b0c2186a99da679b66c169`.

Selection artifact: `artifacts/experiments/step8_full_universe/selected_objective.json`; SHA-256: `9fb3f6d4b02d3c6dad8ed796e4dfd489512178cd6e9fd74db7ab0c0a1edd9203`.

## 9. List/negative strategy results

The selected objective was held fixed while each frozen strategy was trained independently at the reference learning rate 1e-5. Full per-epoch records are in `list_strategy_dev.csv` and `ablation_master.csv`.

| Strategy | Selected epoch | Hit@1 | MRR | Hit@10 | NDCG@10 | P_COMPLEX Choice@1 | P_COMPLEX Complete@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TOP_RANK_HARD | 3 | 0.5808605341 | 0.6803576305 | 0.8627596439 | 0.6880948272 | 0.4682017544 | 0.7631578947 |
| MIXED_RANK | 3 | 0.5942136499 | 0.6892521595 | 0.8664688427 | 0.6958241604 | 0.4682017544 | 0.7631578947 |
| RANDOM_WITHIN_CANDIDATE | 3 | 0.5986646884 | 0.6923714095 | 0.8672106825 | 0.6995468238 | 0.4682017544 | 0.7631578947 |

## 10. Selected strategy

Selected strategy: `RANDOM_WITHIN_CANDIDATE`.

Runner-up: `MIXED_RANK`.

Decisive criterion: `Hit@1` under the frozen rule. Selected checkpoint SHA: `3e8084b8078ffc8de8db2dfbb3219ffb6ff4c5ec762835184f85b3f4b28317ec`.

Selection artifact: `artifacts/experiments/step8_full_universe/selected_list_strategy.json`; SHA-256: `6736eb966c6681ee195aa7ab5d54b2707bc22a8b2077489b9d919ab736193bf4`.

## 11. Learning-rate results

The selected objective and strategy were held fixed while each frozen learning rate was trained independently. Full per-epoch records are in `learning_rate_dev.csv` and `ablation_master.csv`.

| Learning rate | Selected epoch | Hit@1 | MRR | Hit@10 | NDCG@10 | P_COMPLEX Choice@1 | P_COMPLEX Complete@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-05 | 3 | 0.6068249258 | 0.7000713426 | 0.8731454006 | 0.7069241704 | 0.4945175439 | 0.7631578947 |
| 2e-05 | 3 | 0.6313056380 | 0.7198019781 | 0.8813056380 | 0.7249971930 | 0.5010964912 | 0.7631578947 |
| 3e-05 | 3 | 0.6602373887 | 0.7434669439 | 0.8916913947 | 0.7489092634 | 0.4813596491 | 0.7894736842 |

## 12. Selected learning rate

Selected learning rate: `3e-05`.

Runner-up: `2e-05`.

Decisive criterion: `Hit@1` under the frozen rule. Selected checkpoint SHA: `11e4404538e24c31d9ff13f658f3a7d58f4758a3ed7e4a70f90b43985c94260f`.

Selection artifact: `artifacts/experiments/step8_full_universe/selected_learning_rate.json`; SHA-256: `55d536cf203a3472e37d2fa0d29b670b770407e1aeb6775de9ef12441d98190e`.

## 13. Training stability

All 24 epoch rows were valid and finite. Every run used the pinned model revision and independent initialization from the original snapshot; no warm starts, OOMs, or TEST access occurred. Every run retained 9,224 supervised sources, 75 expanded lists, maximum list length 50, zero dropped positives, and zero positive-negative collisions.

The first operational output was quarantined because its stage-independent checkpoint filenames caused an overwrite between repeated configurations. The corrected rerun used stage-specific checkpoint identities, completed all 24 rows, and is the sole scientific evidence used here.

## 14. Deterministic selection reconstruction

`artifacts/experiments/step8_full_universe/selection_reconstruction.json` independently reproduces objective `SET_POSITIVE_LISTWISE`, strategy `RANDOM_WITHIN_CANDIDATE`, learning rate `3e-05`, and epoch 3 for every selected stage without hard-coded winners.

## 15. Final frozen configuration

The complete machine-readable freeze is `artifacts/experiments/step8_full_universe/config_freeze.json`. Retrieval freeze SHA-256 is `a5ec343a8113d65eacaf10a11824edf317847f63c6a15d53aaa5bfce15b9c5df`; candidate contract SHA-256 is `20ebe99b2fbf2a82773a3cd1e5778cdcb68c92b9c830ad689f4afd06e36a07e9`.

Final configuration: `SET_POSITIVE_LISTWISE`, `RANDOM_WITHIN_CANDIDATE`, `3e-05`, AdamW, weight decay 0.01, warmup ratio 0.1, gradient clipping 1.0, max length 96, FP32, source batch 32, effective source batch 32, maximum epochs 3, development seed 17.

## 16. TEST quarantine

The corrected TEST candidate artifact was hash-verified only. Corrected MedCPT TEST scoring count is 0 and corrected MedCPT TEST training count is 0.

**NO CORRECTED MEDCPT TEST SCORING OCCURRED DURING CONFIGURATION SELECTION.**

## 17. Final seed preregistration

The frozen final seed set is `[17, 42, 2026]`, with canonical final seed `17`. No final publication seed was trained.

## Evidence tables

Generated under `reports/tables/step8_full_universe/`:

- `objective_ablation_dev.csv`
- `list_strategy_dev.csv`
- `learning_rate_dev.csv`
- `ablation_master.csv`
- `config_selection_summary.csv`
- `training_runtime_ablation.csv`
- `candidate_invariance_audit.csv`
- `variable_list_audit.csv`
- `test_quarantine_audit.csv`

## Gate summary

- Metadata leakage regression: PASS; seven metadata-only perturbations produced identical tokenized inputs and logits on the selected checkpoint.
- Hit@100 invariance: PASS; all corrected rows equal `0.9777448071`.
- Structural @100 invariance: PASS; P_COMPLEX ChoiceListRecall@100 `0.9638157895` and CompleteScenarioRetrieval@100 `0.9473684211` remained unchanged.
- Local sync: PASS.
- TEST scoring/training: 0/0.

Status: **STEP8_CONFIG_FROZEN**
