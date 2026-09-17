# STEP 10-R1 — Semantic + Hierarchy Fusion Preflight

**Status:** `STEP10_FUSION_PROTOCOL_FROZEN`

## 1. Starting state and Step 9 provenance

- Starting HEAD: `c986ee8719d1f4b57ef21e6590b0b845228df906`
- Branch/worktree at preflight start: `main`, clean.
- Step 9 correction manifest: `artifacts/experiments/step9_hierarchy/r3_evaluation_correction_manifest.json`
- Correction-manifest SHA-256: `483ff129e682131274950a3386a92d820a7e4624fff734aa26168f226d03ef06`
- Original R3 freeze manifest SHA-256: `80286284a49f2a05c5c611382422445882aac0ffde23ac8c85d93c503f09fd59`
- Corrected Step 9 conclusion remains: `HIERARCHY_DEGRADES_RERANKING`.
- Canonical corrected structural populations remain `P_COMBINATION=64`, `P_COMBINATION_WITH_ALTERNATIVES=69`, `P_COMPLEX=133`.
- Canonical candidate-stage P_COMPLEX coverage remains ChoiceListRecall@100 `0.7323308271` and CompleteScenarioRetrieval@100 `0.4887218045`.
- Historical Step 9 artifacts were not modified.

## 2. Historical fusion recovery

The committed repository contains earlier score-fusion and hierarchy-aware concepts, but not this exact bounded residual protocol.

| Concept | Classification | Evidence/status |
|---|---|---|
| Alpha score-fusion / interpolation concept | `PREEXISTING_DOCUMENTED` | Shift-MAP v2 protocol documentation; not reused as a post-hoc DEV-selected method |
| Hierarchy-aware candidate-context reranking | `PREEXISTING_COMMITTED` | Step 9 hierarchy artifacts and implementation; completed negative confirmatory result |
| Earlier v3/v4 reranker concepts | `HISTORICAL_UNIMPLEMENTED` or documented design only where no executable implementation exists | Recovered during repository search; not retroactively treated as this method |
| `z_sem + lambda*tanh(H3 residual)` with λ in `{0.05,0.10,0.20}` | `NEW_STEP10_PROPOSAL` | Frozen by this R1 protocol; no scientific ablation run |

## 3. Semantic anchor and normalization

- Anchor: canonical corrected SHIFT-MAP seed-17 `retriever_score` stored in frozen candidate artifacts.
- Anchor file: `artifacts/candidates/shift_map_full_universe_v2/forward_train_k100.jsonl.gz`.
- TRAIN anchor file SHA-256: `d90dcb937748f1c0b173416d6b8f23f1c97ca8e8caf8934efdf8120993c1cb10`.
- Candidate membership mutation: prohibited.
- No SHIFT-MAP retraining, rescoring, MedCPT input, or semantic model replacement occurred.

For source `i` and its 100 frozen candidates:

```text
z_sem(i,j) = (s_ij - mean_i) / max(population_sd_i, 1e-8)
```

The computation is label-free, per-source, float64 for statistics, and deterministic; model inputs are converted to float32 after normalization. Population standard deviation (`ddof=0`) is frozen. Raw retriever ordering was compared with normalized ordering across the deterministic TRAIN-derived population; ordering remained unchanged. The zero-SD case maps deterministically to zero scores.

## 4. H3 residual contract

The exact reused Step 9 H3 feature order is:

1. `candidate_same_parent_fraction`
2. `candidate_same_family_fraction`
3. `candidate_shared_ancestor_fraction`
4. `candidate_same_root_fraction`

H3 is reused as the previously frozen candidate-context representation. H1, H2, semantic ancestor embeddings, MedCPT scores, raw ICD codes, labels, split metadata, and TEST outcomes are excluded from residual inputs.

The residual network is `4 -> 16 -> 1`, hidden `ReLU`, output `tanh`, so `r_h ∈ [-1,1]`.

```text
s_fused(i,j) = z_sem(i,j) + lambda * r_h(i,j)
```

Therefore `|s_fused-z_sem| <= lambda` for the preregistered λ values.

## 5. Nested development protocol

- Split population: original TRAIN sources only.
- Algorithm: deterministic source-level stratified shuffle within `mapping_kind × candidate-contained-positive-count-bin`.
- Seed: `20260917`.
- Target: approximately 85%/15%.
- FUSION_TRAIN sources: `8,666`.
- FUSION_TRAIN sorted source-ID SHA-256: `150a57de2e9101ee9bb5c1699bbb7b3a7f9b1fc1d0863018a8766808c151f24c`.
- FUSION_VAL sources: `1,531`.
- FUSION_VAL sorted source-ID SHA-256: `225df0f8343bc5358e1980688018916df4c75675595a77f37ca65ceea17bbcb9`.
- Split overlap: `0`.
- Inner scaler: fit on FUSION_TRAIN raw features only; no official DEV or TEST rows entered fitting.
- Inner scaler manifest SHA-256: `ef7d83130e6e295587c8559263fa6c90a099026d78bdda2f04feba9b0f7231e5`.
- Official DEV used in scaler fitting: `0`.
- TEST used in scaler fitting: `0`.
- Gold-missing ordinary FUSION_TRAIN sources are excluded from supervised listwise loss and are not converted to all-negative examples.

## 6. Frozen search and selection rules

- λ search: `0.05`, `0.10`, `0.20`; B0/λ=0 eligible non-trainable control.
- Learning rates: `1e-4`, `3e-4`.
- Weight decay: `1e-4`, `1e-3`.
- Trainable configurations: `12`.
- Source batch size: `32`.
- Epochs: `3`.
- Epoch rule: `E2_BEST_INNER_VAL_CHECKPOINT_WITHIN_3_EPOCHS`; ties select earliest epoch.
- Objective: `FULL_TOP100_SET_POSITIVE_LISTWISE`; all 100 frozen candidates per source, all candidate-contained valid positives in numerator, source-level mean loss.
- Ordered inner selection: ordinary Hit@1, ordinary MRR, P_COMPLEX CompleteScenarioRetrieval@10, P_COMPLEX ChoiceListRecall@10, ordinary NDCG@10, smaller λ, lower complexity/configuration ID, earlier epoch.
- B0 selection eligibility: yes; a fusion model is not guaranteed promotion.
- DEV-selection-rule SHA-256: recorded in `r1_artifact_hashes.json`.
- Interpretation-contract SHA-256: recorded in `r1_artifact_hashes.json`.

Baselines:

- B0: canonical frozen SHIFT-MAP ordering.
- B1: corrected MedCPT seed-17 ordering, descriptive only.
- B2: corrected hierarchy-only seed-17 ordering, descriptive only.
- Selection comparison: fusion versus B0 only.

## 7. Prior TEST exposure and quarantine

`prior_test_exposure.json` records `PRIOR_BENCHMARK_TEST_EXPOSURE_EXISTS`: corrected SHIFT-MAP, MedCPT Step 8, and hierarchy-only Step 9 have observed the benchmark TEST set; the Step 9 structural correction was performed after aggregate TEST outcomes existed. Step 10 inner selection uses TRAIN-internal data only. Official DEV is reserved for later one-shot external confirmation. TEST remains prohibited until a future valid lock.

- Official DEV scientific scoring count: `0`.
- Step 10 TEST feature/scoring count: `0`.
- Full FUSION_VAL scientific evaluation count: `0`.
- Step 10 hyperparameter search count: `0`.

Fail-closed checks produced exactly:

- `STEP10_OFFICIAL_DEV_ACCESS_FORBIDDEN`
- `STEP10_TEST_ACCESS_FORBIDDEN`

## 8. Determinism, leakage, invariance, and smoke validation

- Raw H3 extraction repeated twice on FUSION_TRAIN: identical hashes, `PASS`.
- Semantic normalization order invariance tests: `PASS`.
- λ=0 reproduces B0 ordering: `PASS`.
- Residual bound tests for all λ values: `PASS`.
- Candidate identity/count invariance: 100 candidates/source, mutation count `0`, `PASS`.
- Leakage tests: candidate-is-gold, mapping kind, split, GEM flags, approximate flag, scenario/choice labels, and TEST outcomes are not residual inputs, `PASS`.
- Smoke mode only: 64 supervised FUSION_TRAIN sources, full Top-100 lists, batch size 32.
- Finite losses/gradients: `PASS`.
- Parameter update: `PASS`.
- Observed tanh residual range: `[0.1059767, 0.2842313]`, within `[-1,1]`.
- Checkpoint save/reload: `PASS`.
- No full FUSION_VAL scientific scoring occurred.

## 9. Required artifacts

Created under `artifacts/experiments/step10_fusion/`:

- `research_question.json`
- `inner_split_manifest.json`
- `semantic_anchor_contract.json`
- `inner_train_scaler_manifest.json`
- `fusion_model_contract.json`
- `search_space.json`
- `dev_selection_rule.json`
- `interpretation_contract.json`
- `final_seed_policy.json`
- `prior_test_exposure.json`
- `r2_search_protocol.json`
- `r1_artifact_hashes.json`

Created under `reports/tables/step10_fusion/`:

- `legacy_fusion_recovery.csv`
- `inner_split_distribution.csv`
- `feature_inventory.csv`
- `feature_scale_audit.csv`
- `semantic_normalization_audit.csv`
- `smoke_validation.csv`
- `candidate_invariance_audit.csv`
- `leakage_audit.csv`

## 10. Final-seed policy

- Final seeds: `17`, `42`, `2026`.
- Canonical seed: `17`.
- Seed policy provenance: established project-wide convention; no result-based seed selection.
- Final-seed-policy SHA-256: recorded in `r1_artifact_hashes.json`.

## 11. R1 quality-gate boundary

R1 does not run the full fusion ablation, full FUSION_VAL scientific evaluation, official DEV scientific scoring, or TEST feature extraction/scoring. These are explicitly deferred to the next authorized milestones and future locks.

The next milestone is:

`STEP 10-R2 — TRAIN-INTERNAL FUSION ABLATIONS + CONFIGURATION FREEZE`
