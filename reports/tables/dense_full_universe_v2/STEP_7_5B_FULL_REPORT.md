# STEP 7.5B — Full-Universe Dense Baselines + Retriever Reselection

**Repository:** `C:\Users\rohan\SHIFT-ICD`  
**Branch:** `main`  
**Final local commit:** `e2e79b8091924e967139981604fb04e7bcdb66ab`  
**Commit:** `analysis: rerun dense baselines on full terminology universe`  
**Final local status:** clean

## Executive conclusion

The corrected full-universe dense baseline execution completed for SapBERT, BioLORD-2023, MedCPT dense, and Qwen3-Embedding-0.6B. All target representations were rebuilt against `terminology_universe_v2`; historical GEM-observed target artifacts were not reused. BioLORD-2023 was selected using corrected FORWARD STRATIFIED DEV only. TEST was evaluated after the test-lock workflow and did not influence selection.

The engineering and evaluation gates passed, but this milestone is conservatively classified `BLOCKED_DENSE_REPRODUCIBILITY` rather than fully frozen because some requested compact presentation artifacts/runtime fields are incomplete and the final `test_lock.json` retains stale `pending_remote_checkpoint_verification` strings. No SHIFT-MAP training, candidate generation, hard-negative mining, or MedCPT cross-encoder work was performed.

## 1. Frozen terminology universes

| Direction | Universe | Count | Code SHA-256 | Code+description SHA-256 | Retrieval corpus SHA-256 |
|---|---|---:|---|---|---|
| Forward | ICD-10-CM FY2018 | 71,704 | `8ca0d0cf0213581645fe64d9c1e599552a2fb207eb740f32ca09ffeaaf9f8e26` | `aa5b6b07afd6f699b8d36b98804dfc09fe87cae5ebc5940b394cf92d7919d6e3` | `32f572abeb2ff4cb003145216fa83bfd9984fab96579205f432993aa9c550464` |
| Backward | ICD-9-CM Version 32 | 14,567 | `204be6e29572332f3141a18347a28e95f2b258e5bacec8cf98046974adbc2eb5` | `bcdaf3d3029165d86cdca0461528476ebbddb76a8696c3baf75b6cd3585ce6cf` | `dcfe26da33d422a2395081fffaaf83e29d2f557d33458ade392d83b14e171b3b` |

Target membership was derived independently from the authoritative terminology resources, not from GEM-observed targets. Historical 17,513/11,690/11,689-target artifacts remain legacy evidence and are invalid for corrected absolute comparisons.

## 2. Evaluation contract

- Evaluator: `retrieval_evaluator_v3`
- Ordinary population: `SINGLE_EXACT`, `SINGLE_APPROXIMATE`, `ALTERNATIVE`
- Excluded from ordinary Hit/MRR: `COMBINATION`, `COMBINATION_WITH_ALTERNATIVES`, `NO_MAP`
- Structural populations: `P_COMBINATION`, `P_COMBINATION_WITH_ALTERNATIVES`, `P_COMPLEX`
- `NO_MAP`: separate diagnostic population
- Retrieval depth: exact full-universe Top-100
- Frozen BM25: `k1=2.0`, `b=0.75`
- Dense loading: `trust_remote_code=False`

## 3. Dense model provenance and inference semantics

| Model | Exact model revision | Dim. | Pooling | Max length | Normalization | Similarity |
|---|---|---:|---|---:|---|---|
| SapBERT | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext@090663c3ae57bf35ffe4d0d468a2a88d03051a4d` | 768 | CLS token | 64 | L2 | Exact normalized dot product |
| BioLORD-2023 | `FremyCompany/BioLORD-2023@167aab527b238a50ca65224e6319215d2ff4fc9f` | 768 | Native SentenceTransformer pooling | 64 | L2 | Exact normalized dot product |
| MedCPT dense | `ncbi/MedCPT-Query-Encoder@d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc` | 768 | CLS token | 64 | L2 | Exact normalized dot product |
| Qwen3-Embedding-0.6B | `Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | 1024 | Native SentenceTransformer pooling | 64 | L2 | Exact normalized dot product |

All embeddings were `float32`; inference autocast was disabled. Qwen used the frozen historical query instruction. Target embeddings were regenerated from scratch for the corrected universes.

## 4. Forward STRATIFIED DEV

Population: `P_ORDINARY_ANSWERABLE n=1,348`; full evaluated DEV rows `n=1,457`; `P_COMPLEX n=67`.

| Model | Hit@1 | Hit@5 | Hit@10 | Hit@25 | Hit@50 | Hit@100 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| SapBERT | 0.602374 | 0.781899 | 0.830861 | 0.882047 | 0.907270 | 0.933976 | 0.681550 |
| BioLORD-2023 | **0.635015** | **0.823442** | **0.873145** | **0.916172** | **0.942136** | **0.962908** | **0.720602** |
| MedCPT | 0.496291 | 0.694362 | 0.768546 | 0.836053 | 0.870920 | 0.901335 | 0.586306 |
| Qwen3-Embedding-0.6B | 0.528190 | 0.759644 | 0.825668 | 0.887982 | 0.919139 | 0.942136 | 0.632825 |

Structural DEV values are preserved in `artifacts/experiments/dense_full_universe_v2/neural_reselection.json` and `forward_dev_comparison.json`. For BioLORD, `P_COMPLEX CompleteScenarioRetrieval@100=0.3880597`.

## 5. RRF DEV

RRF was evaluated only as a hybrid comparison baseline, with frozen `rrf_k=60`; it was explicitly excluded from neural initialization. Full RRF DEV results are in `artifacts/experiments/dense_full_universe_v2/rrf_dev.json`.

Forward DEV ordinary RRF values:

| Dense component | Hit@1 | Hit@5 | Hit@10 | Hit@25 | Hit@50 | Hit@100 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| SapBERT + BM25 | 0.512611 | 0.747033 | 0.800445 | 0.873145 | 0.913205 | 0.933234 | 0.619364 |
| BioLORD + BM25 | 0.514837 | 0.764095 | 0.833828 | 0.906528 | 0.936201 | 0.957715 | 0.626795 |
| MedCPT + BM25 | 0.469585 | 0.703264 | 0.776706 | 0.845697 | 0.892433 | 0.920623 | 0.570778 |
| Qwen + BM25 | 0.486647 | 0.724036 | 0.798961 | 0.873887 | 0.913205 | 0.942136 | 0.592124 |

## 6. Neural reselection

Frozen rule: maximize, in order, DEV Hit@10, DEV MRR, DEV Hit@100, `P_COMPLEX CompleteScenarioRetrieval@100`, then lower mean query latency.

- Selected initialization: **BioLORD-2023**
- Runner-up: **SapBERT**
- TEST used for selection: **false**
- BioLORD minus SapBERT:
  - Hit@10: `+0.042285`
  - MRR: `+0.039052`
  - Hit@100: `+0.028932`
  - P_COMPLEX CompleteScenarioRetrieval@100: `+0.179104`

## 7. Corrected TEST lock and forward TEST

TEST lock artifact: `artifacts/experiments/dense_full_universe_v2/test_lock.json`  
Recorded lock SHA-256: `bf706bd320f13b6cda921123a24040e33efd171a5049b7659d8941c673cc94cd`

The required forward stratified ordinary population is `n=2,695`.

| Model | Hit@1 | Hit@5 | Hit@10 | Hit@25 | Hit@50 | Hit@100 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| SapBERT | 0.607792 | 0.789610 | 0.830798 | 0.874954 | 0.903154 | 0.925788 | 0.687709 |
| BioLORD-2023 | **0.636735** | **0.821892** | **0.870130** | **0.915028** | **0.944712** | **0.960668** | **0.719458** |
| MedCPT | 0.499443 | 0.702783 | 0.768089 | 0.831169 | 0.872356 | 0.908720 | 0.589262 |
| Qwen3-Embedding-0.6B | 0.538404 | 0.770315 | 0.832282 | 0.891280 | 0.924675 | 0.948794 | 0.641510 |

Forward TEST structural metrics and NO_MAP rows are in `remote_test_sync/results/test_results/all_metrics.json`.

## 8. RRF TEST

Fixed `rrf_k=60`; RRF was not used for model selection.

Forward stratified TEST ordinary RRF:

- Hit@1: `0.532096`
- Hit@5: `0.784416`
- Hit@10: `0.840445`
- Hit@25: `0.899814`
- Hit@50: `0.933210`
- Hit@100: `0.958071`
- MRR: `0.643283`

RRF forward family-held-out and structural values are in `rrf_test_metrics.json`.

## 9. Family-held-out and backward transfer

The following are ordinary metrics from `all_metrics.json`, reported as Hit@1/5/10/25/50/100/MRR.

### BioLORD-2023

- Forward family-held-out, `n=2,750`: `0.627636 / 0.815273 / 0.872727 / 0.919273 / 0.946182 / 0.968000 / 0.711224`
- Backward stratified, `n=13,346`: `0.305560 / 0.476098 / 0.559419 / 0.667316 / 0.746141 / 0.811104 / 0.388748`
- Backward family-held-out, `n=13,169`: `0.345205 / 0.512947 / 0.585314 / 0.671273 / 0.736123 / 0.797099 / 0.426066`

### MedCPT

- Forward family-held-out, `n=2,750`: `0.491273 / 0.690545 / 0.766182 / 0.840000 / 0.882545 / 0.917091 / 0.582285`
- Backward stratified, `n=13,346`: `0.295220 / 0.429192 / 0.489885 / 0.575378 / 0.623707 / 0.663719 / 0.359924`
- Backward family-held-out, `n=13,169`: `0.311033 / 0.442327 / 0.486673 / 0.558585 / 0.603311 / 0.652138 / 0.374349`

### Qwen3-Embedding-0.6B

- Forward family-held-out, `n=2,750`: `0.546545 / 0.765818 / 0.831273 / 0.893455 / 0.927636 / 0.954182 / 0.644696`
- Backward stratified, `n=13,346`: `0.259478 / 0.411359 / 0.467331 / 0.554548 / 0.624157 / 0.686423 / 0.332717`
- Backward family-held-out, `n=13,169`: `0.314147 / 0.469512 / 0.525476 / 0.596780 / 0.654492 / 0.710456 / 0.388495`

### SapBERT

- Forward family-held-out, `n=2,750`: `0.601455 / 0.789818 / 0.833455 / 0.876364 / 0.902909 / 0.927273 / 0.684909`
- Backward stratified, `n=13,346`: `0.257380 / 0.384085 / 0.432939 / 0.498277 / 0.543983 / 0.593211 / 0.318428`
- Backward family-held-out, `n=13,169`: `0.298656 / 0.439821 / 0.496393 / 0.566482 / 0.612499 / 0.660035 / 0.366434`

Structural metrics for every model/split remain in the machine-readable `all_metrics.json`; no structural number was inferred from ordinary metrics.

## 10. NO_MAP and selected-model failure analysis

NO_MAP was kept descriptive only; no threshold, abstention rule, or AUROC optimization was introduced.

Selected BioLORD forward TEST:

- Hit@10 failures: `350`
- Hit@100 failures: `106`

Largest failure slice:

- `SINGLE_APPROXIMATE / LEXICAL_LOW`: `n=651`, Hit@10 failures `233`, Hit@100 failures `90`
- `ALTERNATIVE / LEXICAL_LOW`: `n=170`, Hit@10 failures `41`, Hit@100 failures `7`
- `SINGLE_APPROXIMATE / LEXICAL_MEDIUM`: `n=466`, Hit@10 failures `37`, Hit@100 failures `6`

Full slice table: `artifacts/experiments/dense_full_universe_v2/selected_model_failure_slices.csv`.

## 11. Corrected BM25 comparison

Frozen forward stratified BM25 ordinary reference:

`Hit@1=0.441558`, `Hit@5=0.648609`, `Hit@10=0.710946`, `Hit@25=0.784045`, `Hit@50=0.817069`, `Hit@100=0.847866`, `MRR=0.529388`.

Paired bootstrap compares selected BioLORD with matched corrected BM25 on `n=2,695`, using seed `20260906` and `10,000` replicates:

| Metric | Dense − BM25 delta | 95% CI |
|---|---:|---:|
| Hit@10 | +0.159184 | [+0.143228, +0.175510] |
| Hit@100 | +0.112801 | [+0.100186, +0.125788] |
| MRR | +0.190070 | [+0.175498, +0.204192] |

These are descriptive baseline comparisons and were not used for selection.

## 12. Runtime and environment

- GPU allocation: GPU 0 SapBERT/BioLORD; GPU 1 MedCPT/Qwen.
- Checkpoint smoke validation: completed for all four exact revisions.
- Query timing is present in `all_metrics.json`.
- Complete per-model target-encoding time, source-encoding time, model-load time, p95 latency, peak allocated/reserved memory, and embedding byte size were not consistently materialized into a final runtime table: **NOT FULLY RECORDED**.
- No local embedding matrices were synchronized.

## 13. Remote execution failures and handling

Several early background launches failed before scientific execution:

1. `OWNER.json identity mismatch` during stale task-wrapper validation.
2. `state/run.lock` already existed on a subsequent duplicate launch.

Read-only inspection showed the task was already `finished` with recorded exit code `0`, and both recorded PIDs were inactive. No process was killed, restarted, or modified. Later corrected task workspaces produced the authoritative results.

## 14. Quality gates

### Local

- pytest: `99 passed`
- Ruff: passed
- mypy: passed; 34 source files
- package import: `PACKAGE_IMPORT_PASS`
- `git diff --check`: passed
- final Git status: clean

### Remote

- targeted dense tests: `19 passed`
- remote Ruff/mypy/import/diff checks: passed
- target counts, corpus hashes, target-order hashes, model revisions, and output manifests: verified

## 15. Provenance and artifact status

Committed result artifacts include:

- corrected model registry;
- corrected dense DEV comparison and reselection records;
- corrected TEST lock and synchronized TEST results;
- RRF DEV/TEST metrics;
- paired bootstrap JSON;
- selected-model failure rows and slices;
- compact ordinary report tables;
- remote output manifest and execution metadata.

Local compact TEST package hash:
`165b6f7ab4f7c9558e2d4946922bee1daab3ee6df09c18f9f14e7d77ed2e04fe`.

The selected model’s large target embeddings remain on the private remote workspace for Step 7.5C. They must be reused only after fail-closed verification of corpus hash, target-order hash, model revision, and encoding configuration.

## 16. Final gate

**Status: `BLOCKED_DENSE_REPRODUCIBILITY`**

The core corrected dense execution and evaluation evidence is present and locally verified. The milestone is not labelled fully frozen because:

1. some requested structural/runtime presentation tables are not fully materialized;
2. the final test lock contains stale pending-verification strings despite successful checkpoint smoke evidence;
3. the final remote scientific source was packaged rather than maintained as a tracked checkout exactly at the final local commit.

These are reproducibility/reporting blockers, not evidence of model failure. Do not train SHIFT-MAP or generate Step 8 candidates until the blockers are resolved or explicitly accepted.

## 17. Explicit next milestone

`STEP 7.5C — FULL-UNIVERSE SHIFT-MAP RETRAINING + CANDIDATE FREEZE`

Planned scope after closure: selected-model hard-negative construction, SHIFT-MAP training, full-universe candidate freeze, and subsequent candidate-generation validation. No such work was started in Step 7.5B.

## 18. STEP 7.5B-R REPRODUCIBILITY CLOSURE

This section records closure of the blockers identified in the original report. It distinguishes original scientific results from deterministic posthoc presentation/provenance reconstruction.

| Blocker | Original state | Corrective action | Evidence | Final status |
|---|---|---|---|---|
| B01 structural presentation | Required structural CSVs were incomplete | Aggregated existing corrected per-row Top-100 results by population and split using the v3 metric fields; no embedding inference rerun | `forward_test_structural.csv`, `forward_family_structural.csv`, `backward_stratified_structural.csv`, `backward_family_structural.csv` | CLOSED |
| B02 runtime evidence | Component runtime/p95/VRAM fields were not consistently materialized | Materialized original elapsed/mean timings and marked unavailable fields `NOT_RECORDED`; no original runtime was fabricated | `runtime_original.csv` | CLOSED |
| B03 TEST-lock metadata | Lock contained stale descriptive pending-verification strings | Verified lock chronology, terminology hashes, evaluator, model/revision bindings, inference semantics, TEST exclusion from selection, and exact lock SHA-256. The original lock bytes were not rewritten | `test_lock.json`, `test_lock.sha256`, closure tests | CLOSED |
| B04 remote code provenance | Remote execution was packaged rather than a tracked checkout at final local HEAD | Verified OWNER/task/input/source archive hashes and preserved the exact source archive locally. Classified as RP2, not RP1 | `remote_code_provenance.csv`; `C:\Users\rohan\SHIFT-ICD-provenance\step_7_5b\source.tar.gz` | CLOSED |
| B05 selected-cache retention | Selected BioLORD cache lacked a dedicated fail-closed manifest | Recorded exact remote paths, model revision, corpus/order hashes, shapes, dtype, normalization, similarity, and array SHA-256 values | `selected_biolord_cache_manifest.json` | CLOSED |
| B06 release freeze graph | No final hash-linked freeze manifest existed | Created canonical compact export, freeze manifest, and freeze-manifest checksum | `freeze_manifest.json`, `freeze_manifest.sha256`, compact export ZIP | CLOSED |

### Original scientific result versus reconstruction

The four full-universe dense model runs, corrected DEV-only selection, corrected locked TEST evaluation, RRF comparison, family-held-out evaluation, backward transfer, failure analysis, and paired bootstrap are **ORIGINAL SCIENTIFIC RESULTS**. Their metrics and rankings were not rerun in this closure.

The structural CSVs, canonical table materialization, blocker matrix, selected-cache manifest, remote provenance record, compact export, and freeze manifest are **POSTHOC PRESENTATION / PROVENANCE RECONSTRUCTION** from existing result rows, manifests, and hashes. Runtime fields marked `NOT_RECORDED` were never available in the original evidence and remain explicitly unavailable.

### Closure verification

- Corrected target metadata: 8/8 embedding metadata files passed count, corpus, code, revision, dtype, normalization, and similarity checks.
- Old-universe cache rejection: passed; no corrected artifact references 17,513, 11,690, or 11,689 target counts.
- DEV reselection: deterministic reconstruction selects BioLORD-2023 on `FORWARD_STRATIFIED_DEV/P_ORDINARY_ANSWERABLE`; TEST and runtime do not alter selection.
- TEST lock: exact SHA-256 `bf706bd320f13b6cda921123a24040e33efd171a5049b7659d8941c673cc94cd`; result files are later than the lock artifact.
- Remote code: RP2; recorded execution commit `59d408e26f78adad7dcbe12529be316e9d0fd6ab`, recorded remote source head `db3898f0f2fb7e66f1ff1e07c99d67c7ba17b4bf`, source archive SHA-256 `1e384a8bc814e518c4a5a6e76f3c61a1c979e2977449085ee986563968e1b062`.
- Remote workspace retained: `/home/gra_rohan/Rohan/tasks/shift-icd-dense-stage_20260906T061131Z_2ca1171d`; corrected BioLORD embeddings remain available for Step 7.5C.
- Local compact export: SHA-256 `cd9c8328a1b550f23a8e1084148ae0b654c25f6053256571b00acded7000f4a6`.
- Local sync verified: `TRUE`.
- Final status: **`DENSE_FULL_UNIVERSE_FROZEN`**.

No SHIFT-MAP training, hard-negative mining, Step 8 candidate generation, fine-tuning, or MedCPT cross-encoder work was performed in this closure.
