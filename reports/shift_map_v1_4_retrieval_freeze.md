# SHIFT-ICD Step 7.4B — SHIFT-MAP v1.4 Retrieval Freeze

**Status: COMPLETE**
**Scope: inference-only diagnostics, candidate generation, freeze, and Step 8 gate**
**Repository: `C:\Users\rohan\SHIFT-ICD`**

## 1. Repository and power preflight

- Repository branch before completion: `main`.
- No Step 7.4B model training, optimizer initialization, checkpoint creation, or weight modification occurred.
- Frozen raw, canonical, benchmark, BM25, dense, and Step 7.3 namespaces were not modified.
- Stable-power verification: `psutil.sensors_battery()` reported `power_plugged=True`, 80% charge, unlimited AC time.
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU; driver 566.07; total VRAM 6,144 MiB; CUDA available.
- Peak observed monitoring sample during the final runtime pass: 78°C, 70% utilization, 2,372 MiB VRAM used.
- Only one GPU job ran at a time; embedding batch size was capped at 16 and CPU threads at 2.
- Final project-process check: no active Step 7.4B process.

## 2. Frozen experiment and checkpoint

- Experiment: SHIFT-MAP v1.4 / Step 7.4B.
- Retriever: SHIFT-MAP v1.3 corrected L2 bi-encoder.
- Base: `FremyCompany/BioLORD-2023`.
- Base revision: `167aab527b238a50ca65224e6319215d2ff4fc9f`.
- Checkpoint: `artifacts/models/shift_map_v1/v1_3_final_l2_n1_seed17/epoch_3`.
- Canonical seed/epoch: 17 / 3.
- Verified checkpoint directory SHA-256: `7a859ff478d801f98a17fc966cb0ae81ba60362725add2735d91c7e01f4fe1d`.
- Positive policy: P2.
- Loss: L2 set-positive InfoNCE.
- Negative strategy: N1_RANDOM.
- Learning rate: `2e-5`.
- Evaluator: 2.0.
- Benchmark: 1.0.
- Canonical schema: 1.0.
- Target terminology: ICD-10-CM 2018.
- Target count: 17,513.
- Target corpus SHA-256: `a394fe6a0055b13e2df4eaeb4ac4b739e7d8d54277adb02f9bdecff67ad80a3a`.
- Similarity: cosine over L2-normalized embeddings.
- Checkpoint release: local only.

## 3. Representation drift

Identical target text, ordering, tokenizer/model-native pooling, maximum length 64, and L2 normalization were used for zero-shot and L2.

| Population | n | Mean cosine | Median | p5 | p95 | Mean distance |
|---|---:|---:|---:|---:|---:|---:|
| All targets | 17,513 | 0.963015 | 0.966681 | 0.929007 | 0.986210 | 0.036985 |
| TRAIN-positive targets | 13,061 | 0.962404 | 0.965956 | 0.929083 | 0.986001 | 0.037596 |
| Non-TRAIN-positive targets | 4,452 | 0.964807 | 0.968953 | 0.928851 | 0.986639 | 0.035193 |
| TRAIN-positive frequency 1 | 11,207 | 0.962702 | 0.966498 | 0.929085 | 0.986310 | 0.037298 |
| TRAIN-positive frequency 2–5 | 1,671 | 0.961514 | 0.963929 | 0.932023 | 0.982835 | 0.038486 |
| TRAIN-positive frequency 6+ | 183 | 0.952250 | 0.949141 | 0.915593 | 0.983947 | 0.047750 |

Interpretation: measurable movement is present, and the small 6+ exposure group has lower mean cosine, but this is not evidence of causality or catastrophic forgetting. There is no destructive-collapse claim.

## 4. Source drift and source/target comparison

| Population | n | Mean cosine | Median | p5 | p95 | Mean distance |
|---|---:|---:|---:|---:|---:|---:|
| Forward TRAIN sources | 10,197 | 0.967917 | 0.971468 | 0.935370 | 0.987455 | 0.032083 |
| Forward DEV sources | 1,457 | 0.968216 | 0.972213 | 0.934229 | 0.986996 | 0.031784 |
| Forward TEST sources | 2,913 | 0.968311 | 0.971883 | 0.935670 | 0.987430 | 0.031689 |
| TRAIN-positive targets | 13,061 | 0.962404 | 0.965956 | 0.929083 | 0.986001 | 0.037596 |
| Non-TRAIN-positive targets | 4,452 | 0.964807 | 0.968953 | 0.928851 | 0.986639 | 0.035193 |
| All targets | 17,513 | 0.963015 | 0.966681 | 0.929007 | 0.986210 | 0.036985 |

Sources have slightly higher old/new cosine than targets. The measured difference is modest and does not support a destructive representation-collapse claim.

## 5. Inference runtime and resource measurements

Measurements used CUDA, batch size 16, identical target encoding, and forward TRAIN/DEV/TEST source encoding. The script records model load, target encoding, source encoding, GPU peak allocation/reservation, and CPU RSS.

| Model | Load s | Target encoding s | TRAIN source s | Peak target allocated MiB | Peak target reserved MiB | CPU RSS after target MiB |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot BioLORD | 2.5342 | 27.4760 | 15.5518 | 496.9 | 568.0 | 1,821.4 |
| Corrected L1 seed 42 | 0.4314 | 29.0098 | 22.3604 | 914.2 | 1,016.0 | 1,938.3 |
| Canonical L2 seed 17 | 0.4242 | 32.7485 | 16.1113 | 1,333.2 | 1,468.0 | 2,039.8 |

Full DEV/TEST source timings are retained in `artifacts/experiments/shift_map_v1_4/runtime.json`. Exact matrix-retrieval latency was not separately instrumented; retrieval rankings came from the existing validated ledgers. Historical training peak memory remains `NOT_AVAILABLE`.

## 6. Previously frozen canonical comparison

Corrected L2 seed 17 versus frozen zero-shot BioLORD on forward ordinary answerable TEST (`n=2,695`):

- L2 Hit@1: 0.719852; Hit@5: 0.887570; Hit@10: 0.929128; Hit@25: 0.962894; Hit@50: 0.979592; Hit@100: 0.990353; MRR: 0.795232.
- Zero-shot Hit@1: 0.717996; Hit@5: 0.882004; Hit@10: 0.921336; Hit@25: 0.957328; Hit@50: 0.974397; Hit@100: 0.984045; MRR: 0.792142.
- Previously verified paired L2-minus-zero-shot: Hit@10 `+0.007792`, 95% CI `[+0.002968,+0.012616]`; Hit@100 `+0.006308`, 95% CI `[+0.003340,+0.009647]`.
- Hit@1 and MRR intervals included zero.

## 7. Post-hoc corrected L1 diagnostic

The recovered L1 seed-42 comparison is TEST-exposed and descriptive only. It does not replace the DEV-selected L2 checkpoint.

| Metric | L1 | L2 | L2 − L1 | 95% CI |
|---|---:|---:|---:|---:|
| Hit@1 | 0.738404 | 0.719852 | -0.018553 | [-0.027458,-0.009647] |
| Hit@5 | 0.906494 | 0.887570 | -0.018924 | [-0.025603,-0.012616] |
| Hit@10 | 0.944712 | 0.929128 | -0.015584 | [-0.021892,-0.009276] |
| Hit@100 | 0.992950 | 0.990353 | -0.002597 | [-0.004824,-0.000742] |
| MRR | 0.788046 | 0.770917 | -0.017129 | [-0.022315,-0.011767] |

Why L1 does not replace L2: L2 was selected using the corrected DEV-controlled protocol; replacing it using exposed TEST evidence would be test-set model selection.

## 8. Family-held-out paired bootstrap

L2-minus-zero-shot, 5,000 paired bootstrap replicates, seed 2026, identical benchmark IDs:

- Hit@10: `+0.003643`, n=549, 95% CI `[-0.003643,+0.010929]`.
- Hit@100: `+0.005464`, n=549, 95% CI `[0.000000,+0.012750]`.
- MRR: `+0.002963`, n=549, 95% CI `[-0.005482,+0.011192]`.

## 9. Backward transfer paired bootstrap

This is forward-trained L2 → backward transfer, not backward fine-tuning.

### Backward stratified TEST

- Hit@10: `+0.029222`, n=13,346, 95% CI `[+0.025249,+0.033343]`.
- Hit@100: `+0.016185`, n=13,346, 95% CI `[+0.013637,+0.018807]`.
- MRR: `+0.012913`, n=13,492, 95% CI `[+0.010813,+0.015071]`.

### Backward family-held-out TEST

- Hit@10: `+0.019658`, n=2,747, 95% CI `[+0.012013,+0.027303]`.
- Hit@100: `+0.008737`, n=2,747, 95% CI `[+0.004368,+0.013105]`.
- MRR: `+0.007873`, n=2,784, 95% CI `[+0.003634,+0.012147]`.

## 10. LEXICAL_LOW paired bootstrap

- Hit@1: `-0.004535`, n=882, 95% CI `[-0.020408,+0.011338]`.
- Hit@10: `+0.019274`, n=882, 95% CI `[+0.005669,+0.031746]`.
- Hit@100: `+0.019274`, n=882, 95% CI `[+0.011338,+0.029478]`.
- MRR: `-0.000035`, n=967, 95% CI `[-0.008582,+0.008669]`.

The result supports improvement at Hit@10/100 but not at Hit@1 or MRR.

## 11. Similarity geometry, hubness, and NO_MAP

Forward TEST target hubness was diffuse:

| Model | Unique Top-1 targets | Top-10 query fraction | Entropy bits | Gini |
|---|---:|---:|---:|---:|
| Zero-shot | 2,484 | 0.02472 | 11.1649 | 0.1334 |
| L1 | 2,498 | 0.02437 | 11.1712 | 0.1302 |
| L2 | 2,493 | 0.02472 | 11.1684 | 0.1315 |

NO_MAP diagnostics cover 85 forward TEST examples and are descriptive. No NO_MAP threshold, calibration, abstention, or classifier was selected.

## 12. Combination transfer

Forward TEST structural population: COMBINATION=64; COMBINATION_WITH_ALTERNATIVES=69; total=133.

At K=100:

- Zero-shot ChoiceListRecall: 0.900000; CompleteScenarioRetrieval: 0.804511.
- L1 ChoiceListRecall: 0.924812; CompleteScenarioRetrieval: 0.849624.
- L2 ChoiceListRecall: 0.920301; CompleteScenarioRetrieval: 0.842105.

L2’s combination incomplete-scenario floor is 0.157895. Combination metrics are structural characterization, not clinical-use evidence.

## 13. Candidate K and retrieval error floor

| K | Ordinary coverage | Ordinary miss rate | L2 complete-scenario coverage | Incomplete rate | Relative compute proxy |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.925743 | 0.074257 | retained in combination table | retained in combination table | 0.10 |
| 25 | 0.962164 | 0.037836 | retained in combination table | retained in combination table | 0.25 |
| 50 | 0.979844 | 0.020156 | retained in combination table | retained in combination table | 0.50 |
| 100 | 0.990806 | 0.009194 | 0.842105 | 0.157895 | 1.00 |

Final candidate K: **100**. This is an engineering coverage choice, not TEST-score tuning. The ordinary retrieval error floor at K=100 is 0.009647 using the canonical ordinary population; the combination incomplete-scenario floor is 0.157895. A reranker cannot recover a missing candidate.

## 14. Frozen candidate generation

Generated from the frozen L2 top-100 ledgers, with `candidate_is_gold` excluded from model input text:

| Split | Source count | Candidate rows | File SHA-256 |
|---|---:|---:|---|
| TRAIN | 10,197 | 1,019,700 | `f632854cd22ff9875390d9ff7280b4033b571c0c41ac54561730bf6f77536a35` |
| DEV | 1,457 | 145,700 | `ea7d94aa7f8e2d73d9d07dfff6d21a1b8e9a4a3db7837dd5dfa9f76feb92eafa` |
| TEST | 2,913 | 291,300 | `adcdcc2348165723c29f76687e5f487c2e5275dac5ff2389352ff4570d8f961b` |

Candidate manifests: `artifacts/candidates/shift_map_v2/manifest.json`.

Reranker eligibility at K=100:

- TRAIN ordinary 9,434; eligible 9,343; no valid candidate 91.
- DEV ordinary 1,348; eligible 1,331; no valid candidate 17.
- TEST characterization ordinary 2,695; eligible 2,669; no valid candidate 26.
- NO_MAP examples are not automatically converted to positive reranker examples.
- Combination labels remain structural and are not flattened into ordinary binary labels.

## 15. Step 8 candidate schema and preregistration

- Schema: `docs/interfaces/shift_map_v2_candidate_schema.md`.
- Freeze contract: `docs/experiments/shift_map_retrieval_freeze.md`.
- Preregistration skeleton: `docs/experiments/shift_map_v2_preregistration_skeleton.md`.
- Required fields include benchmark ID, direction, source/target code and description, rank, score, K, mapping kind, split, families, and `candidate_is_gold`.
- `candidate_is_gold` is permitted only for training-label construction/evaluation bookkeeping and must never enter model input text.
- Cross-encoder implementation/training did not occur.

## 16. Scientific position S1–S7

- S1 — **SUPPORTED**: L2 improves canonical candidate coverage over zero-shot at Hit@10/100 and ordinary K coverage.
- S2 — **PARTIALLY_SUPPORTED**: gains are statistically supported at Hit@10/100, not uniformly across Hit@1/MRR.
- S3 — **INCONCLUSIVE**: cardinality slices are descriptive; L2 is below recovered L1 on exposed TEST.
- S4 — **PARTIALLY_SUPPORTED**: LEXICAL_LOW Hit@10/100 improve; Hit@1/MRR do not.
- S5 — **SUPPORTED**: backward stratified and backward family-held-out paired deltas are positive with CIs excluding zero.
- S6 — **INCONCLUSIVE**: L2 combination structure is measured, but L1 remains higher descriptively and no causal claim is made.
- S7 — **PARTIALLY_SUPPORTED**: measurable drift exists without evidence of destructive collapse; drift is not causal evidence.

## 17. Final Step 8 gate

**GO-A — use frozen SHIFT-MAP v1.3 corrected L2 as the candidate generator.**

Step 8 is authorized to begin only under the frozen contract, candidate K=100, protected TEST policy, and preregistration skeleton. GO-A is based on DEV-controlled selection and corrected zero-shot comparison; the post-hoc TEST-exposed L1 result does not change it.

No clinical validation, clinical equivalence, clinical-use, or medical-decision claim is supported.

## 18. Artifact and quality-gate verification

- Experiment manifest: `artifacts/experiments/shift_map_v1_4/final_manifest.json`.
- Manifest records 90 hashed artifacts and the candidate-manifest hash.
- Tables generated: 26 CSV files under `reports/tables/shift_map_v1_4/`.
- Figures generated: 10 PNG files under `reports/figures/shift_map_v1_4/`.
- Tests added: analysis helpers, paired bootstrap behavior, coverage, grouping, hubness, and candidate-input leakage calculations.
- Pytest: `67 passed`.
- Ruff: `All checks passed!`.
- Mypy: `Success: no issues found in 28 source files`.
- Package import: `package_import_ok`.
- `git diff --check`: passed before commit.
- No model binaries or embeddings were staged for commit.

## 19. Commit and final status

The final commit is verified after commit with `git rev-parse HEAD`; the exact resulting HEAD is reported alongside this file.

## 20. Remaining limitations

1. Exact matrix-retrieval latency was not separately instrumented; encoding runtime and resource measurements are available.
2. Historical training peak memory is unavailable and remains explicitly `NOT_AVAILABLE`.
3. TEST was historically exposed during defective evaluation, corrected diagnostic replay, and corrected Step 7.3 evaluation; all TEST diagnostics here are disclosed as exposed characterization.
4. L1/L2 results are not evidence for changing the DEV-selected L2 checkpoint.
5. Combination and NO_MAP analyses do not establish clinical validity.
6. Step 8 modeling choices remain intentionally unselected beyond the frozen candidate-generator contract.
