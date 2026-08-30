# SHIFT-ICD dense_v1_1 freeze report

## 1. Reason for Step 6.1

Step 6 executed zero-shot SapBERT, BioLORD-2023, MedCPT Query Encoder, Qwen3-Embedding-0.6B, and fixed BM25+BioLORD RRF against the frozen Track A v1.0 benchmark. Step 6.1 creates a new `dense_v1_1` namespace to audit scopes, preserve frozen predictions, reconcile paired metrics, complete tables/figures, document BioLORD release constraints, and preregister the SHIFT-MAP v1 boundary. No encoder was rerun, updated, fine-tuned, or optimized.

## 2. Repository and integrity state

- Branch: `main`
- Commit before Step 6.1: `763b00c78c3201b8da4b800d8104ec3c2a2422c8`
- Canonical schema: `1.0`
- Benchmark: Track A `1.0`
- BM25: `bm25_v1_1`, `k1=2.0`, `b=0.75`, `q1_long_only`
- Raw, canonical, benchmark, BM25, and dense cache verification: passed
- Model weights changed: no
- Optimizer created: no
- Training occurred: no

## 3. Dense experiment version

Historical `dense_v1` outputs were not overwritten. `dense_v1_1` is a corrected/completed analysis namespace built from frozen dense ranking JSONL and the optimized frozen BM25 v1.1 ranking artifacts. The missing fixed-RRF rows were deterministically reconstructed from frozen BM25 and BioLORD rankings with `k=60`; neural encoders were not rerun.

## 4. Numerical-result audit

No Step 6 primary system metric was wrong. The forward stratified TEST values reproduce exactly.

The historical paired BioLORD Hit@10 delta `+0.136139` used `n=2,828` answerable examples and included the 133 combination examples in binary Hit@K. The displayed primary Hit@10 excludes combination, uses `n=2,695`, and gives `+0.139889`. The canonical `dense_v1_1` primary delta is therefore:

```text
mean(per-example BioLORD Hit@10 - BM25 Hit@10)
over answerable, non-combination forward stratified TEST examples
= 0.9213358070500928 - 0.7814471243042672
= +0.1398886827458256
```

The broad answerable-only delta remains a valid secondary contract and is retained in the historical artifact, but it is not the primary non-combination definition.

## 5. Scope audit

Every generated result row carries model, direction, split protocol, partition, benchmark version, experiment version, sample population, total n, answerable n, applicable metric n, and slice fields where relevant.

| Population | Direction | n |
|---|---|---:|
| forward_stratified_test | ICD9CM → ICD10CM | 2,913 |
| forward_family_held_out_test | ICD9CM → ICD10CM | 2,908 |
| backward_stratified_test | ICD10CM → ICD9CM | 14,341 |
| backward_family_held_out_test | ICD10CM → ICD9CM | 14,332 |

All six systems have identical benchmark IDs within each population. Automated assertions pass in `artifacts/experiments/dense_v1_1/scope_audit.json` and `scripts/validate_dense_v1_1.py`.

## 6. DEV selection audit

The declared lexicographic criterion remains:

1. non-combination answerable Hit@100;
2. CompleteScenarioRetrieval@100;
3. Hit@10;
4. MRR.

Independent reconstruction selects **BioLORD-2023**. TEST performance was not used. The complete audit is in `artifacts/experiments/dense_v1_1/dev_selection_audit.json`.

## 7. Primary forward stratified TEST

| System | Hit@1 | Hit@5 | Hit@10 | Hit@25 | Hit@50 | Hit@100 | MRR | n total | n answerable | applicable n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | .584045 | .733952 | .781447 | .841929 | .871243 | .892022 | .651011 | 2913 | 2828 | 2695 |
| SapBERT | .683488 | .834879 | .870501 | .913915 | .936178 | .953618 | .752014 | 2913 | 2828 | 2695 |
| BioLORD-2023 | **.717996** | **.882004** | **.921336** | **.957328** | **.974397** | **.984045** | **.792142** | 2913 | 2828 | 2695 |
| MedCPT | .604453 | .788497 | .844527 | .896846 | .928015 | .950649 | .687618 | 2913 | 2828 | 2695 |
| Qwen3-Embedding-0.6B | .660111 | .851206 | .900186 | .940260 | .961781 | .975881 | .744939 | 2913 | 2828 | 2695 |
| BM25 + BioLORD RRF | .675325 | .837106 | .888312 | .942857 | .970686 | .981447 | .749676 | 2913 | 2828 | 2695 |

The complete machine-readable table is `reports/tables/dense_v1_1/forward_stratified_overall.csv`.

## 8. Family-held-out and backward results

Complete tables with explicit scopes are provided in:

- `forward_family_held_out_overall.csv`
- `backward_stratified_overall.csv`
- `backward_family_held_out_overall.csv`
- `forward_family_comparison.csv`
- `backward_family_comparison.csv`
- `forward_backward_comparison.csv`

BioLORD Hit@10 / Hit@100 / MRR:

| Population | Hit@10 | Hit@100 | MRR |
|---|---:|---:|---:|
| Forward family-held-out | .935273 | .985091 | .799462 |
| Backward stratified | .596958 | .836355 | .419244 |
| Backward family-held-out | .618574 | .811147 | .454878 |

Forward and backward performance is asymmetric; direction is not collapsed into one aggregate.

## 9. Lexical difficulty and mapping kinds

Complete model-by-slice tables are in:

- `forward_stratified_by_lexical_difficulty.csv`
- `forward_stratified_by_mapping_kind.csv`
- `forward_stratified_by_alternative_size.csv`
- `forward_stratified_combination.csv`

On LEXICAL_LOW, BM25 Hit@10 is `.418367`; BioLORD Hit@10 is `.786848`. Combination results are reported with ChoiceListRecall@K and CompleteScenarioRetrieval@K rather than treating components as independent complete mappings.

## 10. Multi-scenario audit

The audit found zero `MULTI_SCENARIO` examples in all four TEST populations:

- forward stratified: 0
- forward family-held-out: 0
- backward stratified: 0
- backward family-held-out: 0

No metric was fabricated for an empty multi-scenario population.

## 11. Complementarity and RRF

Forward stratified TEST BioLORD complementarity:

| K | Both succeed | BM25 only | BioLORD only | Both fail |
|---:|---:|---:|---:|---:|
| 10 | 2,162 | 53 | 438 | 260 |
| 100 | 2,528 | 6 | 257 | 122 |

Fixed RRF (`k=60`) underperformed BioLORD alone on the main metrics. Descriptive failure counts:

| K | BioLORD succeeds, RRF fails | RRF succeeds, BioLORD fails |
|---:|---:|---:|
| 10 | 152 | 62 |
| 100 | 12 | 5 |

The likely operational interpretation is that BM25 rank contributions can demote semantically correct BioLORD candidates; this is a hypothesis for future learned-fusion work, not a causal conclusion. RRF was not tuned.

Complete tables are `forward_bm25_dense_complementarity.csv`, `forward_oracle_union.csv` where applicable, and `forward_rrf_failure_analysis.csv`.

## 12. Statistical comparisons

Paired bootstrap comparisons use the canonical primary population: answerable, non-combination forward stratified TEST examples (`n=2,695`). Results are in `forward_paired_bootstrap.csv`. BioLORD versus Qwen is included.

| Comparison | Metric | Delta | 95% CI |
|---|---|---:|---|
| SapBERT vs BM25 | Hit@10 | +.089054 | [.074583, .102783] |
| BioLORD vs BM25 | Hit@10 | **+.139889** | [.125046, .154731] |
| MedCPT vs BM25 | Hit@10 | +.063080 | [.047124, .078302] |
| Qwen vs BM25 | Hit@10 | +.118738 | [.104267, .132839] |
| RRF vs BM25 | Hit@10 | +.106865 | [.094981, .119109] |
| BioLORD vs Qwen | Hit@10 | +.021150 | [.010761, .032282] |

McNemar and Holm-adjusted p-values were not reported.

## 13. NO_MAP diagnostics

For BM25 and every dense model, the report includes NO_MAP versus ANSWERABLE distributions for maximum score/similarity, top-1/top-2 margin, and mean top-5 score. Dense values are cosine similarities; BM25 values are BM25 scores.

These are explicitly **POST-HOC DIAGNOSTICS ONLY**. No threshold, calibration, abstention classifier, or routing model was trained.

## 14. Runtime and resource comparison

The Step 6 runtime artifact did not retain the complete per-model timing fields required for a valid quality-versus-cost comparison. `dense_v1_1/dense_runtime.csv` therefore preserves the required schema with unavailable values rather than inventing them. Figure 8 is labeled accordingly. A valid runtime comparison requires a separately declared measurement-only rerun; runtime was not used for model selection.

## 15. BioLORD provenance and licensing

BioLORD exact revision:

```text
FremyCompany/BioLORD-2023
167aab527b238a50ca65224e6319215d2ff4fc9f
```

The model card declares `ihtsdo-and-nlm-licences`, says the author’s contributions are MIT, and requires users to ensure appropriate UMLS and SNOMED CT licensing. Local academic fine-tuning is not stated to be prohibited, so it is not automatically blocked. Public redistribution of fine-tuned weights is **unclear** and must not be promised. Code, configs, benchmark-generation code, IDs, seeds, and metrics can be prepared for release subject to data-term review; checkpoints and derived embeddings should remain private/controlled until terms are confirmed.

See `docs/models/biolord_license_notes.md`.

## 16. Error-taxonomy status

Mechanistic samples were generated, but no unsupported medical judgments were assigned. The approximately 100 BioLORD failures and 50 BM25-failed/BioLORD-success cases are labeled `unclear` unless a defensible mechanical mechanism is available. Retrieval failure alone does not establish granularity, clinical equivalence, or ontology error.

## 17. Prediction ledger

`reports/tables/dense_v1_1/forward_stratified_prediction_ledger.csv` contains 17,478 rows: 2,913 benchmark examples × 6 systems. It includes top-1 target/score, best valid-target rank, Hit@K, MRR, mapping kind, lexical difficulty, alternative-size bucket, and combination structural metrics. The ledger manifest is `artifacts/experiments/dense_v1_1/prediction_ledger_manifest.json`.

## 18. Frozen scientific findings

1. Dense retrieval substantially improves over BM25 on forward stratified TEST, especially BioLORD.
2. BioLORD is the strongest DEV-selected and observed forward baseline under the declared criterion.
3. The improvement is disproportionately large on LEXICAL_LOW.
4. Combination mappings remain a separate structural problem and are substantially harder than ordinary mappings.
5. Forward and backward mapping performance is asymmetric.
6. BM25 and BioLORD have complementary errors, but fixed RRF does not improve over BioLORD alone.

These are retrieval findings against CMS-derived benchmark gold, not claims of clinical equivalence.

## 19. SHIFT-MAP v1 boundary

`docs/experiments/shift_map_v1_training_boundary.md` defines:

- gradient updates only from FORWARD STRATIFIED TRAIN;
- DEV only for early stopping, hyperparameters, negative-mining strategy, and checkpoint selection;
- TEST never used until configuration freeze;
- simple pairwise training excludes combination and combination-with-alternative examples;
- alternatives are benchmark-valid positive relations, not claims of clinical interchangeability;
- planned negatives N1–N6;
- no TEST/DEV gold contamination during negative mining.

No negative dataset was mined and no training occurred in Step 6.1.

## 20. Outputs and quality gates

Generated:

- 22 scoped CSV tables under `reports/tables/dense_v1_1/`;
- 9 figures under `reports/figures/dense_v1_1/`;
- `dense_v1_1` manifest, scope audit, frozen metrics, model provenance, paired comparisons, RRF analysis, NO_MAP diagnostics, multi-scenario audit, and ledger manifest;
- BioLORD license notes;
- SHIFT-MAP v1 training boundary.

Validation completed:

```text
pytest: 40 passed
ruff: All checks passed
mypy: Success: no issues found in 24 source files
validate_dense_v1_1.py: passed
raw/canonical/benchmark/BM25/dense cache validation: passed
```

## 21. Recommendation

It is safe to begin a later SHIFT-MAP v1 fine-tuning milestone **only under the documented training boundary and after an additional licensing review before any checkpoint redistribution**. Step 6.1 itself performed no fine-tuning, optimizer creation, hard-negative mining, reranking, calibration, or routing.
