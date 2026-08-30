# BM25 v1.1 Audit and Correction Report

## Reason for audit
Step 5 slice tables reported counts larger than the stated forward stratified TEST population, and the V54.12 case study incorrectly reported the source as absent.

## Scope and root cause
The old slice generation collected metric rows from all four TEST datasets and filtered only `system == BM25 Dev-Tuned`; it omitted the dataset/scope filter. The old mapping-kind and lexical tables therefore each reported **34,494 rows**, the sum of the four TEST dataset sizes: forward stratified 2,913; forward family-held-out 2,908; backward stratified 14,341; backward family-held-out 14,332. The old combination table reported 2,172 complex rows from that same mixed population. The corrected pipeline filters explicit direction, protocol, partition, and benchmark version before aggregation.

## Overall metrics
The frozen forward stratified TEST metrics were independently recomputed and are unchanged: Hit@1 `0.584045`, Hit@5 `0.733952`, Hit@10 `0.781447`, Hit@25 `0.841929`, Hit@50 `0.871243`, Hit@100 `0.892022`, MRR `0.651011`. BM25 tuning remains valid because v1.1 did not retune; it retains `k1=2.0`, `b=0.75`, Q1 long-description-only, selected previously on forward stratified DEV.

## V54.12 / V5412
Canonical and benchmark storage use normalized source code `V5412`; display form is `V54.12`. The case exists as `track_a_v1.0:ICD9CM_TO_ICD10CM:V5412`, mapping kind ALTERNATIVE, 533 valid targets, in forward stratified TRAIN. The prior absence was a case-study lookup bug. It was not a benchmark eligibility exclusion. Its valid-target density is `533 / 17,513 = 0.030435`; the best valid target rank and top-K coverage are in `artifacts/experiments/bm25_v1_1/v54_12_case_study.json`. Hit@K is mechanically easier with larger gold sets and must not be compared without alternative-set context.

## Membership and benchmark version
Canonical and benchmark source-level keys both contain 14,567 forward and 71,704 backward sources. `canonical - benchmark = empty`; `benchmark - canonical = empty`. Unique benchmark IDs and `(direction, source_code)` keys are preserved by existing validation. Benchmark version remains **1.0**; no benchmark semantics changed. Experiment version is **bm25_v1_1**, a result-pipeline correction preserving bm25_v1 provenance.

## Corrected slices
`forward_stratified_by_mapping_kind.csv` contains mutually exclusive mapping-kind membership summing to 2,913. Combination rows report ChoiceListRecall@K and CompleteScenarioRetrieval@K; ordinary Hit@K is intentionally blank for combinations. `forward_stratified_by_lexical_difficulty.csv` contains exact, high, medium, and low slices summing to 2,913; confusable is empty. Lexical slices are mutually exclusive because each example has one generated lexical-difficulty label; the empty confusable slice reflects the existing criterion/metadata, not forced reassignment.

## NO_MAP diagnostics
The corrected comparison contains NO_MAP n=85 and ANSWERABLE n=2,695, matching forward TEST semantics for diagnostic groups. It reports mean, median, standard deviation, Q1, Q3 for maximum BM25 score, top1-top2 margin, mean Top-5 score, and maximum lexical overlap. These are descriptive retrieval diagnostics only; no threshold or classifier was trained. Post-hoc AUROC using maximum BM25 score, if present, is labeled diagnostic only: `0.6746698679471789`.

## Alternative-set chance effect
The normalized buckets are 1, 2–5, 6–20, 21–100, and >100. The chance table reports actual Hit@K, MRR, and the diagnostic expectation `1 - C(N-G,K)/C(N,K)` with N=17,513 candidate concepts and G acceptable targets. It is not a replacement metric and assumes uniform sampling without replacement.

## Runtime and memory
The forward stratified TEST measurement reports index build `0.13826450000124169`, serialized index size `3828453` bytes, mean query latency `10.716734534838103` ms, median `10.572200000751764` ms, p95 `23.57790000314708` ms, and throughput `93.31200625985338` queries/sec over `2913` Top-100 queries. Peak RSS is `None` because psutil was unavailable. The measurement is approximate and not used for configuration selection. No neural packages, neural models, FAISS, PyTorch, Transformers, sentence-transformers, or dense retrieval components were introduced.

## Artifacts and limitations
Corrected machine-readable artifacts are under `artifacts/experiments/bm25_v1_1/`; publication tables are under `reports/tables/bm25_v1_1/`; figures are under `reports/figures/bm25_v1_1/`. The historical v1 tables are preserved. GEM mappings remain translation-assistance/comparison resources, not clinical equivalence guarantees. V5412 is in TRAIN, so its case-study rank is not a TEST result.
