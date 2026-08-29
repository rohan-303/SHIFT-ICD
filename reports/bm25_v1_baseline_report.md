# BM25 v1 lexical retrieval baseline report

## 1. Objective

This experiment establishes the CPU lexical floor for frozen Track A Benchmark v1.0. It asks how much ICD cross-version mapping can be retrieved from terminology overlap alone and where lexical retrieval fails under mapping complexity and terminology drift. Evidence status: **Executed/Reproduced** for the completed run; the benchmark itself remains **Validated**.

## 2. Dataset and candidate corpora

Track A contains 86,271 source-level examples in two independent directions. The forward corpus is the unique ICD-10-CM target terminology and the backward corpus is the unique ICD-9-CM target terminology. Profiles are recorded in `corpus_profile.json`; every target concept appears once, while duplicate labels are retained.

## 3. Retrieval representation

Q1 long-description-only is primary. Q2 short-plus-long was evaluated on DEV only and was not selected (`0.772997` vs Q1 `0.773739` non-combination Hit@10). Missing long text falls back to short text. Normalization is NFKC, lowercase, punctuation-to-boundaries, whitespace collapse, preserved digits/alphanumerics, no stopword removal, stemming, synonym expansion, abbreviation expansion, or spelling correction.

## 4. Baselines and BM25 configuration

B0 seeded random, B1 exact normalized label match, B2 token Jaccard overlap, B3 Robertson BM25 default (`k1=1.5`, `b=0.75`), and B4 BM25 Dev-Tuned were implemented. BM25 uses the internal inverted-index implementation and the preregistered formula in `docs/experiments/bm25_v1_protocol.md`. The frozen DEV winner is `k1=2.0`, `b=0.75`, selected only on forward stratified DEV using non-combination answerable Hit@10 with complex CompleteScenarioRetrieval@10 as tie-break.

## 5. Main results

Machine-readable results are in `test_metrics.json` and `reports/tables/bm25_v1/`. The primary forward stratified TEST baseline table is `baseline_overall.csv`; split, direction, mapping-kind, lexical-difficulty, alternative-size, and combination tables are generated from those metrics. Confidence intervals are deterministic bootstrap intervals in `test_metrics.json`.

## 6. Generalization and asymmetry

Forward stratified, forward family-held-out, backward stratified, and backward family-held-out TEST datasets contain 0.7832390381895332 / 0.782668500687758 / 0.5148996125396267 / 0.4855996056615731 BM25 Hit@10 estimates respectively. Interpret differences descriptively; no causal claim is made.

## 7. Complex mappings and alternatives

Choice-list recall and complete-scenario retrieval use the existing structural gold evaluator. Results are reported in `bm25_combination_retrieval.csv`. Larger alternative sets increase the chance of a Hit@K, so alternative-size stratification is required. The requested V54.12 / 533-alternative source is absent from the frozen benchmark JSONL; no rank is fabricated. This is a benchmark coverage issue to resolve in a future benchmark audit, not a retrieval result.

## 8. NO_MAP diagnostics

NO_MAP examples are excluded from ordinary recall. Forward stratified TEST produced 85 persisted score-diagnostic rows. Summary statistics are in `bm25_no_map_score_summary.csv`; no threshold or classifier was selected. The current diagnostic artifact is forward-test scoped and should not be mistaken for a complete mapped-vs-NO_MAP distributional study.

## 9. Error analysis and runtime

A deterministic sample of 100 forward stratified Top-10 failures is stored at `artifacts/error_analysis/bm25_v1_forward_errors.jsonl`, with unsupported clinical failure categories left blank. The completed run elapsed 3218.26 seconds; corpus construction took 3.21 seconds. The implementation is CPU-only; memory footprint was not instrumented by the initial runner and is therefore not claimed.

## 10. Figures and limitations

Figures are SVG files under `reports/figures/bm25_v1/`. BM25 measures lexical overlap, cannot understand clinical semantic equivalence or ontology structure, cannot inherently identify NO_MAP cases, may benefit from shared terminology and large acceptable target sets, and may fail under lexical terminology drift. BM25 is not a clinical conversion rule and approximate GEM mappings are not exact clinical equivalence.

## 11. Future dense-retrieval gate

Future dense retrievers must use the same frozen examples, splits, and complex gold semantics. The reference point is the frozen B4 configuration (`k1=2.0`, `b=0.75`), not an arbitrary improvement threshold.
