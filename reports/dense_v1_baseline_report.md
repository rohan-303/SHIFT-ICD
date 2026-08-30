# SHIFT-ICD Dense Retrieval v1 Baseline Report

## Objective
Zero-shot pretrained dense retrieval was evaluated without fine-tuning against the frozen Track A v1.0 benchmark and BM25 v1.1.

## Frozen protocol and test protection
Model selection used only forward stratified DEV and selected BioLORD-2023 by the preregistered lexicographic criterion. TEST was then evaluated for all predeclared models. Benchmark data, splits, gold semantics, and BM25 parameters were not changed.

## Main forward stratified TEST results

" + pd.DataFrame(overall)[["model", "n", "Hit@1", "Hit@5", "Hit@10", "Hit@25", "Hit@50", "Hit@100", "MRR"]].to_markdown(index=False) + "

## Findings
BioLORD-2023 was the strongest observed forward TEST baseline in this run and also the DEV-selected future encoder. Qwen3-Embedding-0.6B was a strong general-embedding control. MedCPT and SapBERT were not hidden when they underperformed BioLORD on the reported populations. Dense retrieval materially improved forward Hit@10 over frozen BM25 in this evaluation, but this is retrieval success against CMS-derived structural gold, not clinical truth or guaranteed equivalence.

Lexical-low, mapping-kind, combination, backward, family-held-out, complementarity, oracle-union, no-map, runtime, and error-analysis artifacts are stored under `artifacts/experiments/dense_v1/` and `reports/tables/dense_v1/`. NO_MAP similarity values are exploratory post-hoc diagnostics only; no threshold or routing policy was trained.

## Limitations and recommendation
The local cache and GPU environment are machine-specific; exact revisions and hashes must be retained with the manifest. The current analysis does not establish clinical equivalence, calibration, abstention, or robustness beyond these frozen populations. It is safe to proceed to a later fine-tuning milestone only under a new explicit protocol; this milestone itself stops before fine-tuning, reranking, hierarchy modeling, and routing.
