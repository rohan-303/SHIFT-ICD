# BM25 lexical retrieval baseline protocol v1

**Status:** preregistered before final test evaluation
**Experiment:** `bm25_v1`
**Benchmark:** Track A `1.0`
**Seed:** `20260829`

## Research question

How much of Track A ICD cross-version mapping can be solved by lexical retrieval alone, and where does lexical retrieval fail across mapping complexity and terminology drift?

## Scope and exclusions

This experiment uses only CPU lexical retrieval. It does not use gold mapping metadata in query or document text. It does not use neural embeddings, biomedical language models, synonym resources, UMLS, spelling correction, hierarchy models, reranking, calibration, conformal prediction, or selective routing.

The benchmark is frozen. No benchmark data, gold structure, split, or slice is changed to improve retrieval results.

## Retrieval representation

Q1 is the primary pre-registered representation: the authoritative long description, normalized deterministically. Q2 concatenates short and long descriptions when both are available; identical short/long labels are represented once to avoid accidental duplicate weighting. Missing long descriptions fall back to short descriptions. If both are missing, the candidate remains with an explicit empty-document marker.

Normalization is deliberately conservative:

1. Unicode NFKC normalization.
2. Unicode-aware lowercase.
3. Punctuation and symbols become token boundaries.
4. Alphanumeric medical tokens and digits are preserved.
5. Whitespace is collapsed.
6. No stopword removal, stemming, abbreviation expansion, synonym replacement, spelling correction, or clinical rewriting.

Tokens are deterministic regex tokens containing letters and/or digits. Terms such as `with`, `without`, `due`, `after`, and `following` are retained.

## Candidate corpora

The forward corpus contains each eligible ICD-10-CM target concept exactly once. The backward corpus contains each eligible ICD-9-CM target concept exactly once. Candidate documents are target long descriptions with short-description fallback, independent of GEM row frequency.

## Baselines

- **B0 Random:** seeded random permutation of the complete target corpus; ties and corpus order are deterministic.
- **B1 Exact Label:** normalized exact query/document label matches first, followed by nonmatches ordered deterministically by target code. Gold metadata does not break exact-label ties.
- **B2 Token Overlap:** token Jaccard similarity, then target code ascending.
- **B3 BM25 Default:** Robertson-style BM25 with `k1=1.5`, `b=0.75`.
- **B4 BM25 Dev-Tuned:** grid search over `k1 ∈ {0.8, 1.2, 1.5, 2.0}` and `b ∈ {0.25, 0.50, 0.75, 1.00}` using only forward stratified DEV.

The B4 primary selection metric is non-combination, answerable `Hit@10`. The secondary tie-break is `CompleteScenarioRetrieval@10` on combination/multi-scenario answerable mappings. Ties resolve by higher secondary metric, then lower `k1`, then lower `b`. The selected configuration is frozen before any test evaluation and is used unchanged for all four test protocols and both directions.

## BM25 formula

For query terms `t`, document `D`, and parameters `k1`, `b`:

```text
score(D,Q) = Σ IDF(t) * f(t,D) * (k1 + 1)
             / (f(t,D) + k1 * (1 - b + b * |D| / avgdl))

IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
```

Rankings use score descending and target code ascending. Top 100 candidates are retained.

## Evaluation datasets and K values

DEV selection uses only forward stratified DEV. Frozen evaluation uses:

1. forward stratified TEST;
2. forward family-held-out TEST;
3. backward stratified TEST;
4. backward family-held-out TEST.

All results use `K ∈ {1, 5, 10, 25, 50, 100}`. Training performance is not reported as evidence of generalization.

## Metrics

For answerable non-combination mappings: Hit@K and MRR. For combinations: ChoiceListRecall@K and CompleteScenarioRetrieval@K. Multi-scenario mappings succeed when any complete scenario is recoverable. NO_MAP examples are excluded from ordinary recall and receive score diagnostics only.

Results are reported overall, by mapping kind, lexical difficulty, alternative-set-size bucket, direction, and split protocol. The known forward `V54.12` 533-alternative example is reported separately without changing its gold semantics.

## NO_MAP diagnostics

For mapped and NO_MAP examples, record maximum BM25 score, top1-top2 margin, mean Top-5 score, query token count, and maximum token overlap with any target. Report mean, median, quartiles, and standard deviation. No abstention threshold is selected.

## Uncertainty and error analysis

Primary test metrics include deterministic 95% bootstrap confidence intervals with seed `20260829`. Approximately 100 forward stratified test failures at Top-10 are sampled deterministically and stratified by mapping complexity and lexical difficulty. No unsupported clinical failure labels are assigned.

## Reproducibility and integrity

Every run records the Git commit, benchmark and canonical versions, benchmark manifest hash, configuration hash, Python/package versions, seed, tokenizer/normalization versions, BM25 implementation version, corpus profile, index metadata, runtime measurements, and artifact hashes. Raw, canonical, and benchmark hashes are checked before and after the run. Large rankings and generated figures remain reproducible derived artifacts and are not required to be committed.

## Future dense-retrieval gate

The frozen B4 configuration and its complete metric tables are the lexical reference point. Future dense retrievers must be evaluated on the same frozen examples and splits; no arbitrary improvement threshold is declared in advance.
