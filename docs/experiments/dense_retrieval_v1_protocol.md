# Zero-Shot Dense Retrieval v1 Protocol

## Status

Pre-registered before TEST evaluation. Experiment version: `dense_v1`. Benchmark: Track A v1.0. No fine-tuning is permitted in this milestone.

## Research question

Can zero-shot pretrained biomedical semantic representations outperform the frozen lexical BM25 reference under ICD terminology shift, especially on low lexical-overlap examples?

## Hypotheses

- **H1:** Biomedical dense representations improve retrieval over BM25 on concept mappings with weak lexical overlap.
- **H2:** Dense gains are stronger on `LEXICAL_LOW` than on `LEXICAL_EXACT` or `LEXICAL_HIGH`.
- **H3:** Dense retrieval recovers semantic alternatives missed by BM25.
- **H4:** Dense retrieval alone may struggle to retrieve all components of combination mappings.
- **H5:** A modern general embedding model may rival or exceed biomedical-specialized models; specialization is not assumed to win.
- **H6:** BM25 and dense retrieval make partially complementary errors, motivating hybrid candidate generation.

## Frozen comparator

BM25 `bm25_v1_1`, Q1 long-description-only representation, internal exact inverted index, `k1=2.0`, `b=0.75`, deterministic target-code tie-breaking. Forward stratified TEST reference: Hit@10 `0.781447`, MRR `0.651011`.

## Declared models

1. SapBERT: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`.
2. BioLORD-2023: `FremyCompany/BioLORD-2023`.
3. MedCPT Query Encoder: `ncbi/MedCPT-Query-Encoder`, used symmetrically for source and target descriptions. The Article Encoder is not used in the primary experiment.
4. General control: `Qwen/Qwen3-Embedding-0.6B`; 4B and 8B variants are excluded.

Exact revisions, licenses, and implementation details are recorded in `docs/models/dense_retrieval_v1.md` and machine-readable manifests.

## Inputs

Source and target authoritative long descriptions are used. If long description is unavailable, the authoritative short description is used. Only Unicode normalization, edge trimming, and repeated-whitespace normalization are allowed. Codes, GEM flags, mapping kinds, scenarios, choice lists, lexical labels, source families, target families, and gold metadata are excluded from model inputs.

Qwen queries use exactly this predeclared instruction:

> Retrieve the ICD diagnosis concept that is semantically equivalent or the closest valid cross-version mapping to the source diagnosis description.

Documents receive no instruction or gold metadata.

## Encoding and retrieval

Models run in evaluation mode with gradients disabled. Embeddings are L2-normalized and retrieved by exact normalized dot product against the complete target corpus. Top-100 ties are ordered by descending similarity, then ascending target code. No FAISS or approximate-nearest-neighbor index is used.

The primary maximum length is 64 tokens if it covers at least 99.9% of each model-tokenized description corpus; otherwise the smallest defensible value among 128 or the model limit is selected and recorded before TEST. Truncation counts are reported per tokenizer.

## Evaluation order and selection

1. Implement and unit-test adapters using synthetic fixtures.
2. Build target embeddings.
3. Evaluate all declared models on forward stratified DEV.
4. Select the future-development dense encoder using only forward stratified DEV, lexicographically:
   - non-combination answerable Hit@100;
   - CompleteScenarioRetrieval@100 where applicable;
   - Hit@10;
   - MRR.
5. Freeze `selection.json`.
6. Evaluate all declared models on the frozen TEST populations.
7. Evaluate one fixed BM25 + selected-dense RRF hybrid with RRF constant 60.

The best observed TEST baseline and DEV-selected future-development model are reported separately. TEST performance cannot alter selection.

## Populations and gold semantics

All results retain explicit direction, split protocol, partition, benchmark version, sample population, and counts. Required populations are forward stratified DEV, forward stratified TEST, forward family-held-out TEST, backward stratified TEST, and backward family-held-out TEST. Existing complex-gold semantics are used unchanged: ordinary Hit@K/MRR for single and alternative mappings; ChoiceListRecall@K and CompleteScenarioRetrieval@K for combinations; undefined ordinary recall for NO_MAP.

## Statistical and diagnostic analysis

Report paired bootstrap 95% CIs for dense-minus-BM25 Hit@10, Hit@100, and MRR. If formal multiple-comparison p-values are reported, Holm-Bonferroni correction is applied across predeclared dense-vs-BM25 Hit@10 comparisons. McNemar tests are optional and labeled exploratory. Complementarity and oracle union coverage are descriptive and non-deployable.

NO_MAP similarity distributions and AUROC, if computed, are post-hoc diagnostics only. No threshold, classifier, calibration, conformal predictor, or routing policy is trained or selected.

## Caching and provenance

Target matrices are cached under ignored `artifacts/embeddings/dense_v1/` with tracked manifests containing model ID/revision, license, corpus hash, text representation, dimension, dtype, normalization status, row order, and SHA-256. Model weights remain in the standard Hugging Face cache or ignored local cache and are not redistributed through Git.

## Limitations

Zero-shot embedding similarity is not clinical truth or a validated conversion rule. CMS GEM approximate mappings remain translation-assistance resources. Dense models may encode information from pretraining resources overlapping concept terminology; this is documented, not treated as evidence of clinical equivalence. No fine-tuning, hard-negative training, hierarchy model, reranking model, or selective decision policy is part of this milestone.
