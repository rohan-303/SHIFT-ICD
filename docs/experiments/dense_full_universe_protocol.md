# Full-Universe Dense Retrieval Protocol v2

## Status and scope

This protocol is frozen before corrected full-universe DEV execution. It reproduces the valid `dense_v1_1` lineage while changing only the target universe and evaluation contract. Historical dense artifacts remain legacy evidence and are not reused.

- Git authority: local `SHIFT-ICD` repository
- Required evaluator: `retrieval_evaluator_v3`
- Corrected terminology: `terminology_universe_v2`
- Dense output namespace: `artifacts/experiments/dense_full_universe_v2/`
- Target embedding namespace: remote `<remote_root>/dense_full_universe_v2/`
- Top-K: exact full-matrix retrieval through K=100
- Seed: `20260830` where deterministic sampling/metadata requires a seed
- No fine-tuning, SHIFT-MAP training, hard-negative mining, cross-encoder, or Step 8 generation in this milestone

## Frozen models

| Name | Model ID | Revision | Kind | Dimension |
|---|---|---|---|---:|
| SapBERT | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | `090663c3ae57bf35ffe4d0d468a2a88d03051a4d` | Transformers CLS | 768 |
| BioLORD-2023 | `FremyCompany/BioLORD-2023` | `167aab527b238a50ca65224e6319215d2ff4fc9f` | Sentence Transformer | 768 |
| MedCPT dense | `ncbi/MedCPT-Query-Encoder` | `d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc` | Transformers CLS | 768 |
| Qwen embedding | `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | Sentence Transformer / Qwen | 1024 |

A model is valid only if the exact immutable revision can be downloaded and its resolved configuration/tokenizer is recorded. If a revision cannot be verified, that model is `MODEL_PROVENANCE_UNRESOLVED` and is not substituted.

## Text and inference semantics

- Input text: source/target authoritative long description; fallback to authoritative short description only when long description is unavailable.
- Text preprocessing: Unicode NFKC, edge trimming, repeated-whitespace normalization only.
- Codes, GEM flags, mapping kinds, scenarios, gold targets, lexical labels, family labels, and other metadata are excluded from model text.
- Qwen query formulation: `Instruct: <instruction>\nQuery:<cleaned source text>` with instruction: `Retrieve the ICD diagnosis concept that is semantically equivalent or the closest valid cross-version mapping to the source diagnosis description.` Documents receive no instruction.
- Max sequence length: 64 tokens, unless the pre-TEST token audit proves the frozen coverage rule requires 128 or the model limit. The selected value must be frozen before TEST.
- Padding: tokenizer-native batch padding, with empirical batch-1/equal-length parity checks before scientific scoring.
- Model mode: evaluation mode, gradients disabled, no generation.
- Precision: actual runtime precision must be recorded. Existing lineage emitted float32 embeddings with inference autocast disabled; no silent precision change is allowed.
- Pooling: SapBERT and MedCPT use the committed CLS-pooling path; BioLORD uses its SentenceTransformer native pooling; Qwen uses its SentenceTransformer native pooling.
- Normalization: L2-normalize embeddings and record the actual norm check.
- Similarity: exact normalized dot product (`matrix @ query`); deterministic tie break by descending score then ascending target code.
- Approximate ANN/FAISS is prohibited for corrected scientific ranking.

## Evaluation order and lock

1. Verify local repository and corrected corpora.
2. Recover and verify exact model provenance/configuration.
3. Rebuild all target embeddings from the corrected full universe.
4. Evaluate every valid model on forward stratified DEV only.
5. Evaluate corrected fixed-RRF on DEV using corrected BM25 plus each appropriate dense ranking, with `rrf_k=60`; RRF is a hybrid comparison baseline.
6. Freeze neural reselection using only forward stratified DEV and the rule below.
7. Create and hash the corrected TEST lock.
8. Evaluate frozen baselines on TEST, family-held-out, and backward transfer splits.
9. Perform paired bootstrap against corrected BM25; descriptive only.

No corrected TEST metrics, rankings, failure slices, or model comparisons may be inspected before the lock exists.

## Frozen neural reselection rule

Use forward stratified DEV and `P_ORDINARY_ANSWERABLE` as the primary population. Select only among pretrained dense neural encoders, never RRF, lexicographically:

1. maximize Hit@10;
2. maximize MRR;
3. maximize Hit@100;
4. maximize `P_COMPLEX CompleteScenarioRetrieval@100`;
5. minimize mean query latency as a tie-break only.

The selected neural encoder becomes the initialization candidate for STEP 7.5C. Selection is not based on historical winners or TEST values.

## Metric contract

Use `docs/interfaces/full_universe_dense_evaluation_contract.md` and evaluator v3. Ordinary Hit@K/MRR uses only `SINGLE_EXACT`, `SINGLE_APPROXIMATE`, and `ALTERNATIVE`. Combination populations use ChoiceListRecall@K and CompleteScenarioRetrieval@K separately. NO_MAP is diagnostic only; no abstention threshold is selected.

## Corrected universes

- Forward ICD-10-CM FY2018: 71,704 concepts; retrieval corpus hash `32f572abeb2ff4cb003145216fa83bfd9984fab96505f432993aa9c550464`.
- Backward ICD-9-CM Version 32: 14,567 concepts; retrieval corpus hash `dcfe26da33d422a2395081fffaaf83e29d2f557d33458ade392d83b14e171b3b`.

Every embedding manifest must include model ID/revision, tokenizer revision, corpus hash, target count, target-order hash, text formulation, max length, pooling, normalization, similarity, dtype, embedding shape, array hash, Git commit, and runtime versions.

## Historical exclusion

The old `dense_v1`/`dense_v1_1` matrices, indices, target lists, rankings, candidates, negatives, and checkpoints were built for the GEM-observed target universe and are labeled `LEGACY_GEM_OBSERVED_TARGET_UNIVERSE`. They must not be reused for corrected metrics.
