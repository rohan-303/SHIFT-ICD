# SHIFT-ICD Step 8 — SHIFT-MAP v2 Cross-Encoder Report

**Status: BLOCKED — compute feasibility under selected CPU-only mode**
**Repository:** `C:\Users\rohan\SHIFT-ICD`
**Experiment version:** `2.0`
**Report evidence level:** implemented + technically smoke-tested; scientific reranking results not executed.

## 1. Repository state and frozen contract

- Branch: `main`.
- Starting HEAD: `e6aaea5ca7a65c748c4794bffafdd165448db9a`.
- Working tree was clean at preflight.
- Raw, canonical, benchmark, BM25, dense_v1_1, SHIFT-MAP v1.3, and Step 7.4 validators passed.
- Pytest before Step 8 changes: `67 passed`.
- Frozen retriever: corrected SHIFT-MAP v1.3 L2, seed 17, epoch 3.
- Frozen candidate K: `100`.
- Retriever checkpoint hash: `7a859ff478d801f98a17fc966cb0ae81ba60362725add2735d91c7e01f4fe1d`.
- Target corpus hash: `a394fe6a0055b13e2df4eaeb4ac4b739e7d8d54277adb02f9bdecff67ad80a3a`.
- Candidate files were not modified.

## 2. Coverage denominator reconciliation

Canonical Step 8 coverage uses all forward ordinary sources (`SINGLE_EXACT`, `SINGLE_APPROXIMATE`, `ALTERNATIVE`) as denominator, including retrieval misses. A source is covered when at least one complete valid gold target occurs in its frozen top-K candidates. `NO_MAP` and combination mappings are excluded from ordinary coverage; they are not silently treated as ordinary misses.

| Split | All | Ordinary denominator | Gold-present/eligible | Misses | Combinations | NO_MAP | K=100 coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 10,197 | 9,434 | 9,343 | 91 | 468 | 295 | 0.990354 |
| DEV | 1,457 | 1,348 | 1,331 | 17 | 67 | 42 | 0.987389 |
| TEST | 2,913 | 2,695 | 2,669 | 26 | 133 | 85 | 0.990353 |

The historical `0.9908062234794908` equals `1401/1414` and comes from a different filtered population. It is not the canonical Step 8 end-to-end denominator. The canonical TEST miss rate is `26/2695 = 0.0096474954`, reported previously as `0.009647`.

Artifact: `artifacts/experiments/shift_map_v2/candidate_coverage_reconciliation.json`.

## 3. Candidate training population

- TRAIN reranker-eligible sources: `9,343`.
- TRAIN excluded ordinary retrieval misses: `91`.
- TRAIN `NO_MAP` excluded from ranking loss: `295`.
- TRAIN combinations excluded from ordinary ranking loss: `468`.
- DEV eligible: `1,331`; DEV misses: `17`; DEV NO_MAP: `42`; DEV combinations: `67`.
- TEST eligible: `2,669`; TEST misses: `26`; TEST NO_MAP: `85`; TEST combinations: `133`.
- Alternative valid-positive sets remain sets; no valid alternative is used as a negative.
- Primary text input excludes codes, ranks, scores, mapping metadata, family metadata, split identifiers, and `candidate_is_gold`.

## 4. Cross-encoder provenance

- Model ID: `ncbi/MedCPT-Cross-Encoder`.
- Exact revision: `71caf65d4927987813984f54c284405a13fcca49`.
- Architecture: `BertForSequenceClassification`.
- Measured parameter count: `109,483,009`.
- Hidden size: `768`.
- Layers/heads: `12/12`.
- Classifier output: one scalar logit.
- Model maximum positions: `512`.
- Loaded dtype: FP32.
- Model-card license metadata: `other`, `public-domain`, with repository LICENSE authoritative; this is not a legal determination.
- Weight file size: `437,998,062` bytes.
- Provenance: [Hugging Face model card](https://huggingface.co/ncbi/MedCPT-Cross-Encoder), [MedCPT publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC10627406/).
- Weights remain in the external Hugging Face cache and are not committed.

Artifact: `artifacts/experiments/shift_map_v2/model_provenance.json`.

## 5. Input length audit

TRAIN+DEV candidate pairs audited: `1,165,400`.

| Statistic | Value |
|---|---:|
| Maximum | 79 tokens |
| P95 | 42 |
| P99 | 54 |
| >64 | 1,842 / 1,165,400 = 0.1581% |
| >96 | 0 |
| >128 | 0 |

Frozen Step 8 max length: `96`. No pair in the audited TRAIN+DEV population is truncated at 96.

Artifact: `artifacts/experiments/shift_map_v2/sequence_length_audit.json`.

## 6. Technical real-model smoke test

One DEV source and all 100 frozen candidates were scored on CPU. MedCPT loaded successfully, emitted one scalar score per pair, produced a reranking, and preserved exact candidate membership. The one-source score was not promoted to a DEV result.

## 7. Zero-shot MedCPT DEV

**NOT COMPUTED.** The full 145,700-pair CPU run was stopped after more than 10 minutes without producing a valid artifact. A prior GPU attempt was stopped at observed 85°C under the limited-GPU safety policy. No zero-shot DEV metric is reported.

- Hit@1/5/10/25/50/100: `NOT COMPUTED`.
- MRR/NDCG@5/NDCG@10: `NOT COMPUTED`.
- Hit@100 preservation: technically verified on the one-source smoke only; full-population validation pending.

## 8. Objective, negative, learning-rate, initialization, and final configuration

The following were preregistered but **NOT EXECUTED**:

- CE-BCE versus CE-LISTWISE objective ablation.
- TOP-RANK HARD versus MIXED-RANK versus RANDOM-WITHIN-CANDIDATE.
- Learning rates `1e-5`, `2e-5`, `3e-5`.
- Optional BiomedBERT/PubMedBERT control.
- Final seeds `17`, `42`, `2026`.
- Epoch selection.
- Test lock.
- TEST evaluation.

Therefore:

- DEV-selected objective: `NOT COMPUTED`.
- DEV-selected negative strategy: `NOT COMPUTED`.
- DEV-selected learning rate: `NOT COMPUTED`.
- Initialization-control result: `NOT PERFORMED`.
- Final configuration: `NOT FROZEN`.
- Test-lock hash: `NOT CREATED`.
- Canonical seed: `NOT SELECTED`.

Initial preregistered settings remain AdamW, weight decay 0.01, maximum 3 epochs, warmup 0.10, linear schedule, max grad norm 1.0, and FP16 only if stable; these are design values, not executed results.

## 9. TEST exposure

Track A TEST was historically exposed during defective Step 7 evaluation, corrected Step 7.2/7.3 evaluation, and Step 7.4 characterization. Step 8 did not use TEST for any configuration decision. No Step 8 TEST result was generated.

## 10. TEST metrics and comparisons

All fields below are `NOT COMPUTED` because the Step 8 model was not trained or evaluated:

- Forward TEST per-seed results: `NOT COMPUTED`.
- Three-seed mean/SD: `NOT COMPUTED`.
- Canonical v2 TEST: `NOT COMPUTED`.
- Frozen L2 comparison: `NOT COMPUTED`.
- Paired v2-minus-L2 Hit@1/Hit@5/Hit@10/MRR/NDCG@10: `NOT COMPUTED`.
- Hit@100 preservation on full TEST: `PENDING`; candidate membership contract itself remains unchanged.
- Oracle ceiling/headroom recovery: `NOT COMPUTED`.
- Conditional reranker and end-to-end system performance: `NOT COMPUTED`.
- Lexical slices: `NOT COMPUTED`.
- Mapping-kind slices: `NOT COMPUTED`.
- Positive-cardinality slices: `NOT COMPUTED`.
- Combination ChoiceListRecall and CompleteScenarioRetrieval: `NOT COMPUTED`.
- Family-held-out: `NOT COMPUTED`.
- Backward transfer: `NOT PERFORMED`.
- NO_MAP score diagnostics: `NOT COMPUTED`.
- Score-fusion DEV ablation: `NOT COMPUTED`.
- Error-analysis samples: `NOT COMPUTED`.

## 11. Compute status

- CPU-only mode selected by user after thermal stop.
- Threads capped at 2.
- Full DEV zero-shot scoring was not feasible within the bounded execution window.
- No cross-encoder training was launched.
- No TEST process was launched.
- No GPU process remains active.
- Model/checkpoint size: 437,998,062-byte Hub weight file; no trained checkpoint exists.
- Runtime/peak memory tables: only technical smoke data exists; full reranker runtime is `NOT COMPUTED`.

## 12. Artifacts and quality gates

Created:

- `docs/experiments/shift_map_v2_protocol.md`
- `artifacts/experiments/shift_map_v2/model_download_manifest.json`
- `artifacts/experiments/shift_map_v2/model_provenance.json`
- `artifacts/experiments/shift_map_v2/candidate_coverage_reconciliation.json`
- `artifacts/experiments/shift_map_v2/sequence_length_audit.json`
- `artifacts/experiments/shift_map_v2/zero_shot_cross_encoder_dev_status.json`
- `src/shift_icd/reranking/shift_map_v2.py`
- `src/shift_icd/reranking/__init__.py`
- `scripts/run_shift_map_v2.py`
- `tests/unit/test_shift_map_v2_reranking.py`

Focused Step 8 tests: `7 passed`.

Full pre-Step-8 tests: `67 passed`.

Final post-edit full-suite/lint/type gates after the latest runner edits: `PENDING`; this blocked report intentionally does not claim a final commit.

## 13. Outcome classification

No V2-C1–V2-C6 scientific outcome is supported because the reranker experiment did not execute. V2-C7 does not apply: the candidate contract and technical smoke path were valid. The correct milestone status is:

```text
BLOCKED_COMPUTE_FEASIBILITY
```

## 14. Main supported findings

1. The frozen candidate contract is internally consistent.
2. The canonical ordinary denominator is 2,695 TEST sources, not the historical 1,414-source filtered population.
3. Canonical TEST K=100 coverage is 0.990353 with 26 irrecoverable ordinary retrieval misses.
4. MedCPT loads correctly at the pinned revision and has a one-scalar classification head.
5. Max length 96 covers all audited TRAIN+DEV pairs without truncation.
6. One-source technical reranking preserved all 100 candidate members.

## 15. Negative findings and limitations

1. Full CPU-only MedCPT scoring is not practical within the current bounded runtime.
2. A GPU run reached 85°C and was stopped; no further GPU run was made.
3. No zero-shot full DEV result exists.
4. No objective, negative-strategy, learning-rate, seed, or epoch selection exists.
5. No test lock exists and no TEST result exists.
6. No scientific improvement/degradation claim is supported.
7. No model checkpoint was trained or committed.

## 16. Recommendation for Step 9

```text
BLOCK
```

Hierarchy-aware SHIFT-MAP v3 is **not safe to begin**. Step 8 must first complete on a safe, bounded compute environment with full DEV-only selection, immutable test lock, and post-lock TEST evaluation. No hierarchy-aware reranking, GNN, calibration, conformal prediction, or routing was implemented.
