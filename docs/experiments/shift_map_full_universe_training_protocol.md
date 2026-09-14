# SHIFT-MAP Full-Universe Corrected Training Protocol

**Milestone:** STEP 7.5C — FULL-UNIVERSE SHIFT-MAP RETRAINING + CORRECTED TOP-100 CANDIDATE FREEZE  
**Status:** preregistered before corrected negative mining, corrected ablations, and corrected fine-tuning  
**Namespace:** `shift_map_full_universe`  
**Target universe:** `terminology_universe_v2`, forward ICD-10-CM FY2018, 71,704 targets  
**Evaluator:** `retrieval_evaluator_v3`

## 1. Historical protocol recovered from committed Step 7/7.2/7.3 artifacts

The committed historical protocol and implementation establish the following semantics. Historical outputs and caches are legacy evidence only and are not corrected scientific inputs.

- Gradient updates use only forward stratified TRAIN examples.
- Forward stratified DEV is used for ablations, checkpoint/epoch selection, and learning-rate selection.
- Forward TEST, family-held-out partitions, and backward partitions remain untouched until a formal TEST lock exists.
- Pairwise training excludes `COMBINATION`, `COMBINATION_WITH_ALTERNATIVES`, and `NO_MAP`.
- The historical optimizer is AdamW with weight decay `0.01`, maximum sequence length `64`, temperature `0.05`, maximum gradient norm `1.0`, maximum `3` epochs, effective source batch `32`, and deterministic seeded execution. The historical implementation records FP32 and a linear warmup/scheduler path; exact scheduler behavior is revalidated in the corrected runner rather than inferred from old results.
- The historical learning-rate search evidence contains `5e-6`, `1e-5`, and `2e-5`; this is the frozen corrected LR grid.
- Historical model initialization is BioLORD-2023, but all historical SHIFT-MAP checkpoints are rejected for corrected training.

Historical selection artifacts used a legacy ranking/candidate universe and a historical tie-break definition. They are not reused for corrected selection.

## 2. Corrected immutable foundation

Every corrected training relation, negative, embedding, ranking, and candidate artifact must bind to:

- terminology universe: `terminology_universe_v2`;
- forward target count: `71,704`;
- forward retrieval corpus hash: `32f572abeb2ff4cb003145216fa83bfd9984fab96579205f432993aa9c550464`;
- forward code hash: `8ca0d0cf0213581645fe64d9c1e599552a2fb207eb740f32ca09ffeaaf9f8e26`;
- base model: `FremyCompany/BioLORD-2023`;
- immutable base revision: `167aab527b238a50ca65224e6319215d2ff4fc9f`;
- native Sentence-Transformers pooling, L2 normalization, maximum length `64`;
- frozen BM25 parameters `k1=2.0`, `b=0.75`.

No historical 17,513-target or 11,690/11,689-target artifacts may enter corrected training or candidate generation.

## 3. Training population and positive policies

Training eligibility is source-level and source-balanced. Each eligible source contributes at most one anchor update per epoch; multi-target mappings are not expanded into independent source rows.

- **P0:** `SINGLE_EXACT` only.
- **P1:** `SINGLE_EXACT` + `SINGLE_APPROXIMATE`.
- **P2:** P1 + `ALTERNATIVE`.

For every eligible source, all benchmark-valid target identities remain in its protected positive set. For `ALTERNATIVE`, every valid alternative target represented by the benchmark remains positive and can never be mined as a negative. `COMBINATION`, `COMBINATION_WITH_ALTERNATIVES`, and `NO_MAP` are excluded from pairwise gradient updates, but retained for evaluation and candidate generation.

## 4. Corrected negative strategies

All strategies are regenerated from scratch against all 71,704 forward targets and use only TRAIN gold for exclusion. DEV/TEST relevance judgments and outcomes are never used in mining decisions.

- **N1_RANDOM:** deterministic uniform sample of four non-positive targets from the corrected full universe.
- **N2_LEXICAL_HARD:** the highest-ranked eligible non-positive target from corrected BM25 (`k1=2.0`, `b=0.75`), with deterministic ordering.
- **N3_DENSE_HARD:** the highest-ranked eligible non-positive target from frozen zero-shot BioLORD-2023 against the corrected 71,704-target cache.
- **N4_SAME_FAMILY_HARD:** one non-positive target in a legitimate ICD-10 family/code-prefix group represented by a valid positive, ordered by frozen zero-shot dense ranking; if unavailable, record the strategy as unavailable for that source and use no silent semantic substitute.
- **N5_MIXED:** exactly one corrected lexical, dense, and same-family component where available, deduplicated and deterministically filled from corrected random negatives to four where possible. Component provenance is retained per row.

For every row, retain `source_code`, `benchmark_id`, `negative_type`, `target_code`, universe count/hash, and applicable mining model/config hash. Required audits: zero positive-negative collisions; no forbidden duplicates; every negative exists in the full corpus; deterministic regeneration; TRAIN-only gold boundary; and zero legacy-universe hashes.

## 5. Losses

Only source-to-target training is used; no symmetric target-to-source objective is added.

- **L1 masked single-positive InfoNCE:** one deterministic positive is the numerator term. All valid positives are masked from negative status.
- **L2 set-positive InfoNCE:** all valid positives represented in the candidate set enter a stable, count-normalized log-sum-exp numerator; all valid positives are protected from negative status. Loss normalization is source-balanced.

Synthetic hand-computed tests must verify positive masking, multi-positive numerator membership, stable computation, direction, and source-balanced normalization.

## 6. Frozen corrected DEV-only configuration rule

Before corrected fine-tuning outcomes are inspected, configurations are selected lexicographically on **FORWARD STRATIFIED DEV**, population `P_ORDINARY_ANSWERABLE`, in this exact order:

1. maximize `Hit@100`;
2. maximize `P_COMPLEX CompleteScenarioRetrieval@100`;
3. maximize `Hit@10`;
4. maximize `MRR`;
5. maximize `Hit@1`;
6. minimize measured inference latency as the final tie-break only.

No TEST, family-held-out, backward, representation-drift, hubness, or posthoc seed result may influence configuration selection.

Staged order is fixed:

1. positive-policy ablation: P0, P1, P2;
2. loss ablation: L1 versus L2 using the selected policy;
3. negative-strategy ablation: every valid N1–N5 strategy;
4. LR ablation over `5e-6`, `1e-5`, `2e-5`;
5. epoch/checkpoint selection through the historical maximum of 3 epochs.

Each ablation uses a fixed development seed, preferably seed `42`, and all non-ablated variables remain fixed. No search expansion is permitted after observing corrected DEV outcomes.

## 7. Final seeds and canonical policy

After the configuration freeze, independently train seeds `17`, `42`, and `2026`, each initialized directly from the original pinned BioLORD revision. No ablation checkpoint, other final seed, historical SHIFT-MAP checkpoint, or old-universe artifact may initialize a final run.

Seed `17` is predeclared canonical. It is not selected by DEV or TEST performance. It may be replaced only for a documented technical failure such as checkpoint corruption, NaN training, hardware interruption, or unrecoverable artifact failure.

## 8. TEST lock and post-lock evaluation

Create and hash the corrected SHIFT-MAP TEST lock before any corrected SHIFT-MAP TEST evaluation. The lock binds the terminology hashes, evaluator, base revision, frozen config hash, final seeds, canonical seed 17, epoch rule, candidate K=100, source provenance, and historical TEST exposure disclosure. After lock creation, no configuration changes are permitted.

Only after the lock: evaluate zero-shot BioLORD and final seeds on forward stratified TEST, then run the preregistered structural, family-held-out, backward-transfer, lexical, drift, hubness, NO_MAP, bootstrap, BM25, and candidate-K diagnostics. Diagnostics do not alter selection.

## 9. Candidate freeze and Step 8 boundary

Generate new Top-100 candidates for every source in forward TRAIN/DEV/TEST against all 71,704 targets, in a new corrected namespace. Every source has exactly 100 unique ranks and target identities. `candidate_is_gold` is evaluation-only and is excluded from Step 8 model inputs.

The canonical retriever is corrected SHIFT-MAP seed 17. Step 8 may consume only source description and target description unless a later preregistered method explicitly introduces additional fields. Step 8 is not run in this milestone.
