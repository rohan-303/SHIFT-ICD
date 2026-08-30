# SHIFT-MAP v1 Protocol

**Status:** preregistered before Step 7 training
**Experiment:** `shift_map_v1`
**Question:** Can cross-version supervised contrastive adaptation of a biomedical concept encoder improve ICD mapping retrieval beyond zero-shot BioLORD while retaining lexical-low, family-held-out, complex, and reverse-direction generalization?

## Frozen boundary

Gradient updates use only forward stratified TRAIN examples from Track A benchmark v1.0. Forward stratified DEV is used for every ablation, checkpoint, learning-rate, and seed decision. All TEST partitions and backward examples remain untouched until `test_lock.json` exists. The zero-shot BioLORD initialization is `FremyCompany/BioLORD-2023`, revision `167aab527b238a50ca65224e6319215d2ff4fc9f`, MPNet-base, native Sentence-Transformers pooling, sequence length 64.

No cross-encoder, hierarchy reranker, GNN, cardinality head, calibration, conformal method, or ACCEPT/REVIEW/ABSTAIN routing is included.

## Hypotheses

- H1: TRAIN-only supervised adaptation improves forward retrieval over zero-shot BioLORD.
- H2: hard negatives improve ranking over random negatives.
- H3: dense and lexical hard negatives provide different signals.
- H4: mixed hard negatives outperform any one source on DEV.
- H5: approximate mappings add useful supervision beyond exact-only.
- H6: alternatives are useful multiple valid positives, never false negatives.
- H7: gains may be largest on approximate, alternative, and lexical-low cases.
- H8: combination improvement is transfer because combinations are excluded from pairwise training.
- H9: forward training may not transfer symmetrically to backward mappings.

These hypotheses are frozen before TEST evaluation.

## Positive policies

- **P0:** `SINGLE_EXACT` only.
- **P1:** `SINGLE_EXACT` + `SINGLE_APPROXIMATE`.
- **P2:** P1 + `ALTERNATIVE`; each source retains its complete valid target set.

`COMBINATION` and `COMBINATION_WITH_ALTERNATIVES` are excluded from simple pairwise gradient supervision. `NO_MAP` is excluded because it has no positive target. All remain in evaluation.

Training is source-balanced: each eligible source contributes at most one anchor step per epoch. For multi-positive sources, deterministic seed/epoch/source selection cycles through valid positives; the number of valid targets does not multiply gradient steps.

## Objectives

Only source-to-target loss is used. No symmetric target-to-source loss is implemented.

**L1 masked single-positive InfoNCE:** choose one deterministic positive `p_i` for source `i`. With candidate set `C_i` containing the selected positive and explicit/in-batch candidates:

`L_i = -log_softmax(sim(q_i, C_i) / tau)[p_i]`.

Every candidate in the source's complete valid-positive set is masked from negative status; collisions are asserted impossible for explicit negatives. The selected positive is the only numerator term.

**L2 set-positive InfoNCE:** all valid positives represented in the candidate set are numerator terms:

`L_i = -(logsumexp(sim(q_i, P_i)/tau) - logsumexp(sim(q_i, C_i)/tau))`.

The numerator is count-normalized by subtracting `log(|P_i|)` so sources with larger represented positive sets do not receive artificial objective scaling. Known positives absent from the current candidate set cannot be scored but remain protected by the full-positive mask.

## Negatives

All mining is deterministic and uses TRAIN gold, the complete ICD-10-CM terminology corpus, frozen BM25, frozen zero-shot BioLORD, and source-family structure only.

- N1 random: deterministic uniform non-gold targets.
- N2 lexical: highest frozen BM25-ranked non-gold targets.
- N3 dense: highest frozen BioLORD-ranked non-gold targets.
- N4 same-family: same ICD-10-CM family as a valid target, excluding all gold; frozen dense order breaks ties; deterministic fallback is recorded.
- N5 mixed: exactly one lexical-hard, one dense-hard, and one same-family-hard, deduplicated and filled with deterministic random negatives to four where possible.

Negative collision validation must report zero remaining gold collisions.

## Fixed optimization

AdamW, learning rate `1e-5`, weight decay `0.01`, temperature `0.05`, maximum 3 epochs, linear scheduler with 10% warmup, maximum gradient norm `1.0`, maximum sequence length `64`, FP16 when numerically stable otherwise FP32. Effective source batch size is `32`, achieved by gradient accumulation based on available GPU memory. Micro-batch is a resource setting, never a quality-selected setting.

## Ablations and selection

Positive/loss DEV ablation uses random negatives and identical seed/configuration:
P0+L1, P1+L1, P2+L1, P2+L2.

The selected positive/loss pair then compares N1, N2, N3, and N5. The selected negative strategy is followed by one learning-rate refinement over `5e-6`, `1e-5`, and `2e-5`.

Every selection uses forward stratified DEV only and the frozen lexicographic criterion:
1. non-combination answerable Hit@100;
2. CompleteScenarioRetrieval@100;
3. Hit@10;
4. MRR;
5. LEXICAL_LOW Hit@100 only if still tied.

Checkpoints are selected after each epoch using the same criterion. Three final seeds are fixed before final training: `17`, `42`, and `2026`. The canonical seed is selected by DEV before TEST evaluation.

## Test lock and evaluation

Before any TEST access, `artifacts/experiments/shift_map_v1/test_lock.json` records the complete configuration, seed list, canonical seed, config hash, Git state, and timestamp. After locking, no configuration changes are permitted.

Final evaluation covers forward stratified TEST, forward family-held-out TEST, backward stratified TEST, and backward family-held-out TEST. Backward results are forward-trained zero-shot-transfer results, not backward-trained results. Combinations use ChoiceListRecall and CompleteScenarioRetrieval; ordinary mappings use Hit@K and MRR.

## Release policy

Checkpoints are stored under Git-ignored `artifacts/models/shift_map_v1/`. Only metadata, hashes, configurations, metrics, manifests, and reproducibility records are committed. Fine-tuned BioLORD-derived weights are not uploaded or redistributed pending licensing review.
