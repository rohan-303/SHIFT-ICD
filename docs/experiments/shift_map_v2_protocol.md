# SHIFT-ICD Step 8 — SHIFT-MAP v2 Protocol

## Status and scope

This protocol freezes the first biomedical cross-encoder reranking stage over the immutable SHIFT-MAP v1.3 corrected L2 top-100 candidate files. It permits candidate reordering only. It does not permit candidate addition/removal, retriever changes, hierarchy-aware features, GNNs, cardinality prediction, calibration, conformal prediction, or routing.

Step 8 is not a pristine first-look TEST experiment: Track A TEST was historically exposed during defective Step 7 evaluation, corrected Step 7.2/7.3 evaluation, and Step 7.4 characterization. All model, objective, negative-sampling, length, learning-rate, epoch, seed, and initialization decisions use TRAIN plus forward-stratified DEV only. TEST is evaluated only after `test_lock.json` is written.

## Frozen retrieval contract

- Experiment version: `2.0`
- Stage: `cross_encoder_reranker`
- Candidate generator: corrected SHIFT-MAP v1.3 L2, seed 17, epoch 3
- Candidate K: 100
- Candidate files: `artifacts/candidates/shift_map_v2/`
- Candidate manifest: `artifacts/candidates/shift_map_v2/manifest.json`
- Retriever checkpoint hash: `7a859ff478d801f98a17fc966cb0ae81ba60362725add2735d91c7e01f4fe1d`
- Target corpus hash: `a394fe6a0055b13e2df4eaeb4ac4b739e7d8d54277adb02f9bdecff67ad80a3a`

The canonical checkpoint hash is also recorded in the frozen candidate manifest; any discrepancy is a hard stop.

## Canonical candidate-coverage definition

The primary ordinary coverage population is every forward benchmark source whose mapping kind is one of `SINGLE_EXACT`, `SINGLE_APPROXIMATE`, or `ALTERNATIVE`. A source is covered at K when the intersection of its complete valid benchmark-positive target set and its first K frozen candidate target codes is nonempty. Combinations and `NO_MAP` are excluded from ordinary coverage, not silently counted as misses. The coverage denominator is the number of ordinary sources, including ordinary retrieval misses.

The historical `0.9908062234794908` value equals `1401/1414` and is a different filtered population from the canonical ordinary population. The canonical TEST ordinary population is 2,695 sources, of which 2,669 are covered and 26 are misses, giving `0.9903525046382189` (reported as `0.990353`) and miss rate `0.0096474953617811` (reported as `0.009647`). The 0.990806 value must not be used for Step 8 end-to-end metrics.

## Research questions

1. Can a biomedical cross-encoder improve early-rank accuracy while preserving candidate recall?
2. Are gains larger for `SINGLE_APPROXIMATE`, `ALTERNATIVE`, and `LEXICAL_LOW`?
3. Can cross-attention resolve semantically close ICD concepts?
4. Does reranking transfer to combination structure despite exclusion from ordinary supervision?
5. Does pretrained MedCPT improve ranking zero-shot?
6. Does supervised adaptation outperform pretrained MedCPT?
7. Can Hit@1/MRR improve without materially damaging Hit@10/25/50?

## Model and input contract

Primary initialization: `ncbi/MedCPT-Cross-Encoder`, pinned to its immutable Hub revision after metadata verification. The architecture is `BertForSequenceClassification`; the model-card config reports hidden size 768, 12 layers, 12 attention heads, 512 maximum positions, one scalar classifier output, and FP32 weights. The model-card license metadata is `other` / `public-domain` with the repository LICENSE as the controlling source; this is recorded as model-card provenance, not a legal determination.

Input is tokenizer-native source/target text only: source long description, followed by target long description. Short-description fallback is allowed only when long description is unavailable. The text must not contain source/target codes, retriever rank or score, mapping kind, lexical difficulty, family/GEM metadata, split identifiers, or `candidate_is_gold`.

## Supervision and ablations

Eligible supervised sources are ordinary TRAIN sources with at least one valid gold target in the frozen top-100 list. `NO_MAP`, combinations, and ordinary retrieval misses are excluded from ranking loss. All represented valid alternatives remain positive; no valid positive may be sampled as a negative. Each source contributes one source-balanced list unit.

Candidate list size is selected by measured VRAM/throughput/stability only from 4, 8, or 16. Negative strategies are `TOP-RANK HARD`, `MIXED-RANK` (3 ranks 1–10, 2 ranks 11–25, 2 ranks 26–100, adjusted for positives), and `RANDOM-WITHIN-CANDIDATE`. Objectives are source-balanced `BCEWithLogitsLoss` and stable set-positive listwise log-softmax loss. Initial optimization is AdamW, learning rate grid 1e-5/2e-5/3e-5, weight decay 0.01, maximum 3 epochs, warmup 0.10, linear scheduler, max grad norm 1.0, and FP16 only if stable; otherwise FP32.

The fixed DEV lexicographic selection criterion is: Hit@1, MRR, Hit@10, NDCG@10, CompleteScenarioRetrieval@10. Objective, negative strategy, learning rate, max length, sample list size, checkpoint epoch, and canonical seed are selected from TRAIN/DEV only. Final seeds are 17, 42, and 2026, each initialized from the same pinned pretrained model.

## Evaluation and lock

Zero-shot MedCPT is evaluated on forward DEV before training. Every reranker evaluation asserts candidate membership equality and Hit@100 equality with the frozen candidate generator on the same population. After all DEV decisions, write immutable `artifacts/experiments/shift_map_v2/test_lock.json` containing candidate hashes, retriever hash, model revision, configuration hash, seed policy, and TEST exposure disclosure. Only then evaluate forward TEST.

Primary metrics are Hit@1/3/5/10/25/50/100, MRR, NDCG@5/10, conditional reranker metrics, end-to-end metrics including retrieval misses, paired bootstrap deltas (5,000 replicates, seed 2026), oracle ceiling, lexical/mapping/cardinality slices, combination structural transfer, family-held-out transfer, NO_MAP score diagnostics, and the preregistered score-fusion DEV ablation at alpha 0.25/0.50/0.75.

## Compute and artifact policy

Use one bounded GPU process at a time, `CUDA_VISIBLE_DEVICES=0`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, explicit timeout, monitored temperature/utilization/VRAM/AC state, and no quantized training. Model caches and checkpoints remain outside Git. Generated artifacts are JSON/CSV/PNG/Markdown only; no credentials, raw restricted data, or model binaries may be committed.

## Claims and stopping rules

Classify the outcome V2-C1 through V2-C7 only after final evaluation and integrity checks. Any candidate-membership or Hit@100 change is an evaluation integrity failure and blocks interpretation. Report observed TEST exposure honestly. Stop after Step 8; the next authorized milestone would be Step 9 hierarchy-aware reranking.
