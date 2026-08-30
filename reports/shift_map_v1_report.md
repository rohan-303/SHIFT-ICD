# SHIFT-MAP v1 Report

## 1. Research objective
BioLORD-initialized supervised cross-version retrieval adaptation under a strict TRAIN/DEV/TEST boundary.

## 2. Frozen baselines and boundary
BM25 `bm25_v1_1` and zero-shot `dense_v1_1` remained unchanged. Gradient updates used only forward stratified TRAIN. DEV selected policy, objective, negative strategy, learning rate, epoch, and canonical seed. TEST was locked before evaluation.

## 3. Training design
P2 (SINGLE_EXACT + SINGLE_APPROXIMATE + ALTERNATIVE) with source-balanced sampling was selected. Combinations and NO_MAP were excluded from pairwise contrastive training. L1 masked single-positive InfoNCE was selected; no symmetric target-to-source loss was used. Full valid-positive masking prevents alternatives becoming negatives.

## 4. Final configuration
AdamW, LR `2e-5`, weight decay `0.01`, temperature `0.05`, 3 epochs, linear warmup/decay, max length 64, effective source batch 32, seed set 17/42/2026. DEV-selected canonical seed: 42.

## 5. Test results
See `reports/tables/shift_map_v1/final_three_seed_results.csv`, `final_forward_overall.csv`, and `final_paired_comparisons.csv`. The final model is not clinically validated. On this run, SHIFT-MAP has strong Hit@100 but substantially lower Hit@1/Hit@10 than zero-shot BioLORD; this is a negative adaptation result at the most useful early-candidate ranks and is reported without post-hoc redesign.

## 6. Slices and transfer
Lexical difficulty, mapping kind, alternative-size, family-held-out, backward transfer, combination transfer, NO_MAP diagnostics, and complementarity are serialized in the Step 7 tables. Reverse-direction results are transfer only: the model was trained forward, not backward.

## 7. Safety and limitations
NO_MAP values are post-hoc similarity diagnostics only; no abstention, threshold, calibration, or routing was implemented. CMS GEM retrieval is not clinical equivalence. Combination mappings were never reduced to pairwise positives. Fine-tuned weights remain local and Git-ignored; no redistribution occurred.

## 8. Recommendation
**BLOCKED for SHIFT-MAP v2 cross-encoder reranking.** First investigate the early-rank degradation, score/tie behavior, structural transfer, and representation drift under a separately approved protocol. Do not proceed as if the trained encoder improved retrieval.
