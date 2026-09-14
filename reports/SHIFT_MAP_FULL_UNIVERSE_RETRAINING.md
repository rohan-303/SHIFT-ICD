# SHIFT-MAP Full-Universe Retraining — Step 7.5C-R5

Status: **SHIFT_MAP_FULL_UNIVERSE_FROZEN**

## 1. CU3 rationale and corrected foundation
The earlier target-universe defect invalidated the historical candidate/test evidence. R5 retrained SHIFT-MAP against the corrected `terminology_universe_v2` forward universe of 71,704 ICD-10-CM targets. The corrected forward target-order hash is `8ca0d0cf0213581645fe64d9c1e599552a2fb207eb740f32ca09ffeaaf9f8e26`; the corrected forward retrieval-corpus hash is `32f572abeb2ff4cb003145216fa83bfd9984fab96579205f432993aa9c550464`. The backward universe contains 14,567 ICD-9-CM targets with code hash `204be6e29572332f3141a18347a28e95f2b258e5bacec8cf98046974adbc2eb5` and corpus hash `dcfe26da33d422a2395081fffaaf83e29d2f557d33458ade392d83b14e171b3b`.

## 2. Dense base and immutable configuration
All final runs initialized independently from BioLORD-2023 revision `167aab527b238a50ca65224e6319215d2ff4fc9f`. Frozen R4B configuration: P1 / L1 / `lexical_hard` / learning rate `1e-5` / AdamW / weight decay `0.01` / temperature `0.05` / max sequence length 64 / effective source batch 32 / three epochs / `E2_BEST_DEV_CHECKPOINT_WITHIN_MAX_EPOCH_BUDGET`. The configuration-freeze SHA-256 is `2815310ce7f47ca368d973f8c36aec7261d7af7e9355b62cd8ff134676361ee9`.

## 3. Final seed training and DEV variability
Seeds 17, 42, and 2026 completed independently, with no warm starts, no TEST access during training, and no NaN/Inf/OOM/truncated epochs. The selected DEV epoch is 2 for every seed under the frozen lexicographic rule. The seed-level DEV and mean/std tables are `final_seed_dev.csv` and `final_seed_dev_mean_std.csv`; checkpoint details are in `final_checkpoint_manifest.csv`. Canonical seed 17 was honored by preregistration, not selected by comparative performance.

## 4. TEST lock and corrected TEST
The formal corrected TEST lock was created at `2026-09-14T19:15:59Z`, before any final corrected TEST evaluation. Lock SHA-256: `5f97b347a811fd648e3d99a15b93e1c1ae583ca48c6d74868535c213e0081deb`. Machine chronology verification passed: lock creation precedes the first corrected final TEST result. Historical invalid-universe TEST exposure remains disclosed in the lock; no corrected final SHIFT-MAP TEST metrics were accessed before it.

Forward TEST ordinary answerable (`n=2,695`):

- zero_shot_biolord: n=2695, Hit@1=0.636735, Hit@10=0.870130, Hit@100=0.960668, MRR=0.719458
- final_seed_17: n=2695, Hit@1=0.703154, Hit@10=0.903154, Hit@100=0.978108, MRR=0.776070
- final_seed_42: n=2695, Hit@1=0.704267, Hit@10=0.903154, Hit@100=0.979221, MRR=0.777501
- final_seed_2026: n=2695, Hit@1=0.702412, Hit@10=0.903525, Hit@100=0.979221, MRR=0.776539

The full final TEST, mean/std, structural, family-held-out, and backward tables are under `reports/tables/shift_map_full_universe/`.

## 5. Comparisons and robustness
The paired BioLORD comparison uses 10,000 deterministic paired bootstrap repetitions (seed 20260914) on the same 2,695-source population. BM25 is reported descriptively against the frozen corrected reference and was not used for selection. Structural metrics preserve mapping structure. Forward family-held-out, backward stratified transfer, and backward family-held-out evaluations used no tuning. Lexical-low, hubness, NO_MAP, and representation-drift diagnostics are diagnostic only.

## 6. Corrected Top-100 candidate freeze
Canonical seed 17 generated new candidates in `artifacts/candidates/shift_map_full_universe_v2/`; no historical candidate row was reused. Every source has exactly 100 unique candidates, ranks 1–100, corrected-universe membership, deterministic ordering, and correct split membership. The candidate files contain 10,197 TRAIN sources / 1,019,700 rows, 1,457 DEV sources / 145,700 rows, and 2,913 TEST sources / 291,300 rows. Candidate freeze manifest SHA-256: `2ea718cb2b0f429dec5605bb66add6e5011c7229c5128bbfff54fff1d3ccd696`.

## 7. Step 8 handoff
The frozen contract is `docs/interfaces/step8_corrected_candidate_contract.md` (SHA-256 `20ebe99b2fbf2a82773a3cd1e5778cdcb68c92b9c830ad689f4afd06e36a07e9`). Required input is SOURCE DESCRIPTION + TARGET DESCRIPTION. Source/target codes, ranks, scores, gold flags, mapping kind, split, and GEM flags are forbidden model features; gold flags are evaluation-only.

## 8. Preservation, limitations, and stop
The canonical seed-17 checkpoint is locally preserved and hash-verified; seed 42 and 2026 checkpoint lineage remains in the retained remote workspace. The remote workspace is retained for the next authorized milestone. Limitations include corrected-universe comparability to historical results, transfer-population distribution shift, and diagnostic—not calibrated—NO_MAP scores. MedCPT cross-encoder reranking and Step 8 execution were **not run**.

Next authorized milestone: **STEP 8-R — CORRECTED FULL-UNIVERSE MEDCPT CROSS-ENCODER RERANKING**.
