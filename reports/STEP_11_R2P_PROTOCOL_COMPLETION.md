# STEP 11-R2P — SCIENTIFIC PROTOCOL COMPLETION + PRE-OUTCOME RE-FREEZE

Status: `STEP11_R2_PROTOCOL_COMPLETE_FROZEN`

## 1. Pre-outcome basis

`PROTOCOL_COMPLETION_OCCURRED_BEFORE_ANY_R2_SCIENTIFIC_RESULT = TRUE`.
The blocker commit recorded zero R2 epoch rows, zero scientific FUSION_VAL evaluations, zero official DEV evaluations, zero TEST scoring, and zero trained R2 configurations. No model performance was inspected to complete this protocol.

## 2. Closed blockers

- **R2-B001:** resolved by literal `search_space_v3.json`: two learning rates, two imbalance variants, one fixed R1B architecture, and epochs `[1, 2, 3]`.
- **R2-B002:** resolved by `r2_search_protocol_v2.json`, which freezes optimization, seeds, epoch policy, baselines, metrics, provenance, checkpoint retention, invalid-run rules, and quarantine.
- **R2-B003:** resolved by correcting the one-character transcription only. The selection-rule file was not modified; verified SHA is `f17bd93e0a864070ee13b015c3dc72770ab5c3444f42ebbcecf2ca7eb98cdc8a`.
- **R2-B004:** resolved by exact feature-provenance and metric-contract artifacts plus deterministic tests.

## 3. Search and training contract

- Learning rates: `0.0001`, `0.0003`.
- Imbalance variants: `UNWEIGHTED_PRIMARY`, `TRAIN_DERIVED_FORM_WEIGHTED`.
- Architecture: repaired R1B decoder, unchanged (`S_MAX=6`, `L_MAX=3`).
- Trainable configurations: `4`.
- Development seed: `17`.
- Epochs: `1`, `2`, `3`; maximum `3`; no early stopping or patience.
- Expected scientific epoch rows: `4 × 3 = 12`.
- Optimizer: AdamW.
- Weight decay: `0.0001`.
- Source batch size: `8`.
- Gradient clipping: `1.0`.
- Scheduler: `NONE`; warmup `0`.
- Precision: FP32; no AMP variation.

## 4. Feature provenance

The feature vector is `[source_embedding(768), candidate_embedding(768), raw_retriever_score, normalized_retriever_score, candidate_rank]`, dimension `1539`. Source and candidate embeddings use the frozen `FremyCompany/BioLORD-2023` revision `167aab527b238a50ca65224e6319215d2ff4fc9f`, native SentenceTransformer pooling, `max_seq_length=64`, float32 output, and L2 normalization. Source encoding uses `is_query=True`; candidate encoding uses `is_query=False`. Candidate rank is one-based within the frozen Top-100 list. The normalized score is `z=(score-mean_i)/max(population_sd_i,1e-8)`, float64 statistics and `ddof=0`, then stored as float32. Candidate artifact hashes and canonical semantic/retriever checkpoint SHA are bound in `feature_provenance_v1.json`.

Feature extraction was repeated on 16 bounded FUSION_TRAIN/FUSION_VAL sources. Source and candidate feature hashes, identities, and ordering were identical across repetitions. No full scientific FUSION_VAL scoring occurred.

## 5. Baselines and eligibility

- `B0_TOP1_SINGLE`: most frequent FUSION_TRAIN single subtype, verified as `SINGLE_APPROXIMATE`; rank-1 frozen SHIFT-MAP candidate; exactly one target; never predicts NO_MAP, ALTERNATIVE, or either combination form.
- `B1_THRESHOLD_SET`: retrieval-only set with `gap_j=z_top1-z_j`, include candidates with `gap_j <= delta`; TRAIN-only grid `{0.10,0.25,0.50,1.00}`; source-macro flat-set F1 selection with smaller-delta tie-break. Selected TRAIN delta: `0.10`. Output is SINGLE_APPROXIMATE for singleton and ALTERNATIVE otherwise.
- Retrieval-statistics form baseline: TRAIN-only StandardScaler plus L2 multinomial logistic regression (`C=1.0`, `max_iter=2000`, `random_state=17`) over raw top-1 score, top1-top2 gap, top1-top5 gap, Top-100 mean, Top-100 population SD, and normalized top1 score. Diagnostic only.
- `B2_ORACLE_FORM`: selected model outputs with only predicted mapping form replaced by gold form before assembly.
- `B3_ORACLE_CANDIDATE_ASSIGNMENTS`: selected model structural outputs with only candidate membership/assignment tensors replaced by candidate-contained canonical-gold oracle tensors; candidate universe unchanged.

Only the four trainable configurations can be selected. B0/B1 are deployable comparison baselines; the retrieval-statistics baseline and B2/B3 are diagnostics and are not selection candidates.

## 6. Metric contract

- Exact canonical structure: canonical predicted and gold structures are equal across form, NO_MAP emptiness, scenario partition, choice-list partition, alternatives, and structural cardinality. Report mean over all evaluated sources and separately over frozen representable sources.
- Complete-scenario success: for complex gold forms, at least one predicted scenario satisfies one complete gold scenario, with every required gold choice list matched to a slot in the same predicted scenario and at least one valid alternative in each slot. Matching is permutation-invariant; extra alternatives/scenarios do not negate success.
- Mapping-form metrics: six-class accuracy, macro-F1, and per-class precision/recall/F1; zero division is `0`, macro-F1 includes all six classes.
- NO_MAP: TP, FP, FN, TN, precision, recall, F1, false forced-map rate, and false NO_MAP rate; zero division is `0`.
- Flat sets: per-source precision/recall/F1 with both empty sets scoring `1`; source-macro flat-set F1 is the selection metric; micro metrics are descriptive.
- Cardinality exact accuracy: exact empty/flat cardinality for simple forms and permutation-invariant scenario, slot, and alternative cardinalities for complex forms.
- Structure-validity rate: accepted canonical serialization, Top-100 membership, legal forms/cardinalities, nonempty active slots, and no illegal duplicates, divided by all sources.
- Exact failures are `RETRIEVAL_LIMITED` iff the frozen representability audit says the full gold structure is not Top-100 representable; otherwise `DECODER_LIMITED`.
- Candidate-conditioned populations are frozen by the pre-model representability audit, never by model success. Complex conditioning additionally requires gold form COMBINATION or COMBINATION_WITH_ALTERNATIVES.

## 7. Selection and quarantine

Within each configuration, all three epochs are completed and the representative epoch is the highest frozen lexicographic hierarchy: exact structure, complete scenario, form macro-F1, NO_MAP F1, source-macro flat-set F1, cardinality accuracy. Ties select the earliest epoch. One representative per configuration is then compared under the same hierarchy; a complete tie selects ascending configuration ID. Training loss cannot select a configuration.

Official DEV is guarded by `STEP11_OFFICIAL_DEV_ACCESS_FORBIDDEN`. TEST is guarded by `STEP11_TEST_ACCESS_FORBIDDEN`. Both remain untouched.

## 8. Verification and next contract

Dry selection fixtures verified hierarchy dominance at every level, configuration-ID tie-breaking, and earliest-epoch tie-breaking. Feature-provenance mismatch raises `STEP11_FEATURE_PROVENANCE_MISMATCH`. Metric fixtures cover NO_MAP, singles, alternative/scenario/choice-list permutation invariance, grouping failure, partial scenario success, retrieval-vs-decoder classification, candidate conditioning, empty-set F1, and invalid structures.

Quality gates: pytest `207 passed`; Ruff PASS; mypy PASS on 42 source files; package import PASS; `git diff --check` PASS.

No R2 scientific training, checkpoint, epoch row, FUSION_VAL scoring, DEV scoring, or TEST scoring was performed. The next authorized action is **STEP 11-R2 — TRAIN-INTERNAL STRUCTURED DECODER ABLATIONS + CONFIGURATION FREEZE**, using only FUSION_TRAIN fitting and FUSION_VAL selection under these frozen artifacts.
