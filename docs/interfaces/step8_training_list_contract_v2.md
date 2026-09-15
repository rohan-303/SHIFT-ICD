# Step 8 Training-List Contract v2

Status: frozen before scientific R2 DEV evaluation.

## Why the amendment is required

The original R1 fixed-list-8 rule cannot represent every candidate-contained valid positive: 48 ordinary TRAIN sources have more than eight valid positives in the frozen Top-100 set, with a maximum of 49. These are genuine `ALTERNATIVE` mappings. Truncating them would create false negatives or drop valid supervision; silently excluding them would change the training population.

A global expansion is unnecessary because most sources remain representable by the original width. The amendment therefore changes fixed list size to **minimum list size 8** only.

## Dynamic list rule

For each gold-present ordinary TRAIN source, let `P_i` be the number of unique valid positive targets in its frozen Top-100 candidates:

- `N_i = max(1, 8 - P_i)`
- `L_i = P_i + N_i`

All `P_i` positives are included. Negatives are selected deterministically from the same source's frozen Top-100 candidates, excluding every positive. If no negative exists, execution fails closed with `SOURCE_HAS_NO_VALID_NEGATIVE_IN_FROZEN_CANDIDATE_SET`.

## Source-normalized objectives

For BCE, candidate-level terms are averaged within each source list, then source losses are averaged across the source batch. Thus a 50-candidate alternative list does not receive 50/8 times the weight of an eight-candidate source.

For set-positive listwise loss, the numerator contains all positives in the source list and the denominator contains all positives plus selected negatives. One listwise loss is computed per source and source losses are averaged.

Variable lists are flattened only for model forward scoring with source offsets retained; no padding candidates or artificial labels are used. Batch and effective batch size remain defined in **sources**, not candidate pairs.

## Invariants and boundaries

- Every valid candidate-contained positive is retained.
- No valid positive is selected as a negative.
- Every supervised source has at least one true negative.
- Frozen Top-100 membership is unchanged.
- Model input remains `(source description, target description)` only.
- Rank, score, codes, labels, mapping metadata, list length, and offsets are not model features.
- Corrected TEST is not loaded, scored, or trained on; TEST access count remains zero.
- This amendment occurred before scientific DEV metrics and does not select any R2 configuration.
