# Step 8 Corrected Candidate Contract

Status: FROZEN at Step 7.5C-R5.

- Retriever: corrected SHIFT-MAP
- Canonical seed: 17
- Target universe: terminology_universe_v2, 71,704 ICD-10-CM targets
- Candidate K: 100
- Candidate namespace: `artifacts/candidates/shift_map_full_universe_v2/`

## Required cross-encoder input
For every candidate, the only required text input is:

`SOURCE DESCRIPTION + TARGET DESCRIPTION`

## Forbidden Step 8 model features
Source code, target code, retriever rank, retriever score, `candidate_is_gold`, mapping kind, split, and GEM flags are forbidden as model features unless a later preregistered method explicitly changes this contract. `candidate_is_gold`, when present in artifacts, is `EVALUATION_ONLY`.

Candidate files are new corrected full-universe outputs and must not be replaced by historical candidates.
