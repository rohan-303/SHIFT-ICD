# Retrieval metric population contract v2

## Scope

This contract applies to every retrieval system evaluated on Track A v1.0. The target universe is independently authoritative terminology; mapping artifacts provide gold labels and structure only.

## Source populations

For each requested direction and split/protocol:

- **P_ALL:** every benchmark example in that split/protocol. This is bookkeeping only.
- **P_ORDINARY_ANSWERABLE:** `SINGLE_EXACT`, `SINGLE_APPROXIMATE`, and `ALTERNATIVE`. `COMBINATION`, `COMBINATION_WITH_ALTERNATIVES`, and `NO_MAP` are excluded.
- **P_COMBINATION:** `COMBINATION` only.
- **P_COMBINATION_WITH_ALTERNATIVES:** `COMBINATION_WITH_ALTERNATIVES` only.
- **P_COMPLEX:** the union of the two combination populations (and `MULTI_SCENARIO` where present).
- **P_NO_MAP:** `NO_MAP` only.

Counts are mutually exclusive except that `P_COMPLEX` is an aggregate. Every report must state its population and denominator.

## Ordinary retrieval metrics

For one source example, let `best_valid_rank(source)` be the minimum 1-indexed rank among any valid target in the Top-100 ranking. Missing valid targets have rank infinity and contribute zero.

```text
Hit@K = mean( best_valid_rank(source) <= K )
MRR   = mean( 1 / best_valid_rank(source) )
```

The mean is over `P_ORDINARY_ANSWERABLE` only. Alternatives remain one source-level observation; valid alternatives are not flattened into multiple examples. NO_MAP and structural examples never enter this denominator.

## Structural metrics

For each structured source/scenario/choice-list, a choice list is covered when at least one of its alternatives appears in Top-K. `ChoiceListRecall@K` is the fraction of required choice lists covered, using the existing canonical `choice_list_recall_at_k` implementation and preserving its scenario-level structured gold representation.

`CompleteScenarioRetrieval@K` is one when all choice lists in at least one valid scenario are covered within Top-K, and zero otherwise. A single retrieved component of a multi-component scenario is not complete. Multiple scenarios use “any complete scenario” semantics.

Structural metrics are reported separately for `P_COMBINATION`, `P_COMBINATION_WITH_ALTERNATIVES`, and aggregate `P_COMPLEX`; they do not enter ordinary Hit/MRR.

## NO_MAP

NO_MAP examples have no ordinary target-recall denominator. Report only descriptive score diagnostics (mean, median, SD, p25, p75), separately from answerable sources. No abstention threshold is inferred.

## Selection and test lock

Configuration selection is forward stratified DEV only, using the preregistered criterion. TEST is never used for selection. Once locked, the configuration is reused unchanged for all test protocols and directions.

## Versioning

This corrective evaluator is `retrieval_evaluator_v3`. Historical v1/v2 metrics remain reproducible legacy artifacts and are not silently rewritten.
