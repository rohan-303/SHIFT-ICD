# SHIFT-MAP v1 training boundary

**Status:** preregistered preparation only; no training performed in Step 6.1.

## Allowed training data

SHIFT-MAP v1 may use only **FORWARD STRATIFIED TRAIN** examples for gradient updates:

- canonical source descriptions;
- canonical target descriptions;
- mapping direction;
- gold positive relationships;
- GEM mapping structure and training-partition metadata required to construct positives and negatives.

The target terminology corpus may be indexed globally because it is available at inference, but relevance judgments from held-out partitions may not be used to curate negatives.

## Boundaries

- Forward stratified DEV: allowed only for early stopping, hyperparameter selection, negative-mining strategy selection, and checkpoint selection.
- Forward stratified TEST: never used until the Step 7 configuration is frozen.
- Forward family-held-out TEST: never used for training decisions.
- Backward TEST: never used for training decisions.
- Test slice labels, test retrieval results, and test error categories: never used for training optimization.

## Positive-pair semantics

- SINGLE: source ↔ target.
- ALTERNATIVE: every benchmark-valid alternative is a positive relationship for retrieval training, without claiming that alternatives are clinically interchangeable.
- COMBINATION and COMBINATION_WITH_ALTERNATIVES: each complete mapping scenario is a structured relation; its individual components are not complete semantic equivalents by themselves.

For the initial simple pairwise bi-encoder objective, combination and combination-with-alternative examples are **excluded from pairwise fine-tuning eligibility**. They remain in DEV/TEST evaluation. A later component-aware objective would require a new protocol.

## Planned negative categories

- N1 random target negative;
- N2 lexical hard negative;
- N3 high-BioLORD-similarity dense hard negative;
- N4 same-family non-gold negative;
- N5 semantic near-miss;
- N6 cross-version confuser.

Negative strategies must be compared by DEV ablation, not TEST performance.

## Contamination rule

A TRAIN source may mine candidates from the globally available target terminology corpus using training gold and frozen pretrained-retriever scores. It must never use DEV/TEST gold to decide that a candidate is negative. Test relevance judgments are reserved exclusively for final evaluation.

## Explicit prohibitions

This boundary does not authorize fine-tuning, optimizer creation, hard-negative dataset generation, cross-encoder reranking, hierarchy reranking, calibration, conformal prediction, or routing. Those belong to Step 7 after a separate implementation and review.
