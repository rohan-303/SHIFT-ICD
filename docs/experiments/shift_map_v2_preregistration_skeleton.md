# SHIFT-MAP v2 preregistration skeleton

Status: PREPARED, NOT EXECUTED
Gate: GO-A

## Frozen retriever

SHIFT-MAP v1.3 corrected L2, seed 17, epoch 3, candidate K=100. See `docs/experiments/shift_map_retrieval_freeze.md`.

## Reserved Step 8 decisions

- Cross-encoder base model: TBD before Step 8 execution.
- Source/target input format: TBD; must exclude `candidate_is_gold`.
- Negative-candidate policy: TBD and frozen before training.
- Pairwise/listwise objective: TBD.
- DEV selection rule: TBD; TEST remains protected.
- Complex-mapping policy: TBD; no flattening into ordinary binary labels without a new protocol.
- Hierarchy features: TBD and ablation-controlled.
- Reranking metrics: TBD, including ordinary, combination, NO_MAP, and candidate-floor reporting.

No cross-encoder has been implemented or trained in Step 7.4B.
