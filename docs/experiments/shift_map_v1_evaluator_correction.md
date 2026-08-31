# SHIFT-MAP v1 evaluator correction

## Original bug

The Step 7 evaluator built target descriptions by iterating over all canonical mapping rows. It did not restrict rows to the requested mapping direction. When ICD-9-CM and ICD-10-CM code strings overlapped, reverse-direction descriptions could overwrite forward target descriptions. The resulting fine-tuned and frozen comparisons were not made against the same intended target terminology.

## Affected artifacts

The original `artifacts/experiments/shift_map_v1/test_metrics*.json`, persisted Step 7 evaluation rows, and reports that compare those rows directly with frozen BioLORD are preserved as historical artifacts and must be labeled **INVALID FOR MODEL COMPARISON DUE TO EVALUATOR BUG**.

## Unaffected artifacts

Raw data, canonical schema v1.0, Track A benchmark v1.0, frozen BM25 outputs, frozen dense v1/v1.1 embeddings and metrics, Step 7 training checkpoints, training/negative manifests, and the Step 7 test lock were not overwritten. Step 7.1 showed the training population and negative sets were forward-scoped.

## Corrected corpus logic

`shift_icd.dense.corpus.build_target_corpus(rows, direction)` is the single authoritative builder. It filters canonical rows before target-code/description aggregation, validates ICD-10-CM syntax for forward targets and ICD-9-CM syntax for backward targets, preserves deterministic sorted code order, records terminology version, and computes a corpus hash. The full canonical corpus is required to contain 17,513 forward ICD-10-CM concepts and 11,690 backward ICD-9-CM concepts.

Evaluator version is `2.0`; benchmark version remains `1.0`; canonical schema remains `1.0`.

## Regression tests

Tests cover forward/backward scope, exact corpus counts, shared code-string collision isolation, deterministic hashes/order, and the corrected evaluator path. Frozen cache code order and hashes are checked before replay.

## Disclosure requirements

Corrected TEST results are a bug-fix re-evaluation of a historically observed TEST partition, not a pristine untouched first-look test. No hyperparameter or model-training change was made from the defective TEST result before correction. Corrected DEV replay is used for configuration/epoch/seed decisions; TEST is not used for selection.
