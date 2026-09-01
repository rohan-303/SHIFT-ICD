# SHIFT-MAP v2 candidate record schema

Version: 1.0

Each record passed from the frozen SHIFT-MAP v1 candidate generator to Step 8 contains:

- `benchmark_id`: string; stable benchmark identifier.
- `source_code`: string; source terminology code.
- `source_description`: string; source description shown to the retriever.
- `target_code`: string; candidate target code.
- `target_description`: string; target description from the direction-scoped frozen corpus.
- `retriever_rank`: integer; 1-based rank within the candidate set.
- `retriever_score`: number; normalized cosine similarity from the frozen retriever.
- `direction`: enum: `ICD9CM_TO_ICD10CM` or `ICD10CM_TO_ICD9CM`.
- `mapping_kind`: benchmark mapping category.
- `split`: enum: `train`, `dev`, or `test`.
- `target_family`: optional string metadata.
- `source_family`: optional string metadata.
- `candidate_is_gold`: boolean; evaluation/training-construction metadata only.

`candidate_is_gold` MUST NOT be included in cross-encoder model input. The model input consists only of source and target text plus permitted non-label metadata. Step 8 must not alter the frozen retriever, target corpus, evaluator, or candidate ordering contract.
