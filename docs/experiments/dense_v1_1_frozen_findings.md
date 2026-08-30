# SHIFT-ICD dense_v1_1 frozen findings

## Claim 1 — Dense semantic retrieval improves over BM25 overall

**Supported:** On forward stratified TEST, BioLORD Hit@10 is 0.921336 versus BM25 0.781447; canonical paired non-combination delta is +0.139889 with 95% bootstrap CI [0.125046, 0.154731].

**Limitation:** This is retrieval against CMS-derived structural gold, not clinical-equivalence validation.

## Claim 2 — BioLORD is the strongest DEV-selected zero-shot encoder

**Supported:** BioLORD wins the predeclared DEV lexicographic criterion and has the best forward stratified TEST Hit@10, Hit@100, and MRR among evaluated dense models.

**Limitation:** This is one frozen benchmark and one model revision; it does not establish universal biomedical superiority.

## Claim 3 — Improvement is disproportionately large on LEXICAL_LOW

**Supported:** Forward stratified TEST Hit@10 is 0.418367 for BM25 and 0.786848 for BioLORD on LEXICAL_LOW.

**Limitation:** Slice labels are benchmark metadata; this does not prove a causal semantic mechanism.

## Claim 4 — Combination mappings remain harder

**Supported:** Combination results are materially lower than ordinary non-combination retrieval and are reported with ChoiceListRecall and CompleteScenarioRetrieval.

**Limitation:** MULTI_SCENARIO counts are zero in all four audited TEST populations, so no multi-scenario performance claim is made.

## Claim 5 — Forward/backward mapping is asymmetric

**Supported:** BioLORD forward stratified TEST Hit@10 is 0.921336, while backward stratified TEST Hit@10 is 0.596958.

**Limitation:** The directions have different source populations and corpus structures; the difference is descriptive, not causal.

## Claim 6 — BM25 and BioLORD errors are complementary, but fixed RRF is not better than BioLORD

**Supported:** At K=10 on forward stratified TEST, BM25 succeeds/BioLORD fails on 53 examples, while BioLORD succeeds/BM25 fails on 438. Fixed k=60 RRF Hit@10 is 0.888312, below BioLORD’s 0.921336.

**Limitation:** RRF was not tuned, and the complementarity ceiling is an oracle diagnostic rather than a deployable system.

## Freeze status

These findings are frozen for downstream SHIFT-MAP protocol design. No TEST result was used to modify the benchmark, select a new model, or design a training objective.
