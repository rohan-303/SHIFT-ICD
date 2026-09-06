# BM25 full-universe freeze v2

## Status

`BM25_FULL_UNIVERSE_FROZEN` after metric-population correction. No terminology or BM25 hyperparameter change occurred.

## Provenance

- Terminology: `terminology_universe_v2`
- Forward target: 71,704 ICD-10-CM concepts; corpus hash `32f572abeb2ff4cb003145216fa83bfd9984fab96579205f432993aa9c550464`
- Backward target: 14,567 ICD-9-CM concepts; corpus hash `dcfe26da33d422a2395081fffaaf83e29d2f557d33458ade392d83b14e171b3b`
- Evaluator: `retrieval_evaluator_v3`
- Metric contract: `retrieval_metric_population_contract_v2`
- Frozen BM25: `k1=2.0`, `b=0.75`
- DEV selection: forward stratified DEV, ordinary answerable population only; Hit@10 `0.7040059347181009`
- Test lock: `artifacts/experiments/bm25_v2/test_lock.json`; SHA-256 `c4f625bd175580075a667e8b4716e236c53eda26b28e23b718caf171f0844aae`

## Population correction

The earlier `n=2,913` was `P_ALL`, not the ordinary denominator. The earlier evaluator’s raw vectors included all non-NO_MAP rows, including structural mappings. Canonical ordinary metrics below are recomputed over `P_ORDINARY_ANSWERABLE` only.

Forward stratified TEST: `P_ALL=2,913`, ordinary `2,695`, combination `64`, combination-with-alternatives `69`, complex `133`, NO_MAP `85`.

Backward stratified TEST: `P_ALL=14,341`, ordinary `13,346`, combination `811`, combination-with-alternatives `38`, complex `849`, NO_MAP `146`.

Backward family-held-out TEST: `P_ALL=14,332`, ordinary `13,169`, combination `1,028`, combination-with-alternatives `4`, complex `1,032`, NO_MAP `131`.

## Canonical ordinary metrics

Forward stratified TEST (`n=2,695`): Hit@1 `0.441558`; Hit@5 `0.648609`; Hit@10 `0.710946`; Hit@25 `0.784045`; Hit@50 `0.817069`; Hit@100 `0.847866`; MRR `0.529388`.

Forward family-held-out TEST (`n=2,750`): Hit@1 `0.425091`; Hit@5 `0.618909`; Hit@10 `0.694545`; Hit@25 `0.777455`; Hit@50 `0.824364`; Hit@100 `0.851636`; MRR `0.510758`.

Backward stratified TEST (`n=13,346`): Hit@1 `0.276337`; Hit@5 `0.431740`; Hit@10 `0.485913`; Hit@25 `0.579724`; Hit@50 `0.634872`; Hit@100 `0.671887`; MRR `0.347701`.

Backward family-held-out TEST (`n=13,169`): Hit@1 `0.276407`; Hit@5 `0.404738`; Hit@10 `0.459184`; Hit@25 `0.529653`; Hit@50 `0.565495`; Hit@100 `0.615385`; MRR `0.337469`.

## Forward structural metrics

K order is `1/5/10/25/50/100`.

- COMBINATION ChoiceListRecall: `0.039063 / 0.148438 / 0.257813 / 0.468750 / 0.554688 / 0.656250`
- COMBINATION CompleteScenarioRetrieval: `0 / 0 / 0.015625 / 0.156250 / 0.296875 / 0.468750`
- COMBINATION_WITH_ALTERNATIVES ChoiceListRecall: `0 / 0.132850 / 0.229469 / 0.376570 / 0.482126 / 0.573913`
- COMBINATION_WITH_ALTERNATIVES CompleteScenarioRetrieval: `0 / 0 / 0.014493 / 0.086957 / 0.144928 / 0.217391`
- P_COMPLEX aggregate ChoiceListRecall: `0.018797 / 0.140351 / 0.243108 / 0.420927 / 0.517043 / 0.613534`
- P_COMPLEX aggregate CompleteScenarioRetrieval: `0 / 0 / 0.015038 / 0.120301 / 0.218045 / 0.338346`

All structural metrics are source-level, bounded by 1, and non-decreasing with K. Backward structural metrics are in `artifacts/experiments/bm25_full_universe/structural_metrics.json`.

## NO_MAP diagnostics

Forward stratified NO_MAP (`n=85`): mean `21.3610`, median `19.7318`, SD `7.0366`, p25 `16.8746`, p75 `25.5580`.

Backward stratified NO_MAP (`n=146`): mean `14.7165`, median `12.3125`, SD `7.0206`, p25 `10.5814`, p75 `18.2582`.

No abstention threshold was introduced.

## Runtime

Measured with frozen BM25 on the first 100 benchmark queries per direction: forward index build `0.6690 s`, mean latency `23.13 ms`, p95 `65.07 ms`, `43.23 q/s`; backward index build `0.1227 s`, mean latency `3.45 ms`, p95 `8.34 ms`, `289.77 q/s`. These are explicitly sampled latency measurements, not full-query latency claims. Peak RSS was not recorded.

## Artifacts

Primary artifacts are under `artifacts/experiments/bm25_full_universe/`; tables are under `reports/tables/bm25_full_universe/`. The dense interface contract is `docs/interfaces/full_universe_dense_evaluation_contract.md`.

Historical BM25 remains labeled `LEGACY_GEM_OBSERVED_TARGET_UNIVERSE`; corrected results are `AUTHORITATIVE_FULL_TARGET_UNIVERSE`. Differences must not be described as degradation without accounting for the expanded legitimate distractor universe.

Prior TEST exposure is disclosed. No prior TEST result altered the locked configuration.

## Next milestone

`STEP 7.5B — FULL-UNIVERSE DENSE BASELINES + RETRIEVER RESELECTION`.
