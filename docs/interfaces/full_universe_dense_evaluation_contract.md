# Full-universe dense evaluation contract

This interface is mandatory for STEP 7.5B.

## Frozen target universes

- Forward ICD-9-CM → ICD-10-CM: all **71,704** authoritative FY2018 ICD-10-CM diagnosis concepts.
- Backward ICD-10-CM → ICD-9-CM: all **14,567** authoritative Version 32 ICD-9-CM diagnosis concepts.

Target membership must come from `terminology_universe_v2`, never from GEM occurrence, benchmark gold targets, or split files.

## Frozen populations

- Ordinary retrieval: `P_ORDINARY_ANSWERABLE` (`SINGLE_EXACT`, `SINGLE_APPROXIMATE`, `ALTERNATIVE`).
- Structural retrieval: `P_COMBINATION`, `P_COMBINATION_WITH_ALTERNATIVES`, and aggregate `P_COMPLEX`.
- `P_NO_MAP`: diagnostics only; never included in ordinary recall or MRR.

## Required ordinary metrics

For each direction and protocol, report source-level:

`Hit@1`, `Hit@5`, `Hit@10`, `Hit@25`, `Hit@50`, `Hit@100`, and `MRR`.

## Required structural metrics

For each structural population and K in `{1, 5, 10, 25, 50, 100}`, report:

- `ChoiceListRecall@K`;
- `CompleteScenarioRetrieval@K`.

A complete scenario requires every required choice list of one valid scenario to be covered; alternatives within one choice list are not flattened into independent sources.

## Protocol safeguards

DEV-only model/configuration selection is required. TEST is locked before final evaluation. Every result must include the population name, denominator, target-universe version/hash, evaluator version, and split protocol. Dense model execution may begin only after this contract is accepted.
