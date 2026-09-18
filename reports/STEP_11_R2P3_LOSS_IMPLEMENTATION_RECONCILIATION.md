# STEP 11-R2P3 — Production loss implementation reconciliation

## Status

`STEP11_R2_EXECUTABLE_PROTOCOL_FROZEN`

Starting authoritative HEAD: `cb1c77aefa7fc5691868583f4c144ca349aefc09`.

This was a pre-outcome repair. Before this amendment:

- feature caches: 0;
- trained R2 configurations: 0;
- scientific epoch rows: 0;
- checkpoints: 0;
- FUSION_VAL evaluations: 0;
- official DEV evaluations: 0;
- TEST scoring: 0.

## Integrity blocker resolved

The blocker was the mismatch between `loss_contract_v3` and the production `structured_assignment_loss`: the old function returned one scalar, lacked a distinct `L_flat_set`, used positional structural supervision, and did not call `hierarchical_match`.

The repair occurred before any R2 scientific result, feature-cache materialization, training, checkpoint creation, or validation scoring.

## Canonical production implementation

- Module: `src/shift_icd/structured_decoder.py`
- Function: `structured_assignment_loss`
- Result: `StructuredLossResult`
- Matcher: `hierarchical_match` in the same module
- One implementation is used by tests and future training; no separate runner loss exists.

## Seven components and applicability

`L_form`, `L_scenario_count`, and `L_scenario_activity` apply to every source.

`L_slot_count`, `L_slot_activity`, and `L_assignment` apply only to `COMBINATION` and `COMBINATION_WITH_ALTERNATIVES`.

`L_flat_set` applies only to `SINGLE_EXACT`, `SINGLE_APPROXIMATE`, and `ALTERNATIVE`. It is not applicable to `NO_MAP` or complex forms.

Non-applicable components are excluded from source and component denominators; they are not synthetic zero losses.

See:

- `reports/tables/step11_structured_decoder/r2p3_loss_component_contract.csv`
- `reports/tables/step11_structured_decoder/r2p3_component_applicability.csv`

## Matching and masking

For complex sources, the production loss calls the frozen `hierarchical_match` before constructing scenario activity, slot count, slot activity, or assignment targets. Query-index tie-breaking remains deterministic. A matched gold slot with no candidate-contained alternative is excluded from `L_assignment`; it is not converted into an all-negative target.

## Flat-set semantics

For simple mappable forms, `membership_logits` receives a binary target over the frozen candidate identities. Missing canonical alternatives are not hallucinated; only candidate-contained identities are supervised. Complex mappings use assignment tensors rather than flat membership loss.

## Aggregation

For source `i`, the total is the arithmetic mean of its applicable component losses. Batch loss is the arithmetic mean of source totals. Candidate, slot, scenario, and alternative counts therefore do not change a source's top-level weight.

Form weights are applied only to `L_form` and use the already frozen R2P2 contract.

## Validation

Focused reconciliation, assignment, and form-weighting tests passed: `17 passed`.

The tests cover:

- exact seven-component result and applicability ledger;
- gold scenario-order permutation invariance;
- matcher invocation before structural supervision;
- simple-versus-complex flat-set scope;
- gradient flow to every applicable head;
- form-weighting isolation;
- source-balanced aggregation invariants.

The complete R2 scientific preflight remains pre-outcome and has not materialized feature caches.

## Frozen amendment artifacts

- `loss_implementation_contract_v1.json` — `911691d8ab765042b1b0189517557472489849e07c0db0c30f81995b1afde9af`
- `loss_contract_v4.json` — `76f479de8d3617c0a2d659615be6063466db2d9ec423e6366c16f83c9f700eb4`
- `search_space_v5.json` — `fd40effe794a4874913df990b003e5d78396215835accfc6bd0acb8d0800dbce`
- `r2_configuration_grid_v4.json` — `9675c2fbb4eb463c8f7830f2924eb4856498d1edf2318cb55b165a3d202d1615`
- `r2_search_protocol_v4.json` — `02479b05ae9a511e6d09a300aa12f9c761a7e7d78bb0ee97a9a63c6a56dc59d5`
- `r2p3_loss_implementation_amendment_manifest.json` — `b827acba1dc5b322e54c268b752ade203f980e4d713d7d9ec17e4f2470f15ef4`

Preserved contract hashes also passed: model `7c5d920934305985195827ff0396554c54e13cb126db115e5cf17fd384dd109d`, assembler `62b22e20798208726ef6dbd01444ee4968a1abf03a2035d0c0c7175327c9bc05`, cardinality `2f2fb0e6be37bc2517d22833e6aa95d138e0f4ee147e474a68b0cfa5b7942799`, repair manifest `0b0dc365f549ef34b1bad04ed0c173dedbbdedcc936de3f095be92f912c4c4a4`, feature provenance `adf7030faea7ff7f3dd3d9cd6af3fb5b03dcc7cba8e576e76545d240c10f6165`, baseline `16587bc3c10f6e3b641bc65fa85d570022eda6ce33220dc0fbb5856fb5158f3b`, metric `19fb3122266a3c6f11cec7e58d1bb0f5860faf2dbba113d7c3d099037de740a6`, selection `f17bd93e0a864070ee13b015c3dc72770ab5c3444f42ebbcecf2ca7eb98cdc8a`, and interpretation `ecad6ee74d4bd48e3efca7f45597de729dae0fb8079c0dd8a62087c9f9efb5ef`.

## Scientific boundary

This milestone does not train R2, create scientific checkpoints, score FUSION_VAL, access official DEV, or access TEST. The next authorized milestone is the frozen R2 scientific search only after this amendment passes all quality and provenance gates.
