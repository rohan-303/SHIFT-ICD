# STEP 11-R1B — STRUCTURED ASSIGNMENT REPAIR

## 1. R1A blocker

R1A commit `205e78c8677c7a6ac5dcf417b38879152d28ab6e` proved the original `[B,6]` form, `[B,4]` cardinality, and `[B,K]` membership contract could not identify scenario and choice-list partitions. The prior oracle injected gold structure and was not a valid learned-output oracle.

## 2. Repaired output tensors

The repaired `StructuredAssignmentDecoder` retains the compact DeepSets candidate encoder and adds fixed structural queries:

| Tensor | Shape | Scope |
|---|---|---|
| `form_logits` | `[B,6]` | source |
| `scenario_count_logits` | `[B,7]` | source |
| `scenario_activity_logits` | `[B,6]` | scenario query |
| `slot_count_logits` | `[B,6,4]` | scenario query |
| `slot_activity_logits` | `[B,6,3]` | slot query |
| `assignment_logits` | `[B,K,6,3]` | candidate/scenario/slot |
| `membership_logits` | `[B,K]` | candidate |

`S_MAX=6` and `L_MAX=3` were recovered from the frozen R1 artifacts.

## 3. Assignment representation

`assignment_logits[b,k,s,l]` independently scores candidate `k` as an alternative in scenario query `s`, choice-list/slot query `l`. It is not a mutually exclusive categorical assignment, so legal cross-scenario overlap is representable.

Scenario and slot queries are selected using predicted counts plus deterministic activity ranking. Query indices are not emitted in canonical serialization.

## 4. Matching

`hierarchical_match` uses exhaustive deterministic scenario and slot permutations. Matching costs are summed hierarchically, with query-index tie-breaking. Canonical scenario and choice-list order is normalized before supervision and serialization.

## 5. Loss contract

The repaired source-balanced objective contains:

- `L_form`
- `L_scenario_count`
- `L_scenario_activity`
- `L_slot_count`
- `L_slot_activity`
- `L_assignment`
- `L_flat_set`

Applicable components are normalized within each source, combined, and then averaged across sources. Assignment loss is masked when a gold choice list has zero valid Top-100 candidates. The primary loss weights are equal fixed weights; the only controlled alternative is TRAIN-derived form weighting.

## 6. Assembler

The repaired assembler consumes only model tensors and candidate identities. `GOLD_ONLY_ASSEMBLER_INPUT_COUNT = 0`.

- `NO_MAP`: empty canonical structure.
- `SINGLE_EXACT` / `SINGLE_APPROXIMATE`: one flat candidate with deterministic top-score fallback.
- `ALTERNATIVE`: thresholded flat membership with deterministic top-score fallback.
- `COMBINATION`: one top assignment per active slot.
- `COMBINATION_WITH_ALTERNATIVES`: thresholded assignment alternatives per active slot with top-score fallback.
- Scenario and slot counts determine the number of active queries.
- Canonical serialization removes query-index dependence and preserves scenario/choice-list boundaries.

## 7. Oracle expressivity audit

The repaired oracle constructs only the tensors listed above. It does not pass a gold structure to the assembler.

- Alternative-vs-combination: PASS.
- Same-form/different-choice-list partition: PASS.
- Same-form/different-scenario partition: PASS.
- Scenario-order permutation: PASS.
- Choice-list-order permutation: PASS.
- NO_MAP: PASS.
- SINGLE exact/approximate: PASS.

## 8. TRAIN collision and complex-source audit

Forward TRAIN:

- R1A collision structures: `254`.
- Candidate-contained collision structures: `80`.
- Candidate-contained collision exact reconstruction: `80/80 = 100%`.
- Complex combination sources: `468`.
- Fully candidate-contained complex sources: `188`.
- Exact complex oracle reconstructions: `188/188 = 100%`.
- Partially represented complex sources: `72`.
- Choice-list slots masked because no valid alternative was in Top-100: `248`.
- Multi-scenario sources: `39`.

The remaining cases are separated as candidate-retrieval limitations, not decoder failures.

## 9. Maximum-complexity audit

- Maximum scenarios: `6` — supported.
- Maximum slots per scenario: `3` — supported.
- Maximum alternative size: `533` — candidate capacity is handled through the frozen candidate dimension `K`.
- Multi-scenario and overlap fixtures are supported by independent assignment logits.

Two TRAIN sources contain legally overlapping target identities across structural groups; the independent assignment tensor preserves this possibility.

## 10. Candidate-missing policy

A slot with no candidate-contained valid alternative masks only `L_assignment` for that slot. Scenario count, scenario activity, slot count, and slot activity remain supervised. Partially represented choice lists use the candidate-contained valid alternatives as positives and retain missing alternatives outside the candidate universe.

## 11. Updated contract artifacts

- `model_contract_v2.json`: `7c5d920934305985195827ff0396554c54e13cb126db115e5cf17fd384dd109d`
- `assembler_contract_v2.json`: `62b22e20798208726ef6dbd01444ee4968a1abf03a2035d0c0c7175327c9bc05`
- `loss_contract_v2.json`: `baf5097c321f71df2063d0403457d058a7a38e72052c5cfebeae3ed952bfc33b`
- `cardinality_target_contract_v2.json`: `2f2fb0e6be37bc2517d22833e6aa95d138e0f4ee147e474a68b0cfa5b7942799`
- `search_space_v2.json`: `08f440df60d42193fcc416148dc4454f69bd4b177dd890ab1cbbf913793897de`
- `r1b_protocol_repair_manifest.json`: `0b0dc365f549ef34b1bad04ed0c173dedbbdedcc936de3f095be92f912c4c4a4`
- Information-sufficiency table: `2aa7abeba9035ec21ed4bbaa2a395b41c8fc60119ac6aae615414a88457d9b43`

R1 contracts remain preserved. The v2 contracts are versioned replacements; no old R1 artifact was deleted.

## 12. Smoke lifecycle

`STEP11_R1B_SMOKE_ONLY` passed with five source fixtures covering NO_MAP, SINGLE, ALTERNATIVE, COMBINATION, and COMBINATION_WITH_ALTERNATIVES.

- Finite total/component loss: PASS.
- Finite gradients: PASS.
- Assignment-head update: PASS.
- Scenario-query update: PASS.
- Slot-query update: PASS.
- Checkpoint save/reload: PASS.
- Reload inference equality: PASS.
- Full FUSION_VAL scientific evaluation: `0`.
- Official DEV evaluation: `0`.
- TEST scoring: `0`.
- Full R2 configurations trained: `0`.

## 13. Quality gates

- Full pytest: `199 passed`.
- Ruff: PASS.
- Mypy: PASS.
- Package import: PASS.
- `git diff --check`: PASS.

## 14. Final classification

`DECODER_STRUCTURE_IDENTIFIABLE`

The repaired model-output contract is sufficient for candidate-contained canonical structures, and production assembly consumes zero gold-only inputs.

## 15. Scientific boundary

This was a pre-outcome contract repair. No R2 search, full FUSION_VAL scientific evaluation, official DEV scoring, or TEST scoring occurred.

## 16. Next milestone

`STEP 11-R2 — TRAIN-INTERNAL STRUCTURED DECODER ABLATIONS + CONFIGURATION FREEZE`
