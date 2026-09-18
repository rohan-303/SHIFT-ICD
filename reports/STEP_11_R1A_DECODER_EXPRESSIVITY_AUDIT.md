# STEP 11-R1A — STRUCTURED DECODER EXPRESSIVITY + IDENTIFIABILITY AUDIT

## 1. Motivation

R1 froze a form head, one cardinality head, and candidate membership logits. This audit tests whether those outputs can represent the scenario and choice-list assignments required by canonical GEM structures. No scientific training or evaluation was performed.

## 2. Learned output schema

- `form_logits`: `[B, 6]`, source-level raw logits, supervised by `L_form`.
- `cardinality_logits`: `[B, 4]`, source-level raw logits, supervised by `L_card`; not a scenario/slot assignment tensor.
- `membership_logits`: `[B, K]`, candidate-level raw logits, supervised by `L_set`.

No scenario-assignment, slot-assignment, choice-list-assignment, or structured grouping head exists.

## 3. Assembler inputs

`assemble_structure` consumes `structure.mapping_form`, `structure.scenarios[].choice_lists[].alternatives`, and `candidate_scores`. The first two structural inputs are gold-only in the prior oracle path. Ordinary inference has no model-produced equivalent. `GOLD_ONLY_ASSEMBLER_INPUT_COUNT = 2`.

## 4. Prior oracle-test audit

Classification: `ORACLE_INJECTS_EXTRA_STRUCTURAL_INFORMATION`.

The prior oracle supplied the complete canonical `structure`, including scenario IDs, choice-list IDs, and alternative partitions, then supplied candidate scores. These are not tensors emitted by `StructuredSetDecoder`; the prior PASS therefore does not establish learned-contract expressivity.

## 5. Structure identifiability definition

Two semantically distinct canonical structures are identifiable only if identical source, candidate identities, flat membership, and applicable global cardinalities can yield distinct legal model-output states and distinct decoded structures without gold-only metadata.

## 6. Adversarial collision cases

- Alternative `(A OR B)` versus combination `(A AND B)`: the form head can distinguish the labels, but the current assembler still requires an externally supplied structure.
- Same-form choice partitions `(A,B)|(C,D)` versus `(A,C)|(B,D)`: same form, flat set, scenario count, and slot count; no current output distinguishes them.
- Same-form scenario partitions `scenario(A,B), scenario(C,D)` versus `scenario(A,C), scenario(B,D)`: no current output distinguishes them.
- Overlapping target assignments are permitted by the canonical contract; 2 forward TRAIN sources contain overlap across scenario/choice-list assignments.

## 7. Real benchmark collision audit

Forward TRAIN counts:

- COMBINATION: 225
- COMBINATION_WITH_ALTERNATIVES: 243
- Multi-scenario: 39
- More than one choice list: 468
- STRUCTURE_COLLISION_COUNT: 254
- Structures requiring explicit assignment information: 254

The collision count is the union of combination-with-alternatives and multi-scenario sources; their overlap is retained rather than double-counted.

## 8. Supervision trace

`L_form` teaches mapping form, `L_card` teaches only a global cardinality class, and `L_set` teaches candidate inclusion. No loss teaches candidate-to-slot, candidate-to-scenario, choice-list, or alternative-group assignment. The assembler has no leakage-safe deterministic rule to infer these partitions from membership and global counts.

## 9. Maximum-complexity expressivity

Observed maxima are 6 scenarios and 3 required slots. A valid model-output-only oracle reconstruction rate cannot be reported: the prior oracle injects the missing structure. Result: `NOT_VALIDATED`.

## 10. Candidate-limit distinction

This is not a Top-100 retrieval failure. The adversarial structures use the same candidate identities and all targets are assumed available. The failure is schema/model-output expressivity, before candidate membership limits.

## 11. Information-sufficiency table

`reports/tables/step11_structured_decoder/model_output_information_sufficiency.csv`

SHA-256: `7b032470ba14e02ac598ae8da68c1a3413eb0bff69573e99a84737edbf4c1817`

## 12. Final classification

`DECODER_STRUCTURE_NOT_IDENTIFIABLE`

Exact blocker: R1 provides candidate membership and global cardinality but no learned representation for candidate-to-choice-list or candidate-to-scenario assignment. The assembler currently receives those assignments through a gold-structure argument in the oracle path.

Scientific evaluation counts remain zero: full FUSION_VAL `0`, official DEV `0`, TEST `0`.

Next milestone: `STEP 11-R1B — STRUCTURED ASSIGNMENT HEAD CONTRACT REPAIR`.
