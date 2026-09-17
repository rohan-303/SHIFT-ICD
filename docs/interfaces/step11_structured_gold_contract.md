# Step 11 structured gold contract

## Authoritative source

The contract is recovered from `src/shift_icd/data/gem_parser.py`, `src/shift_icd/data/canonical.py`, `src/shift_icd/data/schemas.py`, `src/shift_icd/benchmark/builder.py`, `src/shift_icd/evaluation/gold.py`, `docs/canonical_gem_representation.md`, and `docs/track_a_benchmark_protocol.md`.

A raw CMS GEM record has five flag positions: approximate, no-map, combination, scenario, and choice-list identifiers. The canonical builder preserves them and groups one source concept into symbolic structure. A CMS combination means one target must be selected from every required choice list in one scenario; alternatives inside a choice list are not independent complete mappings. Multiple scenarios are alternatives at the scenario level: any complete scenario is valid.

## Canonical Step 11 representation

Each source is serialized as:

```text
source_code, direction, mapping_form, approximation,
flat_alternatives OR scenarios[
  scenario_id, choice_lists[
    choice_list_id, alternatives[target_code, ...]
  ]
], no_map, cardinality
```

All lists are deterministically sorted by their semantic identifiers or target code. The serialized JSON uses sorted keys and compact separators. Parsing and reserialization are byte-stable.

- Non-combination mappings use `flat_alternatives`.
- Combination mappings use `scenarios -> choice_lists -> alternatives` and never flatten combinations.
- `NO_MAP` has empty alternatives and scenarios and cannot emit a target.
- `approximate.any` and `approximate.all` are status attributes. `SINGLE_EXACT` and `SINGLE_APPROXIMATE` therefore differ in CMS approximation semantics, not structural cardinality.
- `scenario_count`, maximum required slot count, total choice-list count, flat unique target count, and theoretical valid mapping-set count remain separate fields.

## Mapping forms

The current benchmark's mutually exclusive primary forms are `NO_MAP`, `SINGLE_EXACT`, `SINGLE_APPROXIMATE`, `ALTERNATIVE`, `COMBINATION`, and `COMBINATION_WITH_ALTERNATIVES`. The repository schema also permits `MULTI_SCENARIO`; the primary label precedence is inherited from the canonical builder, while scenario multiplicity remains explicit in the structure.

## Validation boundary

All 86,271 benchmark source objects round-trip through the contract with zero schema-unrepresentable cases. Forward frozen SHIFT-MAP Top-100 representability is a separate candidate ceiling and is not treated as a decoder error. The frozen candidate package contains forward files only; no backward candidate universe is invented.

The machine-readable implementation is `src/shift_icd/structured_decoder.py`. The contract artifact is `artifacts/experiments/step11_structured_decoder/structured_gold_contract.json`.
