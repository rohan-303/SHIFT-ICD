# Canonical CMS GEM representation (Track A)

**Schema version:** `1.0`

This document defines the loss-minimizing representation built from the immutable CMS FY 2018 diagnosis GEM resources. It is a research representation, not a clinical billing rule and not a claim that a GEM is a lossless conversion.

## CMS semantics verified from official documentation

The implementation was checked against `Dxgem_guide_2018.pdf` (2018 package, pp. 9–10 and 14) and `GemsTechDoc_2018.pdf` (questions 17–19, pp. 16–19). The guide describes a **scenario** as one possible variation/complete expression of a combination mapping. A **choice list** is the set of alternatives for one component within that scenario. The documentation’s operational rule is: choose one target from each choice list; the selected targets together form one complete mapping cluster. Choice lists are therefore not independent complete mappings.

Consequently, for a scenario with choice lists `(A or B)` and `(C or D or E)`, the representation records two required components, five raw target rows, and `2 × 3 = 6` theoretical complete mapping sets. It does not emit five one-code mappings. The Cartesian product is symbolic by default.

A combination mapping is thus represented as:

```text
source -> scenario -> required choice lists -> alternatives within each list
```

A scenario with one choice list and several alternatives is still structurally distinct from a non-combination alternative mapping because CMS marked the rows as a combination scenario.

`approximate` means CMS flags the relationship as approximate; it is preserved, not interpreted as an error score. `no-map` means CMS supplies no target mapping; the marker is retained as a no-map row and normalized target code is null. `scenario` and `choice_list` are preserved exactly as CMS identifiers after integer parsing; zero is the non-scenario/non-choice-list value in ordinary rows.

## Four conceptual levels

### Level 1: raw GEM row

`GemRawRow` stores one exact fixed-width record: `row_id`, direction, raw source and target fields, the five CMS flags/identifiers, `raw_flag_string`, and `raw_record`. `row_id` is deterministic (`DIRECTION:zero-padded-line-number`). No raw record is discarded.

### Level 2: normalized mapping row

`GemMappingRow` adds source/target versions, minimally normalized codes, long and short labels, and description provenance through the attached `CodeDescription` values. Normalization trims fixed-width padding only. It does **not** add or remove decimal points. No-map targets become null while the original marker remains in `raw_record` and `target_code_raw`.

### Level 3: scenario structure

`GemSourceMapping.scenarios` contains `GemScenario` objects. Each combination scenario contains `GemChoiceList` objects, each containing all target alternatives and contributing raw row IDs. Non-combination mappings remain represented by source-level target alternatives and raw row IDs; they are not promoted into artificial choice lists.

### Level 4: source-level benchmark object

One `GemSourceMapping` exists per `(direction, source_code)`. It retains all raw row IDs, labels, mapping flags, scenario/choice-list structures, cardinalities, mapping kind, and eligibility properties. This JSONL object is authoritative for future benchmark construction. The Parquet output is only a flattened analysis summary.

## Mapping taxonomy

`mapping_kind` is a primary descriptive label; boolean properties retain overlapping facts.

- `NO_MAP`: all rows are no-map rows.
- `SINGLE_EXACT`: one non-combination, non-approximate target.
- `SINGLE_APPROXIMATE`: one non-combination approximate target.
- `ALTERNATIVE`: multiple non-combination target alternatives.
- `COMBINATION`: combination-marked structure without within-choice-list alternatives.
- `COMBINATION_WITH_ALTERNATIVES`: combination structure with at least one choice list containing multiple alternatives.
- `MULTI_SCENARIO`: more than one independent scenario when a more specific combination label does not apply.

A source may simultaneously be combination, multi-scenario, approximate, or no-map according to secondary properties. Primary-label precedence favors `NO_MAP`, then combination-with-alternatives, combination, multi-scenario, alternatives, and single mappings.

## Separate cardinalities

- `target_row_count`: target-bearing raw rows.
- `alternative_count`: unique non-combination targets, or all alternatives across combination choice lists.
- `required_component_count`: maximum number of choice lists in one scenario.
- `scenario_count`: number of CMS scenarios; zero for ordinary non-combination rows.
- `choice_list_count`: total choice lists across scenarios.
- `unique_target_count`: unique target codes across the source.
- `valid_mapping_set_count`: sum of products of alternative counts across scenarios; zero for no-map, one for one simple target. This is theoretical and does not enumerate combinations.

Enumeration is optional and guarded by `max_sets` (default 1,000). If the theoretical count exceeds the limit, the utility raises a clear error and the symbolic object remains intact.

The coverage audit records one unmatched backward-GEM target, `v5889` (raw row `ICD10CM_TO_ICD9CM:073170`). The CMS ICD-9 description resource does not contain that exact padding-only code spelling. It is retained in canonical output with a null attached label; no row is dropped and no punctuation/case correction is inferred.

The maximum ICD-9-CM → ICD-10-CM raw target-row source is **V54.12**, “Aftercare for healing traumatic fracture of lower arm.” It has 533 raw target rows, zero scenario/choice-list identifiers, 533 unique targets, and 533 theoretical alternatives. All are approximate. It is not a malformed combination: CMS did not mark these rows as combination rows. The size reflects a broad aftercare concept mapping to many more specific ICD-10-CM fracture-aftercare concepts (laterality, anatomic site, and healing/status distinctions). Treating these rows as 533 independent benchmark examples would duplicate one source concept and inflate evaluation; the source-level object correctly preserves them as one alternative set.

## Reproducibility and boundaries

`build_cms_canonical.py` reads only the recorded CMS archives, parses both directions independently, attaches only CMS descriptions, and writes deterministic derived files. Raw archives remain immutable and ignored by Git. No retrieval, model, split, calibration, conformal, or selective-routing artifact is produced here.
