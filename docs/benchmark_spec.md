# Benchmark specification

## Purpose

SHIFT-ICD evaluates terminology translation as structured, version-aware prediction. Reference relationships must be preserved as one-to-one, one-to-many, approximate, combination, information-loss, unsupported, or no-map rather than collapsed into an artificial single-label target.

## Track A — ICD-9-CM → ICD-10-CM

This is the primary initial development benchmark. Use official CMS General Equivalence Mappings and associated official code descriptions and hierarchy resources where legally and technically appropriate.

The benchmark will preserve the mapping records and their flags. Planned slices include:

- one-to-one mappings;
- one-to-many mappings;
- approximate mappings;
- combination mappings where supported by the source documentation;
- no-map cases;
- rare and frequent concepts;
- lexical near-neighbors;
- hierarchy siblings;
- hierarchy parent/child confusions.

The source format, fields, flags, release identity, and licensing/terms must be inspected before any dataset is downloaded or parsed. No final split files will be created in this milestone.

## Track B — WHO ICD-10 → ICD-11 MMS

This is a separate terminology-transition benchmark using authoritative WHO terminology and mapping resources. ICD-10 and ICD-11 must not be treated as universally directly equivalent.

The benchmark must explicitly preserve changed concepts, information-loss mappings, one-to-many candidates, unavailable or unsupported equivalents, and hierarchy changes. Mapping tables are evidence about relationships for comparison and evaluation; they are not automatically safe conversion rules.

The ICD-11 representation must account for the distinction between the Foundation and a linearization such as MMS. Possible postcoordination requirements will be identified and labeled rather than silently reduced to one ordinary code.

## Track C — Temporal ICD-11 drift

This track studies terminology evolution across official ICD-11 releases, initially targeting:

```text
ICD-11 2024 → ICD-11 2025 → ICD-11 2026
```

A planned temporal-generalization design is:

- train/calibrate on the 2024 → 2025 transition;
- evaluate generalization on the 2025 → 2026 transition.

The exact release identifiers, available fields, and valid mapping resources must be verified during source inspection. Temporal results must report release versions and not imply arbitrary future-shift guarantees.

## Splits and leakage control

Random splitting alone is insufficient. Future splits and slices will account for mapping cardinality, frequency, lexical similarity, semantic-neighbor difficulty, hierarchy relationship, version/release, temporal drift, and no-map status.

We will avoid leakage where nearly identical source-target terminology, duplicated concepts, release copies, or structurally trivial neighbors make test performance artificially easy. The split unit and grouping policy will be selected after inspecting authoritative source structure. Final split files are intentionally deferred.

## Dataset record requirements

Each benchmark instance should retain, where available:

- source code and release;
- source label and descriptions;
- target code(s) and release;
- mapping type and flags;
- source/target hierarchy metadata;
- frequency or rarity slice assignment;
- lexical and structural difficulty labels;
- provenance and source row identifiers.

## Primary comparisons

The baseline ladder will compare BM25, biomedical dense retrieval, hybrid retrieval, cross-encoder reranking, hierarchy-aware reranking, cardinality modeling, calibration, conformal prediction, and selective routing. Each additional component requires an ablation on identical splits and preprocessing.

## Evaluation outputs

Results must include retrieval recall at multiple K values, mapping and multi-mapping metrics, hierarchical metrics, calibration, prediction-set behavior, risk-coverage behavior, abstention/review rates, and false acceptance rate. Results must be stratified by the planned difficulty slices where sample size permits.

## Expert validation extension

A future expert study may sample difficult cases across unique, one-to-many, no-equivalent, deprecated, split/merge, hierarchy-change, postcoordination, high-confidence-error, and abstention categories. This is outside the initial benchmark foundation and requires an appropriate protocol and qualified reviewers.
