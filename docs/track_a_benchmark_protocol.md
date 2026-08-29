# Track A benchmark protocol v1.0

**Benchmark version:** `1.0`
**Canonical input:** CMS canonical schema `1.0`
**Primary development direction:** ICD-9-CM → ICD-10-CM
**Reverse validation direction:** ICD-10-CM → ICD-9-CM

This is an experimental benchmark for terminology translation. GEMs are comparison/translation-assistance resources, not guaranteed lossless clinical conversion rules. The two directions are independent and are never treated as inverses.

## Five distinct objects

1. **Raw GEM row:** one immutable CMS fixed-width record.
2. **Canonical source mapping:** one source concept with all rows, scenarios, choice lists, alternatives, flags, and provenance.
3. **Benchmark example:** one source concept plus benchmark metadata and full reconstructable gold structure.
4. **Partition:** a deterministic assignment of complete benchmark examples to train/dev/test.
5. **Evaluation slice:** a deterministic property-based view over examples; it does not change membership.

## Tasks

### A1 — candidate retrieval

Input is a source code and authoritative source description, with optional experiment metadata. The candidate corpus is all valid target-version diagnosis concepts available in the corresponding authoritative terminology. Future output is a ranked list of target codes. No retrieval model is implemented here.

For single/non-combination alternatives, `Hit@K` succeeds when any acceptable target is present. For combinations, `ChoiceListRecall@K` is the fraction of required choice lists with at least one retrieved alternative. `CompleteScenarioRetrieval@K` succeeds when all choice lists of any one scenario are covered. No-map cases have no ordinary target recall denominator.

### A2 — mapping prediction

A model may output a single code, a set of codes, or no mapping. Single mappings require exact target membership. Non-combination alternatives accept any one listed target as a valid top-1 alternative. Combination predictions must select at least one alternative from every choice list in one scenario. Multiple scenarios use “any complete scenario” semantics. No-map is evaluated separately.

### A3 — mapping-structure prediction

Structure labels describe `NO_MAP`, `SINGLE`, `ALTERNATIVE`, `COMBINATION`, `COMBINATION_WITH_ALTERNATIVES`, and `MULTI_SCENARIO`. Structure is not ordinary target cardinality.

### A4 — selective mapping

Reserved for future work. CMS structure is not an authoritative clinical ACCEPT/REVIEW label. Approximate mappings are not automatically REVIEW. CMS no-map is evidence that GEM supplies no mapping, not a clinically validated abstention label. Model policy and future expert validation must remain separate from CMS structure.

## Gold evaluation contract

- **Single:** predicted code must equal the sole target.
- **Non-combination alternative:** any listed target is an acceptable top-1 prediction; predicting every alternative is not required.
- **Combination:** one target from every required choice list in one scenario is required.
- **Combination with alternatives:** alternatives are symbolic; a predicted set is valid if it selects exactly one target per required list in one scenario. Completeness, component precision/recall/F1, and choice-list coverage are reported separately.
- **Multiple scenarios:** satisfying any complete scenario is structurally correct.
- **No-map:** target prediction is not counted as ordinary retrieval success; report no-map precision, recall, F1, and accuracy separately.

## Split protocols

### Stratified source-held-out

The unit is one complete source concept. A deterministic seed of `20260829` is used with nominal `70%/10%/20%` train/dev/test proportions. Assignment is stratified by mapping kind, approximation/no-map status, combination status, and mapping-size bucket as far as group-preserving assignment permits.

### Source-family-held-out

All source concepts in one source category/family remain together. ICD-9 numeric, V, and E diagnosis codes use their three-character categories after removing display punctuation. ICD-10-CM uses its authoritative three-character category. The family assignment is deterministic and approximately follows target proportions; whole-family constraints may cause distribution differences.

Direction-specific partitions are the default. Joint bidirectional training requires an explicit reverse-relation leakage control.

## Target/component-disjoint feasibility

A future target/component-disjoint protocol is not assumed. The builder audits connected components of the source-target bipartite graph separately by direction. If large components prevent meaningful partitioning, the audit records that result rather than forcing a decorative third protocol.

## Difficulty slices

Slices are deterministic and non-exclusive where appropriate:

- `SINGLE_EXACT`
- `SINGLE_APPROXIMATE`
- `NO_MAP`
- `ALTERNATIVE_SMALL`: 2–5 targets
- `ALTERNATIVE_MEDIUM`: 6–20
- `ALTERNATIVE_LARGE`: 21–100
- `ALTERNATIVE_EXTREME`: >100
- `COMBINATION`
- `COMBINATION_WITH_ALTERNATIVES`
- `MULTI_SCENARIO`
- `LOW_MAPPING_COMPLEXITY`: at most 1 scenario, 2 choice lists, 5 alternatives, and 5 theoretical sets
- `HIGH_MAPPING_COMPLEXITY`: otherwise

## Lexical and confusability metadata

No model is used. Descriptions are tokenized deterministically using lowercase alphanumeric tokens. The score is token Jaccard similarity: shared lowercase alphanumeric tokens divided by the union of tokens. Buckets are fixed at `>=0.90` exact, `>=0.60` high, `>=0.30` medium, otherwise low. A source is `LEXICAL_CONFUSABLE` when an invalid target sharing source tokens scores at least as highly as the best valid target. These are audit metadata, not training negatives.

## Output design

`all.jsonl` contains one full benchmark example per source concept. Split files contain compact benchmark-ID references, avoiding duplication of nested GEM structures. Slice files also contain references. The all-example file plus manifest is the reproducible source of truth.

## Reporting

Future results must report overall metrics plus structural and difficulty groups, including single exact, single approximate, alternatives, combinations, no-map, multi-scenario, low lexical similarity, lexical confusability, and family-held-out evaluation. Use 95% bootstrap intervals and paired comparisons on identical examples where models are compared. No model comparison is performed in v1.0.
