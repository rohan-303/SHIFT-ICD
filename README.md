# SHIFT-ICD

## Trustworthy Clinical Terminology Mapping Under ICD Version and Ontology Shift

SHIFT-ICD is a publication-oriented research project for studying whether clinical concepts can be mapped safely across evolving ICD terminology systems. The eventual system, **SHIFT-MAP**, will not force every source concept into one target code. It will support three outcomes: automatic acceptance, human review of plausible alternatives, or abstention when the evidence is insufficient.

## Research question

Can a hierarchy-aware retrieval and reranking system combined with drift-aware uncertainty estimation and selective prediction maintain reliable clinical terminology mappings under ICD version shift while identifying cases that should be accepted automatically, deferred for human review, or abstained from?

## Benchmark tracks

- **Track A:** ICD-9-CM → ICD-10-CM using authoritative CMS General Equivalence Mappings as the initial development benchmark.
- **Track B:** WHO ICD-10 → ICD-11 MMS, preserving changed concepts, information loss, hierarchy changes, and unsupported equivalents.
- **Track C:** Temporal ICD-11 drift across the 2024, 2025, and 2026 releases.

## Planned SHIFT-MAP stages

1. BM25 and pretrained biomedical dense retrieval baselines.
2. Fine-tuned biomedical bi-encoder with hard negatives.
3. Biomedical cross-encoder reranking.
4. Hierarchy-aware reranking.
5. Mapping-cardinality prediction for zero, one, or multiple plausible mappings.
6. Calibrated uncertainty and drift estimation.
7. Set-valued/conformal prediction.
8. Selective ACCEPT / REVIEW / ABSTAIN routing.

No models have been implemented. The current milestone has acquired the authoritative CMS FY 2018 diagnosis GEM archives, recorded their checksums and archive members, inspected their raw fixed-width format, and generated non-transformative audit statistics. Raw downloads remain excluded from Git.

## Reproducibility philosophy

Raw authoritative data will remain immutable. Every downloaded resource, release, license, checksum, transformation, split, configuration, software version, and command will be recorded. Test data will not be used for model selection. Ambiguous, one-to-many, information-loss, and no-map cases will be preserved rather than converted into artificial single labels.

## Repository structure

```text
configs/       Configuration by data, model, and experiment stage
data/         Immutable raw, interim, processed, and external resources
docs/          Research, benchmark, data-governance, and reproducibility specifications
src/shift_icd/ Python package boundary for future components
scripts/       Reproducible data and experiment entry points
tests/         Unit and integration tests
notebooks/     Exploratory work only; no source of truth
reports/       Figures, tables, and error analysis
artifacts/     Explicitly generated experiment artifacts
```

## Current milestone

Track A now has a canonical structured representation built from the immutable CMS FY 2018 diagnosis GEM archives. `data/processed/cms/2018_gem/source_mappings.jsonl` is the authoritative nested source-level representation; the normalized Parquet is one record per raw GEM row, and the source-summary Parquet is for analysis. These are distinct from future benchmark examples and no model or split has been implemented.

Rebuild with:

```bash
python scripts/build_cms_canonical.py --project-root . --force
```

See [`docs/canonical_gem_representation.md`](docs/canonical_gem_representation.md) for the schema and CMS combination semantics.

## Important limitation

SHIFT-ICD is research software. It is **not** a clinical decision-making system, coding service, medical device, or clinically validated tool.
