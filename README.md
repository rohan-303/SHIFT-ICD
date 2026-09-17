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

No models have been implemented. The current milestone has acquired the authoritative CMS FY 2018 diagnosis GEM archives, generated the canonical source-level representation, and built the model-independent Track A Benchmark v1.0. Raw downloads and large derived JSONL outputs remain excluded from Git; manifests and lightweight audit JSON are tracked.

## Reproducibility philosophy

Raw authoritative data will remain immutable. Every downloaded resource, release, license, checksum, transformation, split, configuration, software version, and command will be recorded. Test data will not be used for model selection. Ambiguous, one-to-many, information-loss, and no-map cases will be preserved rather than converted into artificial single labels.

## GitHub publication mirror

The authoritative research archive is maintained locally at `C:\\Users\\rohan\\SHIFT-ICD` and retains the full scientific history and generated evidence. The GitHub repository is a separate publication/reproducibility mirror: generated large ledgers, ranking exports, scored artifacts, caches, and checkpoints may be excluded from its history when hosting limits require it. Compact manifests, reports, tables, and provenance registries preserve hashes and regeneration references; omitted artifacts are not implied never to have existed.

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

Track A now has a canonical structured representation and a reproducible v1.0 benchmark built from the immutable CMS FY 2018 diagnosis GEM archives. `all_examples.jsonl` contains one complete source-level example per `(direction, source_code)`; compact protocol and slice references prevent duplicating nested gold structures. The benchmark includes independent forward/backward tasks, source-held-out and family-held-out splits, symbolic combination evaluation contracts, lexical audit metadata, reverse-relation leakage audits, and target-component feasibility audits. Large derived files are rebuilt locally and the tracked manifest records their hashes.

Rebuild the canonical representation with:

```bash
python scripts/build_cms_canonical.py --project-root . --force
```

Build and validate Track A with:

```bash
python scripts/build_track_a_benchmark.py --project-root . --force
env -u PYTHONPATH .venv/Scripts/python.exe scripts/validate_track_a_benchmark.py
```

See [`docs/track_a_benchmark_protocol.md`](docs/track_a_benchmark_protocol.md) for the schema, task contracts, split policy, and CMS combination semantics.

## Important limitation

SHIFT-ICD is research software. It is **not** a clinical decision-making system, coding service, medical device, or clinically validated tool.
