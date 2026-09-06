# Retrieval-universe correction

**Status:** `RETRIEVAL_UNIVERSE_CORRECTED` for terminology and BM25 v2; dense and SHIFT-MAP work remains deferred to Step 7.5B.

## Original error and discovery

The historical retrieval corpora were built from unique GEM-observed target codes. Step 8.4A-R independently compared those corpora with CMS diagnosis terminology and classified the chain as `CU3 — TARGET-UNIVERSE LEAKAGE CONFIRMED`. Historical sizes matched the GEM target populations rather than the complete terminology universes.

The historical retrieval artifacts are preserved and are not overwritten. They are legacy evidence of the discovered flaw, not publication results:

`candidate_universe_status: LEGACY_GEM_OBSERVED_TARGET_UNIVERSE`

## Corrected provenance

`terminology_universe_v2` is built from description-only CMS resources:

- ICD-10-CM FY 2018: `icd10cm_codes_2018.txt` in the verified archive.
- ICD-9-CM Version 32: `CMS32_DESC_LONG_DX.txt`, with short-title fallback.
- Membership construction does not import or read GEM, canonical mappings, benchmark gold, or candidate files.
- Canonical identity is uppercase, trimmed, and dotless. Display formatting is retained separately where available.

The corrected manifests and JSONL corpora are under `artifacts/terminology_universe_v2/`.

## Coverage and v5889

After canonical normalization, both ordinary gold targets and all combination component alternatives are present in the corrected corpora. The earlier `v5889` discrepancy was a case/dot normalization issue: the authoritative ICD-9 source contains `V58.89`, parsed as `V5889`; no description was fabricated.

## Publication implications

Unaffected components remain valid unless separately challenged: raw GEM files, canonical parsing, mapping semantics, mapping-kind labels, benchmark gold, deterministic splits, family-held-out definitions, and lexical-difficulty metadata.

BM25, SapBERT, BioLORD, MedCPT, Qwen, RRF, SHIFT-MAP, hard-negative mining, candidate-K coverage, retrieval floors, and Step 8 Top-100 candidate files derived from the historical candidate universe remain labeled legacy. Historical metrics and hashes are preserved unchanged.

BM25 v2 rebuilds lexical retrieval over all 71,704 ICD-10-CM concepts and uses the original BM25 grid with forward stratified DEV-only selection. TEST is evaluated only after the v2 lock is written; it is not used for selection. Dense reruns, Step 8, and SHIFT-MAP retraining are explicitly deferred to Step 7.5B.

## Benchmark manifest reconciliation

The benchmark-manifest discrepancy was a CRLF versus LF serialization difference. Field-level scientific content was unchanged; the authoritative repository serialization is LF with SHA-256 `7c7a7bdcd9d0065e5d5a6623d07c6e97f129dd17bdfc51bf5660b96d03947381`. The reconciliation is documented separately in `docs/experiments/shift_map_v2_manifest_reconciliation.md`.
