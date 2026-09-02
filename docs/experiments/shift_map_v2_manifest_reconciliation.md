# SHIFT-MAP v2 benchmark-manifest reconciliation

## Affected artifact

`data/benchmarks/cms_track_a/v1.0/manifest.json`

## Evidence

| Representation | SHA-256 | Bytes |
|---|---|---:|
| Pre-reconciliation Git HEAD (`b01cc03f357805c573468e2d9d8dde75762dc373`) | `7c7a7bdcd9d0065e5d5a6623d07c6e97f129dd17bdfc51bf5660b96d03947381` | 11,381 |
| Windows working-tree serialization | `5464b14e2857affd3bff3801e6497d4342449976a006325f6189271bfcabfd47` | 11,718 |
| Normalized working-tree / Git HEAD semantic content | `7c7a7bdcd9d0065e5d5a6623d07c6e97f129dd17bdfc51bf5660b96d03947381` | 11,381 |

## Root cause

The two representations are semantically identical JSON: after CRLF-to-LF normalization, their bytes match exactly. The 337-byte difference is one carriage-return byte for each of 337 lines. The prior handoff `data_manifest.json` recorded the Windows CRLF raw hash although the tracked repository's canonical Git serialization is LF.

## Authoritative choice

The LF Git object with SHA-256 `7c7a7bdcd9d0065e5d5a6623d07c6e97f129dd17bdfc51bf5660b96d03947381` is authoritative. It is the committed benchmark lineage and preserves normal Git and `git diff --check` behavior. The external handoff manifest was stale in raw serialization only; candidate data, benchmark rows, ordering, and scientific metadata were not changed.

## Reconciliation

`artifacts/experiments/shift_map_v2_external/data_manifest.json` now records the canonical LF byte count and SHA-256. Remote code/data must be refreshed from the final committed reconciliation revision rather than retaining the earlier manually copied CRLF file.

## Resulting Git commit

`2b6dd1a96f6f98556fa0a7f93ee8ba9e4bbf24ec` — `fix: reconcile external benchmark manifest hash`
