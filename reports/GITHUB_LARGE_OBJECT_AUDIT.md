# GitHub Large-Object Audit

**Scope:** reachable Git objects in authoritative repository `C:/Users/rohan/SHIFT-ICD`, audited before mirror filtering.

- Authoritative HEAD: `4566b3e568db99852c71a6a1d6b4c6125454290a`
- Reachable blobs above 25 MiB: `26`
- Reachable blobs at least 95 MiB: `7`
- Reachable blobs at least 100 MiB: `7`
- Current HEAD files at least 100 MiB: `7`
- Largest reachable blob: `287038842` bytes (`273.74 MiB`), `artifacts/experiments/shift_map_v1_4/ledgers/l2_seed17_backward_train.json`

The complete inventory is `reports/tables/repository/github_large_object_inventory.csv`. It preserves blob SHA, exact byte size, MiB, reachable paths, representative commit-touch history, current-HEAD presence, scientific-role classification, regeneration status, and SHA-256 when the current file is available locally. Historical SHA-256 values are `NOT_AVAILABLE` rather than inferred.

## Classification

All objects at or above 95 MiB are generated scientific ledgers, ranking results, scored artifacts, or intermediate synchronization artifacts. No handwritten source, test, configuration, documentation, or license blob was identified in this size class. They are classified `REGENERABLE` from the recorded pipeline/source manifests; this classification does not authorize deleting them from the authoritative repository.

## Publication decision

These generated objects are excluded only from the separate GitHub publication mirror. The authoritative repository and its historical objects remain unchanged.
