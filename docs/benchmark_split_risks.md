# Benchmark split risks (Track A)

Track A must not split raw GEM rows independently. All rows and all reconstructed scenarios belonging to one `(direction, source_code)` source concept must remain in the same future partition.

Potential leakage mechanisms include:

- identical labels across source and target versions;
- minor wording changes that make near-duplicates easy to recognize;
- code families with nearly identical terminology;
- sibling concepts and parent/child concepts;
- the same source concept appearing in both GEM directions;
- forward/backward GEM relationships exposing the evaluation target;
- duplicates induced by alternative scenarios;
- multiple raw rows from one source leaking across partitions;
- description-derived lexical shortcuts;
- repeated or overlapping source concepts in code-description files.

This milestone creates no train/dev/test files and does not finalize a split algorithm. A future protocol must audit duplicate source concepts, code families, direction coupling, hierarchy, and temporal/version structure before assigning partitions. Future benchmark examples are downstream objects and are not interchangeable with raw rows or canonical source-level mappings.
