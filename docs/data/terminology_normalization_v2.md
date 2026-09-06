# Terminology normalization v2

Canonical terminology identity is deterministic and shared by the authoritative terminology layer and mapping joins:

1. trim leading/trailing whitespace;
2. uppercase the identity;
3. remove periods and all whitespace;
4. preserve remaining alphanumeric characters, including numeric, `V`, and `E` ICD-9 identities;
5. use the resulting value as `canonical_code`;
6. retain the source spelling separately as `display_code`.

Examples:

| Display/source identity | Canonical identity |
|---|---|
| `V58.89` | `V5889` |
| `v5889` | `V5889` |
| ` E11.9 ` | `E119` |
| `001.0` | `0010` |

This normalization changes only code identity used for deterministic membership and joins. Benchmark semantic identifiers are not rewritten. The authoritative terminology builders never read GEM rows, canonical mappings, benchmark gold, or candidate files for membership.
