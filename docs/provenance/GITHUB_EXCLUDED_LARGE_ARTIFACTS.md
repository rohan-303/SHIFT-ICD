# GitHub Excluded Large Artifacts

Authoritative repository: `C:/Users/rohan/SHIFT-ICD`

This registry describes generated artifacts excluded from the separate GitHub publication mirror because ordinary GitHub hosting rejects individual files at or above 100 MiB. The authoritative archive retains them. Hashes are copied only when measured from the current authoritative workspace; no missing hash is inferred.

| Artifact | Size | SHA-256 | Blob SHA | Role | Stage | Regeneration source |
|---|---:|---|---|---|---|---|
| `artifacts/experiments/shift_map_v1_4/ledgers/l2_seed17_backward_train.json` | 287038842 bytes | `bfab78312cb177c9662e24cd39535ff39c3ab7ef4137896d6a603b033322515d` | `60e8eefd1c6a9edf392a2ca62dfd17dbcdd08e1b` | GENERATED_CANDIDATE_LEDGER | SHIFT-MAP v1.4 | `scripts/reconstruct_shift_map_v1_4_ledgers.py` |
| `artifacts/experiments/shift_map_v1_4/ledgers/zero_shot_backward_train.json` | 286953501 bytes | `29c615fada573c2a4421f5e6ec749bac4338b84060ff7b534dd64389959a7af2` | `0f6f5a431863c2bb2da6c288e0fe91c6690f180a` | GENERATED_CANDIDATE_LEDGER | SHIFT-MAP v1.4 | `scripts/reconstruct_shift_map_v1_4_ledgers.py` |
| `artifacts/experiments/shift_map_v1_4/ledgers/l1_seed42_backward_train.json` | 285932744 bytes | `ff87c0ec03020a8a95c5008300a0ece14f0ec7ada92fb01f44c68a5a37aceee4` | `dc31a9d48b4f991dba32ba75541fb490c92c64b7` | GENERATED_CANDIDATE_LEDGER | SHIFT-MAP v1.4 | `scripts/reconstruct_shift_map_v1_4_ledgers.py` |
| `artifacts/experiments/bm25_full_universe/rankings/backward_stratified_test.jsonl` | 175032004 bytes | `a378bab5088b1d8143308449d59cbd6e3ad29cb1919a2daac14088be06450e15` | `ccabc3846f0a7e94becee87b37c6eeafc076c8d0` | GENERATED_RANKING_RESULT | BM25 full-universe evaluation | `scripts/run_bm25_v2.py; scripts/evaluate_bm25_v2_locked.py` |
| `artifacts/experiments/bm25_full_universe/rankings/backward_family_held_out_test.jsonl` | 174612295 bytes | `a16a92d542dab4d0eddf366c334c188bdaa27dd6a55adebfe4dfaa63d3e110a2` | `6dc15ef867f0d3945f4fa53cbfbca4eec046d96c` | GENERATED_RANKING_RESULT | BM25 full-universe evaluation | `scripts/run_bm25_v2.py; scripts/evaluate_bm25_v2_locked.py` |
| `artifacts/experiments/bm25_v2/rankings/bm25_dev-tuned.jsonl` | 174607640 bytes | `bdba9af95a9563bb079120afb0b98fcc3a7c777c14dc891e9a4042dcd7b7516d` | `a110b3e909ce6d1784f2db9e0d67ad8953f30c73` | GENERATED_RANKING_RESULT | BM25 v2 evaluation | `scripts/run_bm25_v2.py` |
| `artifacts/experiments/dense_full_universe_v2/remote_test_sync/compact-test-results.tar.gz` | 155936559 bytes | `165b6f7ab4f7c9558e2d4946922bee1daab3ee6df09c18f9f14e7d77ed2e04fe` | `e3e73d213f70f3e89a21112e7a8b1d4f93ee426a` | GENERATED_SCORED_ARTIFACT | Dense full-universe evaluation | `scripts/run_dense_full_universe_v2.py` |

## Publication policy

The mirror removes these paths from rewritten history only. Compact manifests, reports, tables, and this registry remain publication content. The authoritative repository remains the source of truth for full scientific evidence.
