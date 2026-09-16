# STEP 9-R1 — Hierarchy-Aware Reranker Design, Structural Feature Contract, and Implementation Preflight

**Status:** `STEP9_HIERARCHY_PROTOCOL_FROZEN`

**Scope:** design and implementation preflight only. No full Step 9 DEV ablation was run. No Step 9 TEST feature extraction or scoring was run.

## 1. Motivation and Step 8 negative result

Corrected full-universe SHIFT-MAP remains the canonical Top-100 retriever. Step 8 tested `ncbi/MedCPT-Cross-Encoder` under the corrected candidate and listwise contract. On the locked 2,913-source TEST population, SHIFT-MAP scored Hit@1 `0.703154`, MRR `0.776287`, NDCG@10 `0.775580`, Hit@100 `0.978108`; canonical MedCPT seed 17 scored `0.663822`, `0.746122`, `0.747921`, and `0.978108`. Deltas were `-0.039332`, `-0.030165`, and `-0.027660`; all primary paired bootstrap intervals were strictly negative. Candidate membership and Hit@100 were invariant. Step 8 is frozen as `STEP8_MEDCPT_EVALUATION_FROZEN_NEGATIVE`; MedCPT is not promoted.

## 2. Prior TEST exposure disclosure

Corrected SHIFT-MAP TEST and Step 8 MedCPT TEST were evaluated before this design. R1 architecture and features were designed from the roadmap, committed methodology, TRAIN ontology/data, and general structural principles. Individual TEST failure rows/categories were not inspected for feature invention. New Step 9 TEST access is prohibited until a future Step 9 configuration freeze and TEST lock. Machine-readable disclosure: `artifacts/experiments/step9_hierarchy/prior_test_exposure.json` (SHA-256 `b1406a1c0b5d938a062992a15bd4929947b286ca1b3dd70b782d397b48ccea2a`).

## 3. Step 8 freeze verification

- Starting/current expected HEAD: `0e91823d8c974f9468ae5eb18c156cccb59702c5`.
- Step 8 TEST lock SHA: `3ebbad472841743aeb023e3eebab59bcbf9e5c34aa692c91437f7fb0bb3aa543`.
- Step 8 evaluation manifest SHA: `1bbe1ff4945a68ccc629d0d2495a32bbcac106427c00d10cf6635143950ab310`.
- Raw scored artifact SHA: `4ad6d2eea31a6b96b6b18c40579d731ed5595bc4cf5de37a43d98b02b8f85ba3`.
- Comparison artifact SHA: `b96415168aa6f7600b4cdcd3382effa20281dda46060e5937b0886da1edc1c5b`.
- Canonical seed-17 checkpoint hash recorded by the frozen manifest: `1614dfb9377048e2b77b46fd2b82433f842ca063420fe0ad72f2c3c818ece66b`. The checkpoint binary is not present in the local copied artifact directory; this report does not claim a fresh local binary hash.
- No separate committed R3 prose report exists; the evaluation manifest is the authoritative final report artifact.

## 4. Historical design recovery

The recovery table is `reports/tables/step9_hierarchy/legacy_design_recovery.csv`.

Recovered as `PREEXISTING_COMMITTED`: hierarchy-aware reranking/RQ2 stage, a planned v3 reranker stage, and broad parent/ancestor/sibling/depth/branch/graph signal families.

Recovered as `PREEXISTING_DOCUMENTED`: hierarchy siblings and parent-child confusions as structural benchmark considerations.

Classified `HISTORICAL_UNIMPLEMENTED`: concrete equations, parser, feature cache, model family, search grid, selection rule, leakage tests, and TEST fail-closed runner.

Classified `NEW_R1_PROPOSAL`: exact H0/H1/H3 equations, shallow MLP, listwise objective selection, lexicographic DEV rule, and all R1 safeguards.

## 5. Research question and ontology principle

The frozen question is in `r1_research_question.json` (SHA-256 `49871942b121fdeaa3cca5eb67171aa03c8b2ce0d83ece5f30a78ad457d9aacd`). Step 9 may only rerank frozen SHIFT-MAP Top-100 candidates. It cannot add candidates, retrieve all targets, change K, or use MedCPT candidate membership.

Features use ICD code organization, parent/ancestor links, depth, family/root structure, and frozen candidate-set structure. GEM membership, candidate-is-gold, mapping kind, approximate flags, scenario/choice labels, split, gold identities, and TEST outcomes are forbidden inputs. Raw codes are `STRUCTURE_RESOLUTION_ONLY`; arbitrary code strings are not learned tokens.

## 6. Source and target ontology provenance

Source: CMS ICD-9-CM Version 32 diagnosis-title corpus, SHA-256 `45a7d05ddcadf124af88375b64cdf068bb1e3f999ce7fdacb91f65f4e6d55f08`. There are 14,567 terminology codes; all 14,567 benchmark source codes are covered. Parent/ancestor metadata uses deterministic prefix structure with virtual intermediate prefix nodes and no GEM.

Target: CMS FY2018 ICD-10-CM description corpus, SHA-256 `d4954a3fa02e0bfbecb10b20864198dfa50862ded93f2d9d7313e9fe07ef0dfc`. All 71,704 corrected target-universe codes are covered. All 71,704 are present in the official tabular-order audit; the source order file contains 94,127 rows including non-target structural entries. Prefix-derived virtual nodes are used for deterministic parent/ancestor resolution.

Hierarchy manifest: `artifacts/experiments/step9_hierarchy/hierarchy_metadata_manifest.json` (SHA-256 `4a9e99149da5af5f77063a794f4cd8eb55e19c267831b92f9ed53e06e3b37a47`). Integrity table: `reports/tables/step9_hierarchy/hierarchy_integrity.csv` (SHA-256 `c2492380eaa186e6ab9d97bfcb85080d199dc46f0a581ee508b31617244f97c4`). Source metadata nodes: 17,702; target metadata nodes: 98,855. Cycle count, invalid parent count, ancestor-chain errors, and depth-consistency errors are all zero. Root counts are 12 and 25 respectively. No cross-ontology absolute depth difference is used.

## 7. Frozen feature contract

`feature_contract.json` SHA-256: `f3e306a071bb5126a5b833f8f6f4ab4428a00749cadbbcdc2acd248ea0c4df06`. It freezes eight scalar features:

- H1: source depth normalized within ICD-9; target depth normalized within ICD-10; target sibling count `log1p`; target ancestor count normalized within ICD-10.
- H3: fraction of candidates sharing target parent; fraction sharing target family; fraction sharing an ancestor; fraction sharing target root.

H0 is the no-new-feature control. H2 ancestor semantic compatibility is excluded in R1 because no already-pinned ancestor encoder/cache was required to establish this structural preflight. Frozen semantic encoder: **none**. No semantic encoder was selected from DEV outcomes.

## 8. Feature cache and normalization

The deterministic 32-source TRAIN extraction was run twice. Feature hash on both runs: `6bde6b091a8525b7a73b5ab0c0ee9b944d1d70e8c0d8b3144553e4a8134ccd43`. Missingness mask hash: `a4eb54ba9c45022b9dbdd43e0236f8c5118b0203cb8a4798efca1a9ce916d550`. TRAIN-only scaler hash: `e5eb4b27b7debd1172e0db272018bb6a0049b08aa98679a9dedc7117185e4a24`. Cache manifest: `artifacts/experiments/step9_hierarchy/feature_cache_manifest.json` (SHA-256 `a73adbc94b36053fbcd9f73478731d37f294575f082f4a70c1c37f0e335a6718`).

The full ordinary TRAIN population was audited: 9,224 sources and 922,400 candidate rows. Scale statistics in `feature_scale_audit.csv` use TRAIN only; DEV and TEST were not used to fit means/scales. Invalid values are checked; deterministic constant-feature handling is scale `1.0`, not DEV-driven feature deletion.

## 9. Leakage and invariance tests

The test suite verifies that feature extraction is unchanged when irrelevant metadata is altered, including candidate-is-gold, mapping kind, split, approximate flag, and scenario labels. The implementation accepts only ontology records and candidate identities; it has no GEM, gold, TEST-outcome, or mapping-label input path. Candidate extraction preserves the candidate list. Reranking regression coverage verifies permutation-only behavior and Top-100 membership invariance; therefore Hit@100 and structural@100 are invariant by construction.

## 10. Model, objective, and population

Model family: one shallow `8 → 32 → 1` ReLU MLP scoring head. No large pretrained model is introduced. Objective: `SET_POSITIVE_LISTWISE`, source-balanced, preserving every candidate-contained valid positive. Training population: ordinary answerable TRAIN with mapping kinds `SINGLE_EXACT`, `SINGLE_APPROXIMATE`, and `ALTERNATIVE`, with at least one candidate-contained gold; NO_MAP and combination populations are excluded from the first experiment. The audited population contains 9,224 sources, 922,400 candidates, and 13,204 positives.

Contract-v2 list compatibility was verified for P=1, P=8, P=9, and P=49. Resulting list lengths are 8, 9, 10, and 50; no positive is dropped and every list has a true negative.

## 11. Baselines and search space

B0 is frozen SHIFT-MAP candidate ordering. B1 is frozen Step 8 MedCPT seed-17 ordering, used only as a comparison baseline; MedCPT is not a promoted winner. A 32-source DEV smoke reproduction produced deterministic B0/B1 ordering hashes with candidate identity-set checks and no TEST scoring.

Search-space SHA: `bb28e8415a7da78e2c96e98c4ee2526ac45f0eefdaa8b8669b6ce23b2ff96c10`. It contains H0/H1/H3/H1+H3, the single MLP family, SET_POSITIVE_LISTWISE, learning rates `{1e-4, 3e-4}`, weight decay `{1e-4, 1e-3}`, source batch size 4, three epochs, gradient clipping 1.0, and development seed 17. No post-DEV expansion is allowed.

Epoch rule: `E2_BEST_DEV_CHECKPOINT_LEXICOGRAPHIC_PRIMARY_CRITERIA`. DEV selection rule SHA: `c1e9ba21b385d6126f1df981839d23b55b2837c736161a4e009a76f9e461acf8`. Ordered criteria are ordinary Hit@1, ordinary MRR, P_COMPLEX CompleteScenarioRetrieval@10, P_COMPLEX ChoiceListRecall@10, ordinary NDCG@10, then deterministic configuration/epoch ordering. The rule is explicitly `NEW_R1_PREREGISTERED_RULE`.

Success means H beats B0 on all five preregistered primary metrics with unchanged candidate membership and invariant Hit@100/structural@100. A partial primary-family gain without that all-primary condition is mixed. A primary degradation or invariant failure is degradation. These interpretations were frozen before R2 results.

R2 protocol SHA: `c0b70178458863a76ff94ed3ef5bd8e7744de4e574e3d2e8aec46d9f352e19c2`.

## 12. Smoke and fail-closed evidence

`runner.py --stage smoke` returned `STEP9_SMOKE_ONLY`, finite loss, four variable list lengths `[8, 9, 10, 50]`, checkpoint save/reload true, eight DEV smoke scores, full DEV scientific evaluation count 0, and TEST scoring count 0. Checkpoint paths encode configuration, objective, seed, and epoch; collision regression rejects an existing destination. The R1 runner rejects `--stage dev` and requires a valid future lock for `--stage test`; absent lock raises `STEP9_TEST_ACCESS_FORBIDDEN`.

No GPU functionality was needed: the bounded smoke uses CPU PyTorch and passed. No remote targeted run was therefore required. No Step 9 TEST lock exists and no Step 9 TEST access occurred.

## 13. Quality-gate scope and exact next step

R1 local focused tests passed (`9 passed` after the final additions), and the smoke lifecycle passed. Full repository pytest, Ruff, mypy, import, and diff checks must be run before commit; any failure blocks the protocol status. Full Step 9 DEV scientific evaluation count remains exactly zero.

Next milestone: **STEP 9-R2 — HIERARCHY-AWARE DEV ABLATIONS + CONFIGURATION FREEZE**. R2 may begin only after reviewing this frozen contract and must not access TEST until its own configuration freeze and TEST lock exist.
