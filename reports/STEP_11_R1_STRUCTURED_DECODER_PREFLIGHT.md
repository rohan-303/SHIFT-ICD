# STEP 11-R1 — STRUCTURED DECODER PREFLIGHT

## 1. Starting authoritative HEAD

`9aec7322c114286c6b5bd57599d2f720646ac30a` — `chore: define GitHub publication mirror policy`

## 2. Canonical benchmark/gold schema paths

- `src/shift_icd/data/gem_parser.py`
- `src/shift_icd/data/canonical.py`
- `src/shift_icd/data/schemas.py`
- `src/shift_icd/benchmark/builder.py`
- `src/shift_icd/evaluation/gold.py`
- `docs/canonical_gem_representation.md`
- `docs/track_a_benchmark_protocol.md`
- `docs/interfaces/step11_structured_gold_contract.md`
- `artifacts/experiments/step11_structured_decoder/structured_gold_contract.json`

## 3. Mapping-form taxonomy

The authoritative primary taxonomy is `NO_MAP`, `SINGLE_EXACT`, `SINGLE_APPROXIMATE`, `ALTERNATIVE`, `COMBINATION`, and `COMBINATION_WITH_ALTERNATIVES`. The repository schema also permits `MULTI_SCENARIO`; scenario multiplicity is always retained structurally even when a more specific primary combination label has precedence.

## 4. SINGLE exact/approximate semantic treatment

`SINGLE_EXACT` and `SINGLE_APPROXIMATE` have the same structural cardinality: one non-combination target. They differ in CMS approximation attributes only. Approximation is preserved separately as `approximation.any` and `approximation.all`; it is not treated as a new set structure.

## 5. Canonical structured representation summary

- Non-combination mappings use `flat_alternatives`.
- Combination mappings use `scenarios -> choice_lists -> alternatives` and never flatten combinations.
- A combination choice list is a required slot; exactly one candidate is selected from each slot.
- Scenario boundaries are never flattened; multiple scenarios are alternative complete structures.
- `NO_MAP` is an empty first-class structure and cannot emit a target.
- Lists are deterministically ordered; JSON keys and separators are canonicalized.
- Scenario count, required slots, choice-list count, flat unique targets, and theoretical valid mapping-set count remain separate fields.

## 6. Round-trip audited source count

`86,271` benchmark sources across both directions and all benchmark partitions. Exact serialized round trips: `86,271`.

## 7. Unrepresentable gold count

`0` — `UNREPRESENTABLE_GOLD_COUNT = 0`.

## 8. TRAIN mapping-form counts

Primary forward direction, `10,197` sources: `SINGLE_EXACT=2,465`, `SINGLE_APPROXIMATE=5,071`, `ALTERNATIVE=1,898`, `COMBINATION=225`, `COMBINATION_WITH_ALTERNATIVES=243`, `NO_MAP=295`.

## 9. NO_MAP count

TRAIN: `295` forward sources.

## 10. Scenario-count distribution

Forward TRAIN: `0:9,729; 1:429; 2:20; 3:7; 4:10; 6:2`.

Forward FUSION_TRAIN: `0:8,269; 1:367; 2:18; 3:5; 4:7`.

Forward FUSION_VAL: `0:1,460; 1:62; 2:2; 3:2; 4:3; 6:2`.

## 11. Maximum scenario count

`6`.

## 12. Maximum required-slot/cardinality count

Maximum required slots in forward TRAIN: `3`. Maximum flat unique target count: `533`.

## 13. Maximum alternative/choice-list size

`533` alternatives for the largest non-combination alternative source. The complete alternative-size distribution is in `alternative_size_distribution.csv`.

## 14. TRAIN structure SHA

Forward canonical structured-gold SHA: `81ad9ef4e3c24726322cdd87525a910c17965f18ca34e33b4b34adce21158461`.

## 15. DEV structure SHA

Forward canonical structured-gold SHA: `4aee726c695b020f20bb5fc0dafcb11d0c746706b1768925df8c661fc3dfa62e`.

## 16. TEST structure SHA

Forward canonical structured-gold SHA: `ed75988f8f5f274a41e21a15e5001f5438bb9def2c8ae0fc414b8e5ba71aa0ad`. TEST gold was serialized and hash-verified only; no Step 11 TEST model scoring occurred.

## 17–19. Retrieval full-representability rate@100

Using the frozen forward SHIFT-MAP Top-100 candidate files: TRAIN `0.9321369030`; DEV `0.9341111874`; TEST `0.9289392379`. These are candidate ceilings, not model results.

## 20. Per-mapping-form retrieval ceilings

Values are `all gold targets present / complete valid scenario / any valid partial structure`.

| Split | NO_MAP | SINGLE_EXACT | SINGLE_APPROXIMATE | ALTERNATIVE | COMBINATION | COMBINATION_WITH_ALTERNATIVES |
|---|---|---|---|---|---|---|
| TRAIN | 1.000/1.000/0.000 | 0.999/0.999/0.999 | 0.963/0.963/0.963 | 0.883/0.989/0.989 | 0.511/0.524/0.991 | 0.300/0.547/0.996 |
| DEV | 1.000/1.000/0.000 | 0.997/0.997/0.997 | 0.965/0.965/0.965 | 0.886/0.985/0.985 | 0.406/0.406/1.000 | 0.429/0.571/1.000 |
| TEST | 1.000/1.000/0.000 | 0.997/0.997/0.997 | 0.965/0.965/0.965 | 0.865/0.989/0.989 | 0.516/0.516/0.984 | 0.275/0.464/1.000 |

The frozen candidate package contains forward files only. No backward candidate ceiling was fabricated.

## 21. Inner split reuse status

`STEP11_INNER_SPLIT_REUSED_FROM_STEP10`. The exact source-level stratified procedure reproduced FUSION_TRAIN `8,666`, SHA `150a57de2e9101ee9bb5c1699bbb7b3a7f9b1fc1d0863018a8766808c151f24c`; FUSION_VAL `1,531`, SHA `225df0f8343bc5358e1980688018916df4c75675595a77f37ca65ceea17bbcb9`; overlap `0`.

## 22. FUSION_TRAIN structural distribution

Mapping forms: `ALTERNATIVE=1,613; COMBINATION=190; COMBINATION_WITH_ALTERNATIVES=207; NO_MAP=251; SINGLE_APPROXIMATE=4,310; SINGLE_EXACT=2,095`.

Scenario counts: `0:8,269; 1:367; 2:18; 3:5; 4:7`. Required-slot counts: `0:8,269; 2:395; 3:2`.

## 23. FUSION_VAL structural distribution

Mapping forms: `ALTERNATIVE=285; COMBINATION=35; COMBINATION_WITH_ALTERNATIVES=36; NO_MAP=44; SINGLE_APPROXIMATE=761; SINGLE_EXACT=370`.

Scenario counts: `0:1,460; 1:62; 2:2; 3:2; 4:3; 6:2`. Required-slot counts: `0:1,460; 2:71`.

## 24–29. Decoder contract

The model family is a compact DeepSets-style candidate-set encoder. Shared candidate projection and permutation-invariant mean pooling feed source-level mapping-form and cardinality heads; a candidate membership head scores each frozen candidate. Permitted inputs are frozen source/candidate embeddings, retriever score, normalized retriever score, and candidate rank. Mapping kind, GEM flags, gold cardinality, scenario/choice-list IDs, candidate-is-gold, split, and TEST outcomes are excluded from inputs.

The six-class mapping-form head includes NO_MAP. Cardinality heads predict scenario/required-slot bins; flat unique-target and theoretical set counts remain diagnostics. The candidate head scores exactly the frozen Top-100 identities. The deterministic assembler selects only candidate-contained codes, preserves scenarios and choice lists, uses score-then-code tie-breaking, emits empty NO_MAP, and cannot emit unknown or duplicate semantic targets.

## 30. Oracle assembler reconstruction result

`PASS` for the implemented synthetic coverage suite spanning all primary forms, NO_MAP, multi-scenario structures, choice lists, maximum cardinality fixtures, candidate constraints, and deterministic ties. No scientific model evaluation was counted.

## 31–33. Objective and imbalance

`L_form`: source-level mapping-form cross entropy; `L_card`: source-level cardinality cross entropy; `L_set`: source-level candidate-membership BCE. Components are combined within each source and then averaged across sources. Primary imbalance variant is unweighted; a controlled TRAIN-derived inverse-frequency form-weight variant is permitted. No DEV-derived weights.

## 34. Baselines

`B0_TOP1_SINGLE`, `B1_THRESHOLD_SET`, and `B4_FORM_STATISTICS` are deployable baselines. `B2_ORACLE_FORM` and `B3_ORACLE_CANDIDATES` are diagnostic-only upper decompositions.

## 35–36. Metrics and conditioning

The lexicographic metric hierarchy is exact canonical structure match, complete-scenario recovery, mapping-form macro-F1, NO_MAP F1, flat set F1, cardinality exact accuracy, and deterministic configuration ID. End-to-end metrics over every source are primary; candidate-conditioned metrics are reported separately only for fully representable sources.

## 37–39. Contract hashes

- Selection rule: `f17bd93e0a864070ee13b015c3dc72770ab5c3444f42ebbcecf2ca7eb98cdc8`
- Interpretation contract: `ecad6ee74d4bd48e3efca7f45597de729dae0fb8079c0dd8a62087c9f9efb5ef`
- Search space: `4371b7988d25e0c6527deee785e526c89af5c9225a282b4674b2c021660a0e14`

## 40–41. Seeds

Final seeds: `17, 42, 2026`. Canonical seed: `17`.

## 42–43. Fail-closed access

Official DEV result: `STEP11_OFFICIAL_DEV_ACCESS_FORBIDDEN`. TEST result: `STEP11_TEST_ACCESS_FORBIDDEN`.

## 44–46. Leakage, validity, determinism

Gold leakage contract: `PASS`; forbidden metadata is excluded from model inputs. Focused structured-output suite: `7 passed`. Repeated fixed-input decoder forward and byte-stable serialization: `PASS`. Checkpoint names encode configuration, seed, and epoch.

## 47–48. Smoke lifecycle

`STEP11_SMOKE_ONLY`: eight bounded TRAIN sources for one engineering optimization step and four bounded FUSION_VAL smoke sources for shape/inference verification. Loss finite; no smoke output was used for selection. Checkpoint save/reload passed at `artifacts/experiments/step11_structured_decoder/smoke/step11_smoke_only_config_smoke_seed_17_epoch_1.pt`.

## 49–51. Scientific evaluation counts

Full inner-validation scientific evaluation: `0`. Official DEV scientific evaluation: `0`. TEST scoring: `0`.

## 52–54. Quality gates

Full authoritative pytest: `189 passed`. Full Ruff: `All checks passed!`. Mypy: `Success: no issues found in 41 source files`. Package import: `PACKAGE_IMPORT_PASS`. `git diff --check`: `PASS`.

## 55–60. Publication state

Authoritative R1 commit, local tag, mirror commit, mirror push, and Step 11 tag are not yet created. `GITHUB_MAJOR_MILESTONE_PUSH_VERIFIED = FALSE` until all final gates and remote ref checks pass.

## 61. Authoritative Git status

Current worktree is intentionally dirty only with the new Step 11 R1 package pending commit. The required post-commit state is clean `main`.

## 62. Final status

`STEP11_R1_PROTOCOL_FREEZE_CANDIDATE` pending final quality gates, authoritative freeze commit/tag, and documented publication-mirror synchronization.

## 63. Explicit next milestone

`STEP 11-R2 — TRAIN-INTERNAL STRUCTURED DECODER ABLATIONS + CONFIGURATION FREEZE`.

R2 must remain bounded to reused FUSION_TRAIN/FUSION_VAL. No full Step 11 selection, official DEV scoring, or TEST scoring was performed in R1.
