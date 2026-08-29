# Research specification

## Title

**SHIFT-ICD: Trustworthy Clinical Terminology Mapping Under ICD Version and Ontology Shift**

The proposed system is **SHIFT-MAP**.

## Problem statement

Clinical classification systems evolve. A source concept may have one target equivalent, several plausible targets, an approximate or information-loss relationship, no supported equivalent, or a representation that requires structural interpretation. A system that always emits one code can hide ambiguity and create overconfident errors.

SHIFT-ICD studies mapping as a structured, version-aware, selective prediction problem. The project will begin with structured ICD concepts and official mappings, not clinical notes or private patient data.

## Primary research question

Can a hierarchy-aware retrieval and reranking system combined with drift-aware uncertainty estimation and selective prediction maintain reliable clinical terminology mappings under ICD version shift while identifying cases that should be accepted automatically, deferred for human review, or abstained from?

## Secondary research questions

- **RQ1:** How much do lexical, dense, and hybrid retrieval methods differ in candidate recall for common, rare, ambiguous, and semantically similar ICD concepts?
- **RQ2:** Does hierarchy-aware reranking improve mapping quality beyond semantic text similarity alone?
- **RQ3:** Can the system correctly distinguish no reliable mapping, unique mapping, and one-to-many/ambiguous mapping?
- **RQ4:** How does predictive calibration deteriorate under ICD version and ontology shift?
- **RQ5:** Can conformal or selective prediction reduce incorrect automatically accepted mappings while maintaining useful coverage?
- **RQ6:** Do hierarchy-derived uncertainty signals and ontology-drift signals improve failure detection beyond standard classifier confidence?
- **RQ7:** How well does a system calibrated on an earlier terminology transition generalize to later temporal ICD drift?

## Claim boundaries

The project makes predictive and methodological claims, not causal clinical claims. A benchmark result will not establish clinical safety. Conformal coverage claims will be stated only with their assumptions, calibration procedure, target population, and observed shift conditions. “Reliable” will be operationalized using mapping quality, calibration, set coverage, selective risk, and false acceptance—not used as an unsupported global label.

## Planned hypotheses

These are testable hypotheses, not findings:

1. Hybrid lexical+dense retrieval will improve candidate recall over either retriever alone, especially for synonym and terminology variation cases.
2. Structural features will reduce cross-branch and granularity errors beyond text-only reranking.
3. Explicit cardinality prediction will improve detection of no-map and ambiguous cases relative to forced single-label prediction.
4. Calibration and coverage will degrade under version drift; drift-aware routing may reduce incorrect automatic acceptance by deferring difficult cases.
5. Retriever disagreement and hierarchical dispersion will provide failure-detection information beyond top-class confidence.

Rivals include: lexical overlap may dominate on code titles; hierarchy features may encode release artifacts rather than clinical meaning; conformal sets may become too large to be useful; and drift scores may correlate with rarity without adding independent value.

## Planned model stages

| Version | Components | Role | Status |
|---|---|---|---|
| v0 | BM25 + pretrained biomedical dense retrieval | Establish candidate-generation baselines and ceiling | Planned |
| v1 | Fine-tuned biomedical bi-encoder + hard negatives | Learn cross-version concept similarity | Planned |
| v2 | Biomedical cross-encoder | Rerank retrieved candidates precisely | Planned |
| v3 | Hierarchy-aware reranker | Add parent, ancestor, sibling, depth, branch, and graph signals | Planned |
| v4 | Cardinality head | Predict 0, 1, or 2+ plausible mappings | Planned |
| v5 | Calibrated uncertainty | Combine probability, entropy, margin, disagreement, dispersion, OOD, and drift | Planned |
| v6 | Set-valued/conformal prediction | Produce prediction sets with explicit coverage targets | Planned |
| v7 | Selective router | Convert evidence into ACCEPT, REVIEW, or ABSTAIN | Planned |

BM25 and pretrained encoders are baselines. Fine-tuning, structural reranking, cardinality modeling, drift-aware uncertainty, conformal adaptation, and selective routing are proposed components whose value must be demonstrated by ablation.

## System output

SHIFT-MAP will support:

- **ACCEPT:** return one mapping when evidence is strong and estimated risk is below the declared operating threshold.
- **REVIEW:** return a structured candidate set when multiple mappings remain plausible or human interpretation is appropriate.
- **ABSTAIN:** return no mapping when evidence is insufficient, the concept is unsupported, or uncertainty/drift exceeds the allowed threshold.

A future machine-readable prediction record may include: `source_version`, `target_version`, `source_code`, `source_label`, `decision`, `candidate_codes`, `candidate_scores`, `prediction_set_size`, `predicted_cardinality`, `uncertainty_score`, `drift_score`, `hierarchy_dispersion`, and `reason`. This is documentation only at this stage.

## Evaluation plan

### Retrieval

Recall@1, @5, @10, @25, @50, and @100; MRR where useful.

### Mapping quality

Exact match, precision, recall, F1, macro F1, micro F1 where appropriate, and multi-mapping precision/recall/F1.

### Hierarchical quality

Hierarchical precision, recall, and F1; ontology/graph distance between prediction and target; hierarchy violation rate.

### Calibration

Expected Calibration Error, Brier score, Negative Log Likelihood where appropriate, and reliability diagrams.

### Selective prediction

Risk-coverage curve, Area Under the Risk-Coverage Curve, selective risk, coverage, abstention rate, and review rate.

### Conformal prediction

Empirical coverage, target coverage, average prediction-set size, and conditional/group coverage where applicable.

### Headline safety-oriented metric

**False Acceptance Rate (FAR)** is:

```text
incorrect mappings automatically ACCEPTED
----------------------------------------
all mappings automatically ACCEPTED
```

FAR must always be reported with automatic coverage, operating thresholds, and the definition of correctness.

## Failure taxonomy

The planned error labels are: lexical false friend; synonym confusion; hierarchy sibling confusion; parent/child granularity mismatch; cross-branch hierarchy error; deprecated concept; concept split; concept merge; changed definition; information-loss mapping; one-to-many ambiguity; unsupported/no-equivalent concept; retrieval miss; reranker miss; cardinality-prediction error; overconfident wrong prediction; unnecessary abstention; unnecessary review; false automatic acceptance; and possible ICD-11 postcoordination requirement.

## Development rules

1. Build one research stage at a time.
2. Establish baseline evaluation before adding complexity.
3. Give every proposed feature an ablation.
4. Keep held-out test data out of model selection.
5. Never silently modify raw authoritative data.
6. Make every transformation reproducible.
7. Never invent missing medical mappings.
8. Preserve ambiguous and no-map cases.
9. Report negative results.
10. Separate empirical findings from hypotheses.
11. Do not claim clinical validation without expert validation.
12. Do not use protected/private clinical data without an explicitly approved future protocol.
13. Prefer public authoritative terminology data initially.
14. Commit logically separated milestones when Git is configured.

## Current scope boundary

This milestone implements no retrieval, reranking, uncertainty, conformal, parsing, downloading, or training code. The next milestone begins with inspection of official Track A source formats and data-governance records.
