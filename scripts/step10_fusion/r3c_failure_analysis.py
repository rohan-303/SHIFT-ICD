from __future__ import annotations

# ruff: noqa: E501
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/experiments/step10_fusion"
TABLE = ROOT / "reports/tables/step10_fusion"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl_gz(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[str(row["source_id"])] = list(row["ranked_target_codes"])
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def q(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    x = (len(values) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def main() -> None:
    TABLE.mkdir(parents=True, exist_ok=True)
    with (ROOT / "data/benchmarks/cms_track_a/v1.0/forward/all.jsonl").open(encoding="utf-8") as f:
        examples = {str((x := json.loads(line))["benchmark_id"]): x for line in f}
    b0_rows = []
    with gzip.open(ROOT / "artifacts/candidates/shift_map_full_universe_v2/forward_dev_k100.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            b0_rows.append(json.loads(line))
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in b0_rows:
        by_source[str(row["source_id"])].append(row)
    ordinary = {
        sid for sid, ex in examples.items()
        if ex.get("mapping_kind") in {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"}
        and not ex.get("no_map", False)
        and sid in by_source
    }
    ranked = {seed: read_jsonl_gz(ART / f"official_dev_ranked_{seed}.jsonl.gz") for seed in ("B0", "17", "42", "2026")}

    # Semantic Top-1/Top-2 margins from frozen candidate retriever scores.
    margins: dict[str, float] = {}
    for sid in ordinary:
        scores = [float(str(row["retriever_score"])) for row in by_source[sid]]
        mean = statistics.fmean(scores)
        sd = math.sqrt(statistics.fmean([(x - mean) ** 2 for x in scores]))
        z = sorted(
            (
                (float(str(row["retriever_score"])) - mean) / max(sd, 1e-8)
                for row in by_source[sid]
            ),
            reverse=True,
        )
        margins[sid] = z[0] - z[1]

    def positions(seed: str, sid: str) -> dict[str, int]:
        return {code: i + 1 for i, code in enumerate(ranked[seed][sid])}

    displacement: list[float] = []
    unchanged_candidates = 0
    total_candidates = 0
    mrr_groups: dict[str, list[dict[str, float]]] = {"improved": [], "worsened": [], "unchanged": []}
    for sid in sorted(ordinary):
        p0, p1 = positions("B0", sid), positions("17", sid)
        displacement.extend(abs(p1[c] - p0[c]) for c in p0)
        unchanged_candidates += sum(p1[c] == p0[c] for c in p0)
        total_candidates += len(p0)
        ex = examples[sid]
        gold = set(str(x) for x in ex.get("valid_target_codes", ex.get("gold_target_codes", [])))
        # Candidate rows contain the authoritative evaluation flag for this frozen candidate set.
        gold = {
            str(row["target_code"])
            for row in by_source[sid]
            if row.get("candidate_is_gold")
        }
        r0 = min((p0[c] for c in gold if c in p0), default=101)
        r1 = min((p1[c] for c in gold if c in p1), default=101)
        m0, m1 = (0.0 if r0 == 101 else 1.0 / r0), (0.0 if r1 == 101 else 1.0 / r1)
        group = "improved" if m1 > m0 else "worsened" if m1 < m0 else "unchanged"
        mrr_groups[group].append({"margin_1_2": margins[sid], "baseline_rank": float(r0), "delta_mrr": m1 - m0})

    def summary(xs: list[float]) -> dict[str, float | int | str]:
        return {"status": "MEASURED", "n": len(xs), "mean": statistics.fmean(xs) if xs else math.nan,
                "median": statistics.median(xs) if xs else math.nan, "p5": q(xs, .05), "p95": q(xs, .95)}

    write_csv(TABLE / "r3c_residual_distribution.csv", [{
        "status": "NOT_RECORDED",
        "reason": "R3 retained aggregate max_abs_residual/max_abs_fusion_delta only; no per-row residual or score-delta vectors.",
        "residual_mean": "NOT_RECORDED", "residual_sd": "NOT_RECORDED", "residual_median": "NOT_RECORDED",
        "residual_p5": "NOT_RECORDED", "residual_p95": "NOT_RECORDED", "max_abs_residual": 0.31893160939216614,
        "actual_fusion_delta_mean": "NOT_RECORDED", "actual_fusion_delta_sd": "NOT_RECORDED",
        "actual_fusion_delta_p95": "NOT_RECORDED", "maximum_abs_fusion_delta": 0.06378632187843324,
    }])
    write_csv(TABLE / "r3c_rank_displacement.csv", [{
        "status": "POSTHOC_ANALYSIS_ONLY", "seed": 17, "ordinary_source_count": len(ordinary),
        "candidate_count": total_candidates, "mean_abs_rank_displacement": statistics.fmean(displacement),
        "median_abs_rank_displacement": statistics.median(displacement), "p95_abs_rank_displacement": q(displacement, .95),
        "maximum_abs_rank_displacement": max(displacement), "fraction_candidates_unchanged": unchanged_candidates / total_candidates,
        "fraction_top1_unchanged": 1.0, "top1_improved": 0, "top1_worsened": 0, "top1_unchanged": len(ordinary),
    }])
    rows = []
    for group, vals in mrr_groups.items():
        for key in ("margin_1_2", "baseline_rank", "delta_mrr"):
            s = summary([x[key] for x in vals])
            rows.append({"group": group, "variable": key, **s})
    write_csv(TABLE / "r3c_mrr_movement_characteristics.csv", rows)  # type: ignore[arg-type]
    write_csv(TABLE / "r3c_inner_vs_official_feature_shift.csv", [{"status": "NOT_COMPUTED", "reason": "Per-row H3 feature values and FUSION_VAL feature table were not retained in frozen R3 artifacts; no refit or re-extraction performed."}])
    write_csv(TABLE / "r3c_inner_changed_sources.csv", [{"status": "NOT_COMPUTED", "reason": "R2 retained aggregate 2 improved/1 worsened counts but not the three source-level rows required for margin/residual reporting."}])
    write_csv(TABLE / "r3c_seed_variability.csv", [
        {"seed": 17, "Hit@1": 0.6965875370919882, "MRR": 0.772058632582593, "NDCG@10": 0.7736489104270061, "canonical": True},
        {"seed": 42, "Hit@1": 0.6973293768545994, "MRR": 0.7725553407272459, "NDCG@10": 0.7741337650529903, "canonical": False},
        {"seed": 2026, "Hit@1": 0.6973293768545994, "MRR": 0.7724474292237183, "NDCG@10": 0.7741401128916926, "canonical": False},
    ])
    write_csv(TABLE / "r3c_test_quarantine.csv", [{"test_feature_count": 0, "test_scoring_count": 0, "test_training_count": 0, "test_lock": "ABSENT", "status": "PASS"}])
    write_csv(TABLE / "step10_summary.csv", [{"step": "STEP10", "status": "CLOSED", "replication": "OFFICIAL_DEV_FAILS_TO_REPLICATE", "promoted_to_test": False, "next_stage": "STEP 11-R1 — MAPPING CARDINALITY + STRUCTURED SET DECODER DESIGN AND PREREGISTRATION"}])

    correction = {
        "schema": "step10_r3_reporting_correction_v1", "original_wording": "Hit@1 positive inner delta to official zero was labeled direction replicated: Yes.",
        "corrected_wording": "NO_POSITIVE_DIRECTION_REPLICATION: inner positive -> official zero is not positive-direction replication.",
        "affected_metric": "Hit@1", "inner_delta": 0.0007062146892655718, "official_delta": 0.0,
        "metric_values_unchanged": True, "scientific_classification_unchanged": "OFFICIAL_DEV_FAILS_TO_REPLICATE",
    }
    (ART / "r3_reporting_correction.json").write_text(json.dumps(correction, indent=2) + "\n", encoding="utf-8")
    next_stage = {
        "schema": "step10_next_stage_decision_v1", "fusion_promotion_status": "NOT_PROMOTED",
        "reason": "OFFICIAL_DEV_FAILS_TO_REPLICATE", "test_evaluated": False,
        "next_stage": "STEP 11-R1 — MAPPING CARDINALITY + STRUCTURED SET DECODER DESIGN AND PREREGISTRATION",
        "scientific_rationale": "Step 10 preserved the semantic candidate set but its bounded H3 residual did not reproduce the microscopic inner gain. The next preregistered direction should model structured mapping cardinality and set form directly rather than perturbing rank positions.",
    }
    (ART / "next_stage_decision.json").write_text(json.dumps(next_stage, indent=2) + "\n", encoding="utf-8")

    report = f'''# STEP 10 FINAL — Semantic + Hierarchy Fusion\n\n**Status:** `STEP10_FUSION_CLOSED` (scientific closeout; GitHub publication pending verification)\n\n## 1. Motivation\n\nStep 10 tested whether frozen SHIFT-MAP semantic rankings could be improved by a bounded H3 hierarchy residual without replacing the semantic anchor.\n\n## 2. Nested-development design\n\nR2 used only the frozen FUSION_TRAIN/FUSION_VAL source split for configuration selection. R3 refit normalization on full original TRAIN, trained independent seeds 17, 42, and 2026 for the frozen two epochs, froze checkpoints, then created the official-DEV lock before one-shot scoring. No TEST feature extraction, training, scoring, or lock occurred.\n\n## 3. Bounded residual architecture\n\nThe frozen model was 4 → 16 → 1 with ReLU hidden activation, tanh output, H3-only features, and `s_fused = z_sem + 0.20*r_h`. Candidate membership remained frozen.\n\n## 4. Inner split and R2 search\n\nThe selected configuration was λ=0.20, learning rate 3e-4, weight decay 1e-4, epoch 2. The corrected provenance record shows full-precision MRR, not NDCG@10, was the first differing decisive criterion after Hit@1 tied: selected MRR 0.7677976091295469 versus runner-up 0.7677849981529529.\n\n## 5. Microscopic inner improvement\n\nAgainst B0, the selected inner result changed Hit@1 by +0.0007062147, MRR by +0.0006203202, NDCG@10 by +0.0000932353, and neither primary structural @10 metric. The movement was 2 Top-1 improvements, 1 worsening, and 1,413 unchanged sources. The retained R2 package does not contain those three source-level rows, so their exact margins/residuals are **NOT COMPUTED**.\n\n## 6. One-shot official DEV protocol\n\nThe lock preceded feature extraction and scoring. Official DEV ordinary population was 1,348 and P_COMPLEX was 67. Bootstrap used 10,000 paired source-level repetitions with seed 20260917.\n\n## 7. Final seeds and canonical result\n\nSeed 17 remained canonical by preregistration. Its official DEV deltas versus B0 were Hit@1 0.0000000000, MRR -0.0000715519, and NDCG@10 -0.0002173419. P_COMPLEX CSR@10 and ChoiceListRecall@10 were unchanged. The paired 95% CIs were [-0.0004438569, 0.0003014056] for MRR and [-0.0007608319, 0.0001442012] for NDCG@10.\n\n## 8. Failure to replicate\n\nThe frozen classification is `OFFICIAL_DEV_FAILS_TO_REPLICATE`. The canonical model made no Top-1 changes on official DEV: 0 improved, 0 worsened, 1,348 unchanged. Its MRR movement was 14 improved, 25 worsened, and 1,309 unchanged.\n\n## 9. Posthoc residual and rank diagnostics\n\nFrom frozen ranked outputs and candidate scores, the canonical official-DEV mean absolute rank displacement was {statistics.fmean(displacement):.9f}, median {statistics.median(displacement):.9f}, p95 {q(displacement,.95):.9f}, maximum {max(displacement):.0f}; {unchanged_candidates/total_candidates:.9f} of candidate positions were unchanged and the Top-1 unchanged fraction was 1.0.\n\nPer-row residual distributions and actual score-delta distributions were not retained: only aggregate maxima were available (max absolute residual 0.3189316094; maximum absolute fusion delta 0.0637863219). Mean, SD, median, p5/p95 are therefore **NOT RECORDED**, not estimated.\n\nThe B0 semantic Top-1/Top-2 margin was computed posthoc from frozen retriever scores. MRR-improved and MRR-worsened source-group summaries are in `r3c_mrr_movement_characteristics.csv`; these are descriptive and not selection evidence.\n\n## 10. Split-distribution diagnostics\n\nThe frozen R3 package does not retain the per-row FUSION_VAL or official-DEV H3 feature matrix, so H3 mean/SD/median/p5/p95 and standardized mean differences are **NOT COMPUTED**. No normalization adaptation or retraining was performed. Semantic-margin distribution is available for official DEV; an exact FUSION_VAL-versus-DEV standardized comparison is **NOT COMPUTED** because the required per-row R2 table was not retained.\n\n## 11. Seed variability\n\nOfficial DEV Hit@1 was 0.6965875371 for seed 17 and 0.6973293769 for both seeds 42 and 2026. Noncanonical improvements do not supersede the preregistered canonical seed-17 outcome. Seed variability is descriptive and does not reopen selection.\n\n## 12. Evidence versus interpretation\n\n**Observed:** the inner improvement was extremely small; canonical official DEV had zero Top-1 movement, negative MRR/NDCG deltas, unchanged structural @10 metrics, and a larger number of MRR-worsened than MRR-improved sources. The bounded correction preserved most candidate ordering.\n\n**Plausible interpretation, not causal proof:** the H3 residual may contain little incremental information after semantic ranking, and the inner gain may have been concentrated in too few near-tie cases to be stable. The seed spread is also larger than the canonical effect. These interpretations are consistent with, but not proven by, the frozen evidence.\n\n## 13. Limitations\n\nThis is a negative external confirmation on a benchmark with prior exposure in earlier project stages. Missing retained per-row H3/residual and R2 changed-source artifacts limit posthoc mechanism analysis. No additional scoring was used to fill those gaps.\n\n## 14. TEST quarantine\n\nTEST feature count: 0. TEST scoring count: 0. TEST training count: 0. TEST lock: ABSENT. The fusion model was not promoted.\n\n## 15. Final Step 10 conclusion\n\nBounded H3 hierarchy residual fusion produced a microscopic improvement on a TRAIN-internal validation split, but the canonical preregistered model failed to reproduce the gain on one-shot official DEV. The method was therefore not promoted to TEST. This is a valid negative result.\n\n## 16. Next research direction\n\nFreeze `STEP 11-R1 — MAPPING CARDINALITY + STRUCTURED SET DECODER DESIGN AND PREREGISTRATION`. The next stage should model how many targets and what structured mapping form should be returned across SINGLE, ALTERNATIVE, COMBINATION, COMBINATION_WITH_ALTERNATIVES, and NO_MAP, rather than perturbing rank positions.\n'''
    report_path = ROOT / "reports/STEP_10_FINAL_SEMANTIC_HIERARCHY_FUSION.md"
    report_path.write_text(report, encoding="utf-8")

    closeout_paths = {
        "r1_protocol": ART / "r1_artifact_hashes.json", "r2_config_freeze": ART / "config_freeze.json",
        "r2_selection_provenance_correction": ART / "r2_selection_provenance_correction.json",
        "final_seed_manifest": ART / "final_seed_manifest.json", "official_dev_lock": ART / "official_dev_confirmation_lock.json",
        "r3_reporting_correction": ART / "r3_reporting_correction.json",
        "failure_analysis_residual": TABLE / "r3c_residual_distribution.csv",
        "failure_analysis_rank": TABLE / "r3c_rank_displacement.csv",
        "failure_analysis_mrr": TABLE / "r3c_mrr_movement_characteristics.csv",
        "failure_analysis_feature_shift": TABLE / "r3c_inner_vs_official_feature_shift.csv",
        "failure_analysis_inner_changed": TABLE / "r3c_inner_changed_sources.csv",
        "failure_analysis_seed_variability": TABLE / "r3c_seed_variability.csv",
        "failure_analysis_test_quarantine": TABLE / "r3c_test_quarantine.csv",
        "final_step10_report": report_path,
        "test_access_audit": ART / "official_dev_access_audit.json", "next_stage_decision": ART / "next_stage_decision.json",
    }
    closeout = {"schema": "step10_closeout_manifest_v1", "status": "STEP10_FUSION_CLOSED", "test_lock": "ABSENT", "artifacts": {k: {"path": str(v.relative_to(ROOT)), "sha256": sha256(v)} for k, v in closeout_paths.items()}}
    (ART / "closeout_manifest.json").write_text(json.dumps(closeout, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
