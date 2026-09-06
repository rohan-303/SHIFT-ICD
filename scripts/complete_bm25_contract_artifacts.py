# ruff: noqa: E501
from __future__ import annotations

import json
import statistics
from pathlib import Path

from run_bm25_v2 import sha256, write_csv, write_json

from shift_icd.evaluation.metric_contract import population_counts

ROOT = Path(__file__).resolve().parents[1]
K = (1, 5, 10, 25, 50, 100)
NAMES = ("forward_stratified_test", "forward_family_held_out_test", "backward_stratified_test", "backward_family_held_out_test")
POPS = ("P_COMBINATION", "P_COMBINATION_WITH_ALTERNATIVES", "P_COMPLEX")


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "sd": None, "p25": None, "p75": None}
    qs = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
    return {"mean": statistics.mean(values), "median": statistics.median(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0, "p25": qs[0], "p75": qs[2]}


def main() -> None:
    out = ROOT / "artifacts/experiments/bm25_full_universe"
    rows_by = {name: json.loads((out / f"{name}_rows.json").read_text(encoding="utf-8")) for name in NAMES}
    counts = {name: population_counts([type("E", (), row)() for row in rows]) for name, rows in rows_by.items()}
    write_json(out / "population_counts.json", counts)
    ordinary = {}
    structural = {}
    no_map = {}
    for name, rows in rows_by.items():
        ordinary[name] = {f"Hit@{k}": mean_or_none([r[f"Hit@{k}"] for r in rows if r["population"] == "P_ORDINARY_ANSWERABLE"]) for k in K}
        ordinary[name]["MRR"] = mean_or_none([r["MRR"] for r in rows if r["population"] == "P_ORDINARY_ANSWERABLE"])
        structural[name] = {}
        for pop in POPS:
            selected = rows if pop == "P_COMPLEX" else [r for r in rows if r["population"] == pop]
            if pop == "P_COMPLEX":
                selected = [r for r in selected if r["population"] in {"P_COMBINATION", "P_COMBINATION_WITH_ALTERNATIVES"}]
            structural[name][pop] = {metric: mean_or_none([r[metric] for r in selected if metric in r]) for metric in [f"ChoiceListRecall@{k}" for k in K] + [f"CompleteScenarioRetrieval@{k}" for k in K]}
        no_map[name] = {"n": counts[name]["P_NO_MAP"], **({} if not any(r["population"] == "P_NO_MAP" for r in rows) else quantiles([float(r["TopScore"]) for r in rows if r["population"] == "P_NO_MAP"]))}
    write_json(out / "ordinary_metrics.json", ordinary)
    write_json(out / "structural_metrics.json", structural)
    write_json(out / "no_map_diagnostics.json", no_map)
    write_json(out / "metric_population_audit.json", {"evaluator_version": "retrieval_evaluator_v3", "metric_contract": "retrieval_metric_population_contract_v2", "historical_test_exposure_disclosed": True, "datasets": {name: {"population_counts": value, "ordinary_denominator": value["P_ORDINARY_ANSWERABLE"], "no_map_excluded_from_ordinary": True} for name, value in counts.items()}})
    tables = ROOT / "reports/tables/bm25_full_universe"
    write_csv(tables / "metric_population_counts.csv", [{"dataset": n, **c} for n, c in counts.items()])
    write_csv(tables / "forward_ordinary_test.csv", [r for r in rows_by["forward_stratified_test"] if r["population"] == "P_ORDINARY_ANSWERABLE"])
    write_csv(tables / "forward_combination.csv", [r for r in rows_by["forward_stratified_test"] if r["population"] == "P_COMBINATION"])
    write_csv(tables / "forward_combination_with_alternatives.csv", [r for r in rows_by["forward_stratified_test"] if r["population"] == "P_COMBINATION_WITH_ALTERNATIVES"])
    write_csv(tables / "forward_complex_structural.csv", [r for r in rows_by["forward_stratified_test"] if r["population"] == "P_COMPLEX"])
    write_csv(tables / "backward_stratified_ordinary.csv", [r for r in rows_by["backward_stratified_test"] if r["population"] == "P_ORDINARY_ANSWERABLE"])
    write_csv(tables / "backward_stratified_structural.csv", [r for r in rows_by["backward_stratified_test"] if r["population"] in POPS])
    write_csv(tables / "backward_family_ordinary.csv", [r for r in rows_by["backward_family_held_out_test"] if r["population"] == "P_ORDINARY_ANSWERABLE"])
    write_csv(tables / "backward_family_structural.csv", [r for r in rows_by["backward_family_held_out_test"] if r["population"] in POPS])
    write_csv(tables / "no_map_diagnostics.csv", [{"dataset": n, "n": c["P_NO_MAP"], "status": "TOP_SCORE_NOT_RECORDED_IN_PRIOR_ROW_ARTIFACT"} for n, c in counts.items()])
    write_json(out / "legacy_comparison.json", {"status": "DESCRIPTIVE_ONLY", "historical_status": "LEGACY_GEM_OBSERVED_TARGET_UNIVERSE", "corrected_status": "AUTHORITATIVE_FULL_TARGET_UNIVERSE", "matched_old_values": "NOT_RECOMPUTED"})
    write_json(out / "runtime.json", {"evaluator_version": "retrieval_evaluator_v3", "bm25_parameters": {"k1": 2.0, "b": 0.75}, "status": "PARTIAL_RUNTIME_NOT_RECORDABLE_FROM_COMPLETED_ROW_ARTIFACTS", "peak_cpu_rss": "NOT_RECORDED"})
    write_json(out / "freeze_manifest.json", {"schema": "bm25-full-universe-freeze-v2", "evaluator_version": "retrieval_evaluator_v3", "terminology_version": "terminology_universe_v2", "bm25_parameters": {"k1": 2.0, "b": 0.75}, "test_lock_sha256": sha256(ROOT / "artifacts/experiments/bm25_v2/test_lock.json"), "historical_test_exposure_disclosed": True})
    print(json.dumps({"counts": counts, "ordinary": ordinary}, sort_keys=True))


if __name__ == "__main__":
    main()
