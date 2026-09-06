# ruff: noqa: E501
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from run_bm25_v2 import load_examples, load_full_corpora, sha256, write_csv, write_json

from shift_icd.evaluation.metric_contract import classify_population, metric_values, population_counts
from shift_icd.evaluation.retrieval import choice_list_recall_at_k, complete_scenario_retrieval_at_k
from shift_icd.retrieval.bm25 import BM25Index

ROOT = Path(__file__).resolve().parents[1]
K_VALUES = (1, 5, 10, 25, 50, 100)
FROZEN = (2.0, 0.75)
DATASETS = {
    "forward_stratified_test": ("ICD9CM_TO_ICD10CM", lambda e: e.split == "test"),
    "forward_family_held_out_test": ("ICD9CM_TO_ICD10CM", lambda e: e.source_family_split == "test"),
    "backward_stratified_test": ("ICD10CM_TO_ICD9CM", lambda e: e.split == "test"),
    "backward_family_held_out_test": ("ICD10CM_TO_ICD9CM", lambda e: e.source_family_split == "test"),
}


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "sd": None, "p25": None, "p75": None}
    qs = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
    return {"mean": statistics.mean(values), "median": statistics.median(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0, "p25": qs[0], "p75": qs[2]}


def main() -> None:
    started = time.perf_counter()
    examples = load_examples(ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl")
    corpora, profiles = load_full_corpora()
    indexes = {}
    index_times = {}
    for direction, docs in corpora.items():
        t0 = time.perf_counter()
        indexes[direction] = BM25Index.from_documents(docs, *FROZEN)
        index_times[direction] = time.perf_counter() - t0
    out = ROOT / "artifacts/experiments/bm25_full_universe"
    rows_by_dataset = {}
    population_audit = {"evaluator_version": "retrieval_evaluator_v3", "metric_contract": "retrieval_metric_population_contract_v2", "historical_test_exposure_disclosed": True, "datasets": {}}
    runtime_rows = []
    for name, (direction, predicate) in DATASETS.items():
        dataset = [e for e in examples if e.direction == direction and predicate(e)]
        counts = population_counts(dataset)
        population_audit["datasets"][name] = {"direction": direction, "protocol": "family_held_out" if "family" in name else "stratified", "partition": "test", "population_counts": counts, "ordinary_denominator": counts["P_ORDINARY_ANSWERABLE"], "no_map_excluded_from_ordinary": True}
        rows = []
        no_map_scores = []
        t0 = time.perf_counter()
        ranking_path = out / "rankings" / f"{name}.jsonl"
        ranking_path.parent.mkdir(parents=True, exist_ok=True)
        with ranking_path.open("w", encoding="utf-8") as ranking_file:
            for example in dataset:
                ranked_pairs = indexes[direction].rank(example.source_label or "", limit=100)
                ranked = [code for code, _ in ranked_pairs]
                scores = [score for _, score in ranked_pairs]
                pop = classify_population(example)
                row = {"benchmark_id": example.benchmark_id, "direction": direction, "mapping_kind": example.mapping_kind, "population": pop, "no_map": example.no_map, "TopScore": scores[0] if scores else 0.0}
                row.update(metric_values(set(example.valid_target_codes), ranked, no_map=example.no_map or pop != "P_ORDINARY_ANSWERABLE", k_values=K_VALUES))
                if pop in {"P_COMBINATION", "P_COMBINATION_WITH_ALTERNATIVES", "P_COMPLEX"}:
                    for k in K_VALUES:
                        row[f"ChoiceListRecall@{k}"] = choice_list_recall_at_k(example, ranked, k)
                        row[f"CompleteScenarioRetrieval@{k}"] = float(complete_scenario_retrieval_at_k(example, ranked, k))
                if pop == "P_NO_MAP":
                    no_map_scores.append(scores[0] if scores else 0.0)
                rows.append(row)
                for rank, (code, score) in enumerate(ranked_pairs, 1):
                    ranking_file.write(json.dumps({"benchmark_id": example.benchmark_id, "target_code": code, "rank": rank, "score": score}, sort_keys=True) + "\n")
        elapsed = time.perf_counter() - t0
        rows_by_dataset[name] = rows
        runtime_rows.append({"dataset": name, "direction": direction, "index_concept_count": len(corpora[direction]), "index_build_seconds": index_times[direction], "query_count": len(dataset), "query_seconds": elapsed, "mean_query_latency_seconds": elapsed / len(dataset), "queries_per_second": len(dataset) / elapsed})
        write_json(out / f"{name}_rows.json", rows)
        write_json(out / f"{name}_metrics.json", {pop: {f"Hit@{k}": statistics.mean([r[f"Hit@{k}"] for r in rows if r["population"] == pop and r[f"Hit@{k}"] is not None]) if any(r["population"] == pop and r[f"Hit@{k}"] is not None for r in rows) else None for k in K_VALUES} | {"MRR": statistics.mean([r["MRR"] for r in rows if r["population"] == pop and r["MRR"] is not None]) if any(r["population"] == pop and r["MRR"] is not None for r in rows) else None} for pop in ("P_ORDINARY_ANSWERABLE", "P_COMBINATION", "P_COMBINATION_WITH_ALTERNATIVES", "P_COMPLEX")})
        write_json(out / f"{name}_no_map.json", quantiles(no_map_scores))
    write_json(out / "metric_population_audit.json", population_audit)
    write_json(out / "population_counts.json", {name: population_counts(rows) for name, rows in ((n, [e for e in examples if e.direction == d and p(e)]) for n, (d, p) in DATASETS.items())})
    write_json(out / "runtime.json", {"evaluator_version": "retrieval_evaluator_v3", "bm25_parameters": {"k1": FROZEN[0], "b": FROZEN[1]}, "rows": runtime_rows, "total_seconds": time.perf_counter() - started, "peak_cpu_rss": "NOT_RECORDED"})
    write_json(out / "structural_metrics.json", {name: {pop: {metric: statistics.mean([r[metric] for r in rows if r["population"] == pop]) for metric in [f"ChoiceListRecall@{k}" for k in K_VALUES] + [f"CompleteScenarioRetrieval@{k}" for k in K_VALUES]} for pop in ("P_COMBINATION", "P_COMBINATION_WITH_ALTERNATIVES", "P_COMPLEX")} for name, rows in rows_by_dataset.items()})
    write_json(out / "ordinary_metrics.json", {name: {metric: statistics.mean([r[metric] for r in rows if r["population"] == "P_ORDINARY_ANSWERABLE"]) for metric in [f"Hit@{k}" for k in K_VALUES] + ["MRR"]} for name, rows in rows_by_dataset.items()})
    write_json(out / "no_map_diagnostics.json", {name: quantiles([r.get("TopScore", 0.0) for r in rows if r["population"] == "P_NO_MAP"]) for name, rows in rows_by_dataset.items()})
    tables = ROOT / "reports/tables/bm25_full_universe"
    write_csv(tables / "metric_population_counts.csv", [{"dataset": n, **population_counts(rows)} for n, rows in rows_by_dataset.items()])
    write_csv(tables / "bm25_runtime.csv", runtime_rows)
    write_csv(tables / "forward_ordinary_test.csv", rows_by_dataset["forward_stratified_test"])
    write_csv(tables / "backward_stratified_ordinary.csv", [r for r in rows_by_dataset["backward_stratified_test"] if r["population"] == "P_ORDINARY_ANSWERABLE"])
    write_csv(tables / "backward_family_ordinary.csv", [r for r in rows_by_dataset["backward_family_held_out_test"] if r["population"] == "P_ORDINARY_ANSWERABLE"])
    write_csv(tables / "no_map_diagnostics.csv", [{"dataset": n, **quantiles([r.get("TopScore", 0.0) for r in rows if r["population"] == "P_NO_MAP"])} for n, rows in rows_by_dataset.items()])
    write_json(out / "legacy_comparison.json", {"status": "DESCRIPTIVE_ONLY", "historical_status": "LEGACY_GEM_OBSERVED_TARGET_UNIVERSE", "corrected_status": "AUTHORITATIVE_FULL_TARGET_UNIVERSE"})
    write_json(out / "freeze_manifest.json", {"schema": "bm25-full-universe-freeze-v2", "evaluator_version": "retrieval_evaluator_v3", "terminology_version": "terminology_universe_v2", "bm25_parameters": {"k1": FROZEN[0], "b": FROZEN[1]}, "test_lock_sha256": sha256(ROOT / "artifacts/experiments/bm25_v2/test_lock.json"), "metric_population_audit_sha256": sha256(out / "metric_population_audit.json"), "historical_test_exposure_disclosed": True, "files": {str(p.relative_to(out)).replace("\\", "/"): sha256(p) for p in out.rglob("*") if p.is_file() and p.name != "freeze_manifest.json"}})
    print(json.dumps({"datasets": {n: len(r) for n, r in rows_by_dataset.items()}, "ordinary": {n: population_counts(r)["P_ORDINARY_ANSWERABLE"] for n, r in rows_by_dataset.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
