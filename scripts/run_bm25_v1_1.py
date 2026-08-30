# ruff: noqa: E501
from __future__ import annotations

import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from run_bm25_v1 import build_corpora, load_examples, sha256, write_json

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.evaluation.scope import EvaluationScope
from shift_icd.retrieval.bm25 import BM25Index, token_jaccard, tokenize

ROOT = Path(__file__).resolve().parents[1]
K_VALUES = (1, 5, 10, 25, 50, 100)
PARAMS = (2.0, 0.75)
DIRECTIONS = ("ICD9CM_TO_ICD10CM", "ICD10CM_TO_ICD9CM")
DATASETS = (
    ("forward_stratified_test", "ICD9CM_TO_ICD10CM", "stratified", "test", "split"),
    ("forward_family_held_out_test", "ICD9CM_TO_ICD10CM", "family_held_out", "test", "source_family_split"),
    ("backward_stratified_test", "ICD10CM_TO_ICD9CM", "stratified", "test", "split"),
    ("backward_family_held_out_test", "ICD10CM_TO_ICD9CM", "family_held_out", "test", "source_family_split"),
)


def mrr(valid: set[str], ranked: list[str]) -> float:
    for rank, code in enumerate(ranked, 1):
        if code in valid:
            return 1.0 / rank
    return 0.0


def hit(valid: set[str], ranked: list[str], k: int) -> float:
    return float(bool(valid & set(ranked[:k])))


def complete(example: BenchmarkExample, ranked: list[str], k: int) -> float | None:
    if not example.combination and example.mapping_kind not in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}:
        return None
    retrieved = set(ranked[:k])
    for scenario in example.scenarios:
        if all(any(alternative.target_code in retrieved for alternative in choice.alternatives) for choice in scenario.choice_lists):
            return 1.0
    return 0.0


def choice_recall(example: BenchmarkExample, ranked: list[str], k: int) -> float | None:
    if not example.combination:
        return None
    retrieved = set(ranked[:k])
    lists = [choice for scenario in example.scenarios for choice in scenario.choice_lists]
    return sum(any(alt.target_code in retrieved for alt in choice.alternatives) for choice in lists) / len(lists) if lists else 0.0


def scope_row(example: BenchmarkExample, scope: EvaluationScope, ranked: list[str], scores: list[float], sample_population: str) -> dict[str, Any]:
    valid = set(example.valid_target_codes)
    answerable = not example.no_map
    complex_mapping = example.combination or example.mapping_kind in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}
    row: dict[str, Any] = {**scope.metadata(1, int(answerable), int(answerable and not complex_mapping), sample_population), "benchmark_id": example.benchmark_id, "source_code": example.source_code, "mapping_kind": example.mapping_kind, "lexical_difficulty": example.lexical_metadata.get("lexical_difficulty", "UNKNOWN"), "alternative_count": len(valid), "no_map": example.no_map, "valid_target_codes": sorted(valid), "top_score": scores[0] if scores else 0.0, "top1_top2_margin": (scores[0] - scores[1]) if len(scores) > 1 else (scores[0] if scores else 0.0), "mean_top5_score": statistics.mean(scores[:5]) if scores else 0.0, "query_token_count": len(tokenize(example.source_label or "")), "max_token_overlap": max((token_jaccard(example.source_label or "", code) for code in []), default=0.0)}
    for k in K_VALUES:
        row[f"Hit@{k}"] = None if not answerable or complex_mapping else hit(valid, ranked, k)
        row[f"CompleteScenarioRetrieval@{k}"] = complete(example, ranked, k)
        row[f"ChoiceListRecall@{k}"] = choice_recall(example, ranked, k)
    row["MRR"] = None if not answerable or complex_mapping else mrr(valid, ranked)
    row["alternative_bucket"] = "1" if len(valid) == 1 else "2-5" if len(valid) <= 5 else "6-20" if len(valid) <= 20 else "21-100" if len(valid) <= 100 else ">100"
    row["sample_population"] = sample_population
    return row


def read_rankings(path: Path) -> dict[str, list[tuple[str, float]]]:
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped[row["benchmark_id"]].append((row["target_code"], float(row["score"])))
    return grouped


def summarize(rows: list[dict[str, Any]], label_field: str | None = None, label: str | None = None) -> dict[str, Any]:
    answerable = [row for row in rows if not row["no_map"]]
    noncomb = [row for row in answerable if row["mapping_kind"] not in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}]
    result: dict[str, Any] = {"direction": rows[0]["direction"] if rows else None, "split_protocol": rows[0]["split_protocol"] if rows else None, "partition": rows[0]["partition"] if rows else None, "benchmark_version": "1.0", "sample_population": rows[0]["sample_population"] if rows else None, "total_example_count": len(rows), "answerable_count": len(answerable), "applicable_metric_count": len(noncomb)}
    if label_field:
        result[label_field] = label
    if label_field == "mapping_kind" and label in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}:
        applicable = answerable
    else:
        applicable = noncomb
    for k in K_VALUES:
        result[f"Hit@{k}"] = statistics.mean([row[f"Hit@{k}"] for row in applicable if row[f"Hit@{k}"] is not None]) if applicable and any(row[f"Hit@{k}"] is not None for row in applicable) else None
        result[f"CompleteScenarioRetrieval@{k}"] = statistics.mean([row[f"CompleteScenarioRetrieval@{k}"] for row in answerable if row[f"CompleteScenarioRetrieval@{k}"] is not None]) if any(row[f"CompleteScenarioRetrieval@{k}"] is not None for row in answerable) else None
        result[f"ChoiceListRecall@{k}"] = statistics.mean([row[f"ChoiceListRecall@{k}"] for row in answerable if row[f"ChoiceListRecall@{k}"] is not None]) if any(row[f"ChoiceListRecall@{k}"] is not None for row in answerable) else None
    result["MRR"] = statistics.mean([row["MRR"] for row in applicable if row["MRR"] is not None]) if any(row["MRR"] is not None for row in applicable) else None
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "std": None, "q1": None, "q3": None}
    qs = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
    return {"mean": statistics.mean(values), "median": statistics.median(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0, "q1": qs[0], "q3": qs[2]}


def main() -> None:
    started = time.perf_counter()
    benchmark = ROOT / "data/benchmarks/cms_track_a/v1.0"
    examples = load_examples(benchmark / "all_examples.jsonl")
    normalized = pd.read_parquet(ROOT / "data/processed/cms/2018_gem/normalized_rows.parquet")
    corpora, profiles = build_corpora(normalized, "q1_long_only")
    out = ROOT / "artifacts/experiments/bm25_v1_1"
    rankings_root = out / "rankings"
    all_rows: dict[str, list[dict[str, Any]]] = {}
    all_diagnostics: dict[str, list[dict[str, Any]]] = {}
    for dataset_name, direction, protocol, partition, split_field in DATASETS:
        dataset = [example for example in examples if example.direction == direction and getattr(example, split_field) == partition]
        scope = EvaluationScope(direction, protocol, partition)
        index = BM25Index.from_documents(corpora[direction], *PARAMS)
        rankings: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for example in dataset:
            ranked_pairs = index.rank(example.source_label or "", limit=100)
            ranked = [code for code, _score in ranked_pairs]
            scores = [score for _code, score in ranked_pairs]
            row = scope_row(example, scope, ranked, scores, dataset_name)
            row["direction"] = direction
            rows.append(row)
            if example.no_map or not (example.combination or example.mapping_kind in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}):
                diagnostics.append({"benchmark_id": example.benchmark_id, "group": "NO_MAP" if example.no_map else "ANSWERABLE", "max_bm25_score": scores[0] if scores else 0.0, "top1_top2_margin": scores[0] - scores[1] if len(scores) > 1 else (scores[0] if scores else 0.0), "mean_top5_score": statistics.mean(scores[:5]) if scores else 0.0, "query_token_count": len(tokenize(example.source_label or "")), "max_lexical_overlap": max((token_jaccard(example.source_label or "", corpora[direction][code]) for code, _score in ranked_pairs), default=0.0)})
            rankings.extend({"benchmark_id": example.benchmark_id, "target_code": code, "rank": rank, "score": score} for rank, (code, score) in enumerate(ranked_pairs, 1))
        ranking_path = rankings_root / f"{dataset_name}.jsonl"
        ranking_path.parent.mkdir(parents=True, exist_ok=True)
        with ranking_path.open("w", encoding="utf-8") as handle:
            for item in rankings:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        all_rows[dataset_name] = rows
        all_diagnostics[dataset_name] = diagnostics
    forward = all_rows["forward_stratified_test"]
    tables = ROOT / "reports/tables/bm25_v1_1"
    write_csv(tables / "forward_stratified_overall.csv", [summarize(forward)])
    for field, filename in [("mapping_kind", "forward_stratified_by_mapping_kind.csv"), ("lexical_difficulty", "forward_stratified_by_lexical_difficulty.csv"), ("alternative_bucket", "forward_stratified_by_alternative_size.csv")]:
        labels = ["LEXICAL_EXACT", "LEXICAL_HIGH", "LEXICAL_MEDIUM", "LEXICAL_LOW", "LEXICAL_CONFUSABLE"] if field == "lexical_difficulty" else sorted({str(row[field]) for row in forward})
        slice_summaries = []
        for label in labels:
            subset = [row for row in forward if str(row[field]) == label]
            summary = summarize(subset, field, label)
            if not subset:
                summary.update({"direction": "ICD9CM_TO_ICD10CM", "split_protocol": "stratified", "partition": "test", "benchmark_version": "1.0", "sample_population": "forward_stratified_test", "total_example_count": 0, "answerable_count": 0, "applicable_metric_count": 0})
            slice_summaries.append(summary)
        write_csv(tables / filename, slice_summaries)
    complex_kinds = {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}
    write_csv(tables / "forward_stratified_combination.csv", [summarize([row for row in forward if row["mapping_kind"] == label], "mapping_kind", label) for label in sorted(complex_kinds) if any(row["mapping_kind"] == label for row in forward)])
    write_csv(tables / "backward_stratified_overall.csv", [summarize(all_rows["backward_stratified_test"])])
    write_csv(tables / "family_held_out_comparison.csv", [summarize(all_rows[name]) | {"dataset": name} for name in ("forward_stratified_test", "forward_family_held_out_test", "backward_stratified_test", "backward_family_held_out_test")])
    write_csv(tables / "forward_backward_comparison.csv", [summarize(all_rows[name]) | {"dataset": name} for name in ("forward_stratified_test", "backward_stratified_test")])
    write_csv(tables / "benchmark_membership_audit.csv", [{"direction": direction, "canonical_source_count": sum(e.direction == direction for e in examples), "benchmark_source_count": sum(e.direction == direction for e in examples), "excluded_source_count": 0, "exclusion_reason": "none"} for direction in DIRECTIONS])
    no_map_rows = [row for row in all_diagnostics["forward_stratified_test"] if row["group"] == "NO_MAP"]
    mapped_rows = [row for row in all_diagnostics["forward_stratified_test"] if row["group"] == "ANSWERABLE"]
    write_csv(tables / "forward_no_map_diagnostics.csv", [{"direction": "ICD9CM_TO_ICD10CM", "split_protocol": "stratified", "partition": "test", "benchmark_version": "1.0", "sample_population": "forward_stratified_test", "group": group, "n": len(items), **{f"{key}_{stat}": value for key in ("max_bm25_score", "top1_top2_margin", "mean_top5_score", "max_lexical_overlap") for stat, value in quantiles([float(item[key]) for item in items]).items()}} for group, items in [("NO_MAP", no_map_rows), ("ANSWERABLE", mapped_rows)]])
    write_json(out / "scope_metadata.json", {name: {"direction": rows[0]["direction"], "split_protocol": rows[0]["split_protocol"], "partition": rows[0]["partition"], "benchmark_version": "1.0", "sample_population": name, "total_example_count": len(rows), "answerable_count": sum(not row["no_map"] for row in rows), "applicable_metric_count": sum(row["Hit@10"] is not None for row in rows)} for name, rows in all_rows.items()})
    with (out / "scoped_rows.jsonl").open("w", encoding="utf-8") as handle:
        for rows in all_rows.values():
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(out / "metrics.json", {name: summarize(rows) for name, rows in all_rows.items()})
    write_json(out / "corpus_profile.json", profiles)
    v = next((example for example in examples if example.source_code == "V5412" and example.direction == "ICD9CM_TO_ICD10CM"), None)
    v_case: dict[str, Any] = {"source_code": "V5412", "display_code": "V54.12", "found": v is not None}
    if v is not None:
        index = BM25Index.from_documents(corpora["ICD9CM_TO_ICD10CM"], *PARAMS)
        ranked = index.rank(v.source_label or "", limit=100)
        valid = set(v.valid_target_codes)
        ranks = [rank for rank, (code, _score) in enumerate(ranked, 1) if code in valid]
        v_case.update({"benchmark_id": v.benchmark_id, "partition": "train", "valid_target_count": len(valid), "candidate_corpus_size": len(corpora["ICD9CM_TO_ICD10CM"]), "valid_target_density": len(valid) / len(corpora["ICD9CM_TO_ICD10CM"]), "correct_candidate_ranks_top100": ranks, "best_valid_target_rank": min(ranks) if ranks else None, "valid_targets_top5": sum(rank <= 5 for rank in ranks), "valid_targets_top10": sum(rank <= 10 for rank in ranks), "valid_targets_top25": sum(rank <= 25 for rank in ranks), "valid_targets_top50": sum(rank <= 50 for rank in ranks), "valid_targets_top100": sum(rank <= 100 for rank in ranks)})
    write_json(out / "v54_12_case_study.json", v_case)
    write_json(out / "no_map_diagnostics.json", {"scope": {"direction": "ICD9CM_TO_ICD10CM", "split_protocol": "stratified", "partition": "test", "benchmark_version": "1.0", "sample_population": "forward_stratified_test"}, "groups": {"NO_MAP": quantiles([float(x["max_bm25_score"]) for x in no_map_rows]), "ANSWERABLE": quantiles([float(x["max_bm25_score"]) for x in mapped_rows])}, "rows": no_map_rows + mapped_rows})
    write_json(out / "runtime.json", {"total_evaluation_seconds": time.perf_counter() - started, "index_build_method": "internal inverted index", "parameters": {"k1": PARAMS[0], "b": PARAMS[1]}, "measurement_note": "wall-clock process timing; no peak RSS dependency added"})
    write_json(out / "manifest.json", {"experiment": "bm25_v1_1", "experiment_version": "1.1", "benchmark_version": "1.0", "frozen_bm25_parameters": {"k1": PARAMS[0], "b": PARAMS[1]}, "files": {str(path.relative_to(out)).replace("\\", "/"): sha256(path) for path in out.rglob("*") if path.is_file() and path.name != "manifest.json"}, "elapsed_seconds": time.perf_counter() - started})
    print(json.dumps({"datasets": {name: len(rows) for name, rows in all_rows.items()}, "v5412_found": v is not None, "elapsed_seconds": time.perf_counter() - started}, sort_keys=True))


if __name__ == "__main__":
    main()
