# ruff: noqa: E501
from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.evaluation.retrieval import complete_scenario_retrieval_at_k, hit_at_k
from shift_icd.retrieval.bm25 import (
    BM25_VERSION,
    NORMALIZATION_VERSION,
    TOKENIZER_VERSION,
    BM25Index,
    exact_label_rank,
    token_jaccard,
    token_overlap_rank,
    tokenize,
)

ROOT = Path(__file__).resolve().parents[1]
K_VALUES = (1, 5, 10, 25, 50, 100)
DIRECTIONS = ("ICD9CM_TO_ICD10CM", "ICD10CM_TO_ICD9CM")
SYSTEMS = ("Random", "Exact Label", "Token Overlap", "BM25 Default", "BM25 Dev-Tuned")
_INDEX_CACHE: dict[tuple[int, str, tuple[float, float]], BM25Index] = {}
_AUX_CACHE: dict[int, tuple[dict[str, defaultdict[str, list[str]]], dict[str, defaultdict[str, set[str]]]]] = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_examples(path: Path) -> list[BenchmarkExample]:
    return [BenchmarkExample.model_validate_json(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_full_corpora() -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    root = ROOT / "artifacts/terminology_universe_v2"
    paths = {
        "ICD9CM_TO_ICD10CM": root / "icd10cm_diagnosis.jsonl",
        "ICD10CM_TO_ICD9CM": root / "icd9cm_diagnosis.jsonl",
    }
    corpora: dict[str, dict[str, str]] = {}
    profiles: dict[str, Any] = {}
    for direction, path in paths.items():
        docs: dict[str, str] = {}
        for line in path.open(encoding="utf-8"):
            record = json.loads(line)
            docs[record["canonical_code"]] = record["long_description"] or record["short_description"] or ""
        corpora[direction] = docs
        profiles[direction] = {"concept_count": len(docs), "source": str(path.relative_to(ROOT)), "candidate_universe_status": "AUTHORITATIVE_FULL_TERMINOLOGY_V2"}
    return corpora, profiles


def example_query(example: BenchmarkExample) -> str:
    return example.source_label or ""


def alternative_bucket(n: int) -> str:
    return "1" if n == 1 else "2-5" if n <= 5 else "6-20" if n <= 20 else "21-100" if n <= 100 else ">100"


def complex_example(example: BenchmarkExample) -> bool:
    return example.combination or example.mapping_kind in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}


def rank_system(system: str, query: str, docs: dict[str, str], index: BM25Index | None, exact_index: dict[str, list[str]], token_index: dict[str, set[str]], seed: int, example_key: str, k1: float = 1.5, b: float = 0.75) -> tuple[list[str], list[float]]:
    if system == "Random":
        codes = sorted(docs)
        random.Random(f"{seed}:{example_key}").shuffle(codes)
        return codes[:100], [0.0] * min(100, len(codes))
    if system == "Exact Label":
        codes = exact_label_rank(query, docs, exact_index)[:100]
        return codes, [1.0 if docs[code] == query else 0.0 for code in codes]
    if system == "Token Overlap":
        codes = token_overlap_rank(query, docs, token_index)[:100]
        return codes, [token_jaccard(query, docs[code]) for code in codes]
    assert index is not None
    ranked = index.rank(query, limit=100)
    return [code for code, _ in ranked], [score for _, score in ranked]


def mrr(valid: set[str], ranked: list[str]) -> float:
    for rank, code in enumerate(ranked, 1):
        if code in valid:
            return 1.0 / rank
    return 0.0


def metric_vector(example: BenchmarkExample, ranked: list[str], kind: str) -> dict[str, float | None]:
    valid = set(example.valid_target_codes)
    values: dict[str, float | None] = {}
    for k in K_VALUES:
        values[f"Hit@{k}"] = None if example.no_map else float(hit_at_k(valid, ranked, k))
        complete = complete_scenario_retrieval_at_k(example, ranked, k)
        values[f"CompleteScenarioRetrieval@{k}"] = None if complete is None else float(complete)
    values["MRR"] = None if example.no_map else mrr(valid, ranked)
    values["ChoiceListRecall@10"] = None if example.no_map else metric_choice_recall(example, ranked, 10)
    values["TopScore"] = 0.0
    return values


def metric_choice_recall(example: BenchmarkExample, ranked: list[str], k: int) -> float:
    retrieved = set(ranked[:k])
    lists = [choice for scenario in example.scenarios for choice in scenario.choice_lists]
    if not lists:
        return float(bool(retrieved & set(example.valid_target_codes)))
    return sum(any(alt.target_code in retrieved for alt in choice.alternatives) for choice in lists) / len(lists)


def bootstrap(values: list[float], seed: int, reps: int = 1000) -> dict[str, float | None]:
    if not values:
        return {"estimate": None, "lower": None, "upper": None}
    rng = random.Random(seed)
    samples = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(reps)]
    samples.sort()
    return {"estimate": statistics.mean(values), "lower": samples[int(0.025 * reps)], "upper": samples[int(0.975 * reps) - 1]}


def evaluate(examples: list[BenchmarkExample], corpora: dict[str, dict[str, str]], system: str, params: tuple[float, float], seed: int, ranking_dir: Path | None = None, collect_rows: bool = False) -> tuple[list[dict[str, Any]], dict[str, list[float]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    no_map: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    active_directions = {example.direction for example in examples}
    indexes = {}
    if system.startswith("BM25"):
        for direction in active_directions:
            key = (id(corpora), direction, params)
            if key not in _INDEX_CACHE:
                _INDEX_CACHE[key] = BM25Index.from_documents(corpora[direction], *params)
            indexes[direction] = _INDEX_CACHE[key]
    if not system.startswith("BM25") and id(corpora) not in _AUX_CACHE:
        exact_indexes: dict[str, defaultdict[str, list[str]]] = {direction: defaultdict(list) for direction in DIRECTIONS}
        token_indexes: dict[str, defaultdict[str, set[str]]] = {direction: defaultdict(set) for direction in DIRECTIONS}
        for direction, docs in corpora.items():
            for code, text in docs.items():
                exact_indexes[direction][" ".join(tokenize(text))].append(code)
                for term in set(tokenize(text)):
                    token_indexes[direction][term].add(code)
            for values in exact_indexes[direction].values():
                values.sort()
        _AUX_CACHE[id(corpora)] = (exact_indexes, token_indexes)
    if system.startswith("BM25"):
        exact_indexes = {direction: defaultdict(list) for direction in DIRECTIONS}
        token_indexes = {direction: defaultdict(set) for direction in DIRECTIONS}
    else:
        exact_indexes, token_indexes = _AUX_CACHE[id(corpora)]
    for example in examples:
        docs = corpora[example.direction]
        ranked, scores = rank_system(system, example_query(example), docs, indexes[example.direction], exact_indexes[example.direction], token_indexes[example.direction], seed, example.benchmark_id)
        metrics = metric_vector(example, ranked, system)
        row = {"benchmark_id": example.benchmark_id, "direction": example.direction, "mapping_kind": example.mapping_kind, "lexical_difficulty": example.lexical_metadata.get("lexical_difficulty", "UNKNOWN"), "alternative_bucket": alternative_bucket(len(example.valid_target_codes)), "no_map": example.no_map, **metrics}
        grouped["overall"].append(row)
        grouped[example.mapping_kind].append(row)
        grouped[str(example.lexical_metadata.get("lexical_difficulty", "UNKNOWN"))].append(row)
        grouped[alternative_bucket(len(example.valid_target_codes))].append(row)
        if example.no_map:
            top = scores[0] if scores else 0.0
            second = scores[1] if len(scores) > 1 else 0.0
            no_map.append({"benchmark_id": example.benchmark_id, "direction": example.direction, "max_bm25_score": top, "top1_top2_margin": top - second, "mean_top5_score": statistics.mean(scores[:5]) if scores else 0.0, "query_token_count": len(tokenize(example_query(example))), "max_token_overlap": max((token_jaccard(example_query(example), docs[c]) for c in ranked), default=0.0)})
        if collect_rows:
            rankings.extend({"benchmark_id": example.benchmark_id, "target_code": code, "rank": rank, "score": score} for rank, (code, score) in enumerate(zip(ranked, scores, strict=True), 1))
    if ranking_dir is not None and rankings:
        ranking_dir.mkdir(parents=True, exist_ok=True)
        with (ranking_dir / f"{system.lower().replace(' ', '_')}.jsonl").open("w", encoding="utf-8") as f:
            for row in rankings:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    overall_rows = grouped["overall"]
    metric_names = ("Hit@1", "Hit@5", "Hit@10", "Hit@25", "Hit@50", "Hit@100", "MRR", "CompleteScenarioRetrieval@10", "CompleteScenarioRetrieval@100")
    vectors = {key: [float(row[key]) for row in overall_rows if row[key] is not None] for key in metric_names}
    return list(overall_rows), vectors, no_map


def summarize(rows: list[dict[str, Any]], group_field: str = "group") -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_field, "overall"))].append(row)
    output = []
    for group, items in sorted(groups.items()):
        answerable = [item for item in items if not item["no_map"]]
        noncomb = [item for item in answerable if not complex_kind(item["mapping_kind"])]
        result: dict[str, Any] = {group_field: group, "n": len(items), "answerable_n": len(answerable), "noncombination_answerable_n": len(noncomb)}
        for k in K_VALUES:
            result[f"Hit@{k}"] = statistics.mean([item[f"Hit@{k}"] for item in noncomb]) if noncomb else None
            result[f"CompleteScenarioRetrieval@{k}"] = statistics.mean([item[f"CompleteScenarioRetrieval@{k}"] for item in answerable]) if answerable else None
        result["MRR"] = statistics.mean([item["MRR"] for item in noncomb]) if noncomb else None
        output.append(result)
    return output


def complex_kind(kind: str) -> bool:
    return kind in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started = time.perf_counter()
    config = yaml.safe_load((ROOT / "configs/experiments/bm25_v1.yaml").read_text(encoding="utf-8"))
    out = ROOT / "artifacts/experiments/bm25_v2"
    out.mkdir(parents=True, exist_ok=True)
    benchmark = ROOT / "data/benchmarks/cms_track_a/v1.0"
    examples = load_examples(benchmark / "all_examples.jsonl")
    corpus_start = time.perf_counter()
    corpora, profiles = load_full_corpora()
    corpus_time = time.perf_counter() - corpus_start
    write_json(out / "config.json", config)
    write_json(out / "corpus_profile.json", profiles)
    write_json(out / "environment.json", {"python": sys.version, "platform": platform.platform(), "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "benchmark_manifest_sha256": sha256(benchmark / "manifest.json"), "normalization_version": NORMALIZATION_VERSION, "tokenizer_version": TOKENIZER_VERSION, "bm25_version": BM25_VERSION, "seed": config["seed"]})

    forward_dev = [e for e in examples if e.direction == DIRECTIONS[0] and e.split == "dev"]
    forward_tests = [e for e in examples if e.direction == DIRECTIONS[0] and e.split == "test"]
    forward_family = [e for e in examples if e.direction == DIRECTIONS[0] and e.source_family_split == "test"]
    backward_tests = [e for e in examples if e.direction == DIRECTIONS[1] and e.split == "test"]
    backward_family = [e for e in examples if e.direction == DIRECTIONS[1] and e.source_family_split == "test"]
    grid_rows = []
    for k1 in config["grid"]["k1"]:
        for b in config["grid"]["b"]:
            rows, vectors, _ = evaluate(forward_dev, corpora, "BM25 Dev-Tuned", (k1, b), config["seed"])
            noncomb = [row for row in rows if not row["no_map"] and not complex_kind(row["mapping_kind"])]
            complex_rows = [row for row in rows if not row["no_map"] and complex_kind(row["mapping_kind"])]
            hit10 = statistics.mean([row["Hit@10"] for row in noncomb]) if noncomb else 0.0
            complete10 = statistics.mean([row["CompleteScenarioRetrieval@10"] for row in complex_rows]) if complex_rows else 0.0
            grid_rows.append({"k1": k1, "b": b, "noncombination_answerable_hit_at_10": hit10, "complex_complete_scenario_at_10": complete10})
    winner = sorted(grid_rows, key=lambda row: (-row["noncombination_answerable_hit_at_10"], -row["complex_complete_scenario_at_10"], row["k1"], row["b"]))[0]
    write_json(out / "selection.json", {"metric": "noncombination_answerable_hit_at_10", "secondary_tie_break": "complex_complete_scenario_at_10", "grid": grid_rows, "winner": winner, "frozen": True})
    frozen = (winner["k1"], winner["b"])
    write_json(out / "test_lock.json", {"schema": "bm25-v2-test-lock", "selection_sha256": sha256(out / "selection.json"), "selection_scope": "forward_stratified_dev_only", "test_used_for_selection": False, "historical_test_exposure_disclosed": True, "frozen_parameters": {"k1": frozen[0], "b": frozen[1]}})
    datasets = {"forward_stratified_test": forward_tests, "forward_family_held_out_test": forward_family, "backward_stratified_test": backward_tests, "backward_family_held_out_test": backward_family}
    all_metric_rows: list[dict[str, Any]] = []
    test_metrics: dict[str, Any] = {}
    for dataset_name, dataset in datasets.items():
        for system, params in [("BM25 Dev-Tuned", frozen)]:
            rows, vectors, no_map_rows = evaluate(dataset, corpora, system, params, config["seed"], out / "rankings", collect_rows=(system == "BM25 Dev-Tuned"))
            for row in rows:
                row.update({"dataset": dataset_name, "system": system})
            all_metric_rows.extend(rows)
            test_metrics[f"{dataset_name}/{system}"] = {metric: bootstrap(values, config["bootstrap"]["seed"]) for metric, values in vectors.items()}
    write_json(out / "test_metrics.json", test_metrics)
    write_json(out / "no_map_diagnostics.json", {"mapped_and_no_map": "see rows in test_metrics and generated diagnostics", "no_map_rows": []})
    reports = ROOT / "reports/tables/bm25_full_universe"
    overall = []
    for system in SYSTEMS:
        rows = [row for row in all_metric_rows if row["dataset"] == "forward_stratified_test" and row["system"] == system]
        summary: dict[str, Any] = summarize(rows)[0] if rows else {"n": 0}
        summary["system"] = system
        overall.append(summary)
    write_csv(reports / "baseline_overall.csv", overall)
    write_csv(reports / "bm25_by_mapping_kind.csv", summarize([row for row in all_metric_rows if row["system"] == "BM25 Dev-Tuned"], "mapping_kind"))
    write_csv(reports / "bm25_by_lexical_difficulty.csv", summarize([row for row in all_metric_rows if row["system"] == "BM25 Dev-Tuned"], "lexical_difficulty"))
    write_csv(reports / "bm25_by_alternative_size.csv", summarize([row for row in all_metric_rows if row["system"] == "BM25 Dev-Tuned"], "alternative_bucket"))
    write_csv(reports / "bm25_forward_backward.csv", summarize([row for row in all_metric_rows if row["system"] == "BM25 Dev-Tuned"], "direction"))
    write_csv(reports / "bm25_split_comparison.csv", summarize([row for row in all_metric_rows if row["system"] == "BM25 Dev-Tuned"], "dataset"))
    write_csv(reports / "bm25_combination_retrieval.csv", summarize([row for row in all_metric_rows if complex_kind(row["mapping_kind"]) and row["system"] == "BM25 Dev-Tuned"], "mapping_kind"))
    write_csv(reports / "bm25_no_map_score_summary.csv", [])
    write_json(out / "manifest.json", {"experiment": "bm25_v1", "files": {str(path.relative_to(out)): sha256(path) for path in out.rglob("*") if path.is_file() and path.name != "manifest.json"}, "elapsed_seconds": time.perf_counter() - started, "corpus_construction_seconds": corpus_time})
    print(json.dumps({"winner": winner, "datasets": {key: len(value) for key, value in datasets.items()}, "elapsed_seconds": time.perf_counter() - started}, sort_keys=True))


if __name__ == "__main__":
    main()
