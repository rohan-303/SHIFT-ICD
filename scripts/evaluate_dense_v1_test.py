# ruff: noqa: E501
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from run_dense_v1 import (
    BACKWARD,
    EMBED,
    FORWARD,
    K_VALUES,
    OUT,
    SPECS,
    build_corpora,
    dump_json,
    evaluate_rows,
    load_examples,
    population,
    source_text,
    summarize,
)

from shift_icd.evaluation.retrieval import choice_list_recall_at_k, complete_scenario_retrieval_at_k, hit_at_k
from shift_icd.retrieval.bm25 import BM25Index


def bm25_rows(examples: list[Any], corpora: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    indexes = {direction: BM25Index.from_documents(corpora[direction], 2.0, 0.75) for direction in (FORWARD, BACKWARD)}
    rows: list[dict[str, Any]] = []
    for example in examples:
        ranked_pairs = indexes[example.direction].rank(source_text(example), limit=100)
        ranked = [code for code, _score in ranked_pairs]
        scores = [float(score) for _code, score in ranked_pairs]
        valid = set(example.valid_target_codes)
        row: dict[str, Any] = {"benchmark_id": example.benchmark_id, "direction": example.direction, "mapping_kind": example.mapping_kind, "lexical_difficulty": example.lexical_metadata.get("lexical_difficulty", "UNKNOWN"), "no_map": example.no_map, "ranked_codes": ranked, "scores": scores}
        for k in K_VALUES:
            row[f"Hit@{k}"] = None if example.no_map else float(hit_at_k(valid, ranked, k))
            row[f"ChoiceListRecall@{k}"] = choice_list_recall_at_k(example, ranked, k)
            row[f"CompleteScenarioRetrieval@{k}"] = complete_scenario_retrieval_at_k(example, ranked, k)
        row["MRR"] = None if example.no_map else next((1.0 / rank for rank, code in enumerate(ranked, 1) if code in valid), 0.0)
        rows.append(row)
    return rows


def main() -> None:
    examples = load_examples()
    normalized = pd.read_parquet(Path(__file__).resolve().parents[1] / "data/processed/cms/2018_gem/normalized_rows.parquet")
    corpora = build_corpora(normalized)
    dev_meta = json.loads((OUT / "dev_metrics.json").read_text(encoding="utf-8"))
    test_metrics: dict[str, Any] = {}
    all_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    populations = {"forward_stratified_test": (FORWARD, "stratified_test"), "forward_family_held_out_test": (FORWARD, "family_held_out_test"), "backward_stratified_test": (BACKWARD, "stratified_test"), "backward_family_held_out_test": (BACKWARD, "family_held_out_test")}
    for spec in SPECS:
        encoder = __import__("shift_icd.dense.models", fromlist=["load_encoder"]).load_encoder(spec, "cuda")
        max_length = int(dev_meta[spec.name]["max_length"])
        batch_size = 32 if spec.kind != "transformers_cls" else 64
        matrices: dict[str, np.ndarray] = {}
        for direction in (FORWARD, BACKWARD):
            path = EMBED / f"{spec.name.lower().replace('-', '_')}_{direction.lower()}_targets.npy"
            matrices[direction] = np.load(path)
        code_lists = {direction: sorted(corpora[direction]) for direction in (FORWARD, BACKWARD)}
        test_metrics[spec.name] = {}
        all_rows[spec.name] = {}
        for population_name, (direction, split_name) in populations.items():
            selected = population(examples, direction, split_name)
            started = time.perf_counter()
            rows, query_meta = evaluate_rows(selected, encoder, matrices, code_lists, max_length, batch_size)
            test_metrics[spec.name][population_name] = {"scope": {"direction": direction, "protocol": "source_stratified_and_family_held_out", "partition": split_name, "benchmark_version": "1.0", "sample_population": population_name}, "summary": summarize(rows), "query": query_meta, "elapsed_seconds": time.perf_counter() - started}
            all_rows[spec.name][population_name] = rows
            ranking_path = OUT / "rankings" / spec.name.lower().replace("-", "_") / f"{population_name}.jsonl"
            ranking_path.parent.mkdir(parents=True, exist_ok=True)
            with ranking_path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
        encoder.close()
    dump_json(OUT / "test_metrics.json", test_metrics)
    bm25_all: dict[str, list[dict[str, Any]]] = {}
    for population_name, (direction, split_name) in populations.items():
        bm25_all[population_name] = bm25_rows(population(examples, direction, split_name), corpora)
    dump_json(OUT / "bm25_test_rows.json", bm25_all)
    selected_name = "BioLORD-2023"
    rrf_metrics: dict[str, Any] = {}
    for population_name in populations:
        hybrid_rows: list[dict[str, Any]] = []
        dense_by_id = {row["benchmark_id"]: row for row in all_rows[selected_name][population_name]}
        for bm_row in bm25_all[population_name]:
            dense_row = dense_by_id[bm_row["benchmark_id"]]
            fused = __import__("shift_icd.dense.retrieval", fromlist=["rrf_fuse"]).rrf_fuse(bm_row["ranked_codes"], dense_row["ranked_codes"], 60, 100)
            ranked = [code for code, _score in fused]
            example = next(example for example in examples if example.benchmark_id == bm_row["benchmark_id"])
            valid = set(example.valid_target_codes)
            row = {"benchmark_id": example.benchmark_id, "direction": example.direction, "mapping_kind": example.mapping_kind, "lexical_difficulty": example.lexical_metadata.get("lexical_difficulty", "UNKNOWN"), "no_map": example.no_map, "ranked_codes": ranked, "scores": [score for _code, score in fused]}
            for k in K_VALUES:
                row[f"Hit@{k}"] = None if example.no_map else float(hit_at_k(valid, ranked, k))
                row[f"ChoiceListRecall@{k}"] = choice_list_recall_at_k(example, ranked, k)
                row[f"CompleteScenarioRetrieval@{k}"] = complete_scenario_retrieval_at_k(example, ranked, k)
            row["MRR"] = None if example.no_map else next((1.0 / rank for rank, code in enumerate(ranked, 1) if code in valid), 0.0)
            hybrid_rows.append(row)
        rrf_metrics[population_name] = {"scope": {"direction": populations[population_name][0], "protocol": "source_stratified_and_family_held_out", "partition": populations[population_name][1], "benchmark_version": "1.0", "sample_population": population_name}, "summary": summarize(hybrid_rows)}
        path = OUT / "rankings" / "bm25_dense_rrf" / f"{population_name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for row in hybrid_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    dump_json(OUT / "rrf_metrics.json", {"model": selected_name, "rrf_k": 60, "metrics": rrf_metrics})
    print(json.dumps({"models": {name: {pop: value["summary"] for pop, value in pops.items()} for name, pops in test_metrics.items()}, "rrf": rrf_metrics}, indent=2))


if __name__ == "__main__":
    main()
