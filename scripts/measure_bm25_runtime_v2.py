# ruff: noqa: E501
from __future__ import annotations

import statistics
import time
from pathlib import Path

from run_bm25_v2 import load_examples, load_full_corpora, write_json

from shift_icd.retrieval.bm25 import BM25Index

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    examples = load_examples(ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl")
    corpora, _ = load_full_corpora()
    output = {}
    for direction, docs in corpora.items():
        start = time.perf_counter()
        index = BM25Index.from_documents(docs, 2.0, 0.75)
        build_seconds = time.perf_counter() - start
        queries = [e.source_label or "" for e in examples if e.direction == direction][:100]
        latencies = []
        for query in queries:
            t0 = time.perf_counter()
            index.rank(query, limit=100)
            latencies.append(time.perf_counter() - t0)
        ordered = sorted(latencies)
        output[direction] = {"index_concept_count": len(docs), "index_build_seconds": build_seconds, "query_sample_n": len(queries), "mean_query_latency_seconds": statistics.mean(latencies), "median_query_latency_seconds": statistics.median(latencies), "p95_query_latency_seconds": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], "queries_per_second": len(queries) / sum(latencies), "sampling_note": "first 100 benchmark queries in direction; no full-query latency claim"}
    write_json(ROOT / "artifacts/experiments/bm25_full_universe/runtime.json", {"evaluator_version": "retrieval_evaluator_v3", "bm25_parameters": {"k1": 2.0, "b": 0.75}, "directions": output, "peak_cpu_rss": "NOT_RECORDED"})
    print(output)


if __name__ == "__main__":
    main()
