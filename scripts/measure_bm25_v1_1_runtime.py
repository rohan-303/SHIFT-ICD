# ruff: noqa: E501
from __future__ import annotations

import json
import pickle
import statistics
import time
from pathlib import Path

from run_bm25_v1 import build_corpora, load_examples

from shift_icd.retrieval.bm25 import BM25Index

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    benchmark = ROOT / "data/benchmarks/cms_track_a/v1.0"
    examples = [x for x in load_examples(benchmark / "all_examples.jsonl") if x.direction == "ICD9CM_TO_ICD10CM" and x.split == "test"]
    import pandas as pd  # type: ignore[import-untyped]
    normalized = pd.read_parquet(ROOT / "data/processed/cms/2018_gem/normalized_rows.parquet")
    corpora, _profiles = build_corpora(normalized, "q1_long_only")
    process_rss = None
    try:
        import psutil  # type: ignore[import-untyped]
        process_rss = psutil.Process().memory_info().rss
    except ImportError:
        pass
    start = time.perf_counter()
    index = BM25Index.from_documents(corpora["ICD9CM_TO_ICD10CM"], 2.0, 0.75)
    build_seconds = time.perf_counter() - start
    serialized_size = len(pickle.dumps(index, protocol=pickle.HIGHEST_PROTOCOL))
    latencies = []
    for example in examples:
        qstart = time.perf_counter()
        index.rank(example.source_label or "", limit=100)
        latencies.append((time.perf_counter() - qstart) * 1000)
    latencies.sort()
    p95 = latencies[max(0, min(len(latencies) - 1, int(len(latencies) * 0.95) - 1))]
    payload = {"direction": "ICD9CM_TO_ICD10CM", "split_protocol": "stratified", "partition": "test", "sample_population": "forward_stratified_test", "index_build_seconds": build_seconds, "index_serialized_bytes": serialized_size, "process_peak_rss_bytes": process_rss, "query_count": len(latencies), "average_query_latency_ms": statistics.mean(latencies), "median_query_latency_ms": statistics.median(latencies), "p95_query_latency_ms": p95, "throughput_queries_per_second": 1000.0 / statistics.mean(latencies), "methodology": "single-process wall-clock timing; queries ranked to Top-100; RSS is process RSS at measurement point when psutil is available; not used for tuning"}
    out = ROOT / "artifacts/experiments/bm25_v1_1/runtime_measurement.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
