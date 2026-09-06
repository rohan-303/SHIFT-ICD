# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.dense.corpus import BACKWARD, FORWARD
from shift_icd.dense.models import ModelSpec, load_encoder
from shift_icd.dense.text import clean_dense_text
from shift_icd.evaluation.metric_contract import classify_population, metric_values
from shift_icd.evaluation.retrieval import choice_list_recall_at_k, complete_scenario_retrieval_at_k
from shift_icd.terminology import TerminologyRecord

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/benchmarks/cms_track_a/v1.0"
OUT = Path(os.environ.get("DENSE_OUTPUT", str(ROOT / "artifacts/experiments/dense_full_universe_v2")))
EMBED = OUT / "embeddings"
TERM = ROOT / "artifacts/terminology_universe_v2"
K_VALUES = (1, 5, 10, 25, 50, 100)
SPECS = (
    ModelSpec("SapBERT", "cambridgeltl/SapBERT-from-PubMedBERT-fulltext", "090663c3ae57bf35ffe4d0d468a2a88d03051a4d", "transformers_cls", "Apache-2.0", 768),
    ModelSpec("BioLORD-2023", "FremyCompany/BioLORD-2023", "167aab527b238a50ca65224e6319215d2ff4fc9f", "sentence_transformer", "other (model-card terms)", 768),
    ModelSpec("MedCPT", "ncbi/MedCPT-Query-Encoder", "d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc", "transformers_cls", "other (repository license)", 768),
    ModelSpec("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "qwen_sentence_transformer", "Apache-2.0", 1024),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_examples() -> list[BenchmarkExample]:
    return [BenchmarkExample.model_validate_json(line) for line in (BENCHMARK / "all_examples.jsonl").open(encoding="utf-8") if line.strip()]


def build_corpora(rows: pd.DataFrame) -> dict[str, dict[str, str]]:
    corpora: dict[str, dict[str, str]] = {}
    for direction, filename, terminology in ((FORWARD, "icd10cm_diagnosis.jsonl", "ICD-10-CM"), (BACKWARD, "icd9cm_diagnosis.jsonl", "ICD-9-CM")):
        records = [TerminologyRecord(**json.loads(line)) for line in (ROOT / "artifacts/terminology_universe_v2" / filename).read_text(encoding="utf-8").splitlines() if line]
        corpora[direction] = {record.canonical_code: clean_dense_text(record.long_description or record.short_description or "") for record in records if record.terminology == terminology}
    return corpora


def source_text(example: BenchmarkExample) -> str:
    return clean_dense_text(example.source_label or "")


def population(examples: list[BenchmarkExample], direction: str, name: str) -> list[BenchmarkExample]:
    selected = [example for example in examples if example.direction == direction]
    if name == "stratified_dev":
        return [example for example in selected if example.split == "dev"]
    if name == "stratified_test":
        return [example for example in selected if example.split == "test"]
    if name == "family_held_out_test":
        return [example for example in selected if example.source_family_split == "test"]
    raise ValueError(name)


def choose_max_length(lengths: list[int]) -> int:
    if not lengths or sum(length <= 64 for length in lengths) / len(lengths) >= 0.999:
        return 64
    if sum(length <= 128 for length in lengths) / len(lengths) >= 0.999:
        return 128
    return max(lengths)


def encode_targets(encoder: Any, spec: ModelSpec, corpora: dict[str, dict[str, str]], max_length: int, batch_size: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    matrices: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for direction, docs in corpora.items():
        codes = sorted(docs)
        result = encoder.encode([docs[code] for code in codes], max_length, batch_size, is_query=False)
        matrix = result.embeddings.astype(np.float32, copy=False)
        path = EMBED / f"{spec.name.lower().replace('-', '_')}_{direction.lower()}_targets.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
        corpus_hash = hashlib.sha256("\n".join(f"{code}\t{docs[code]}" for code in codes).encode()).hexdigest()
        manifest = json.loads((TERM / ("icd10cm_manifest.json" if direction == FORWARD else "icd9cm_manifest.json")).read_text(encoding="utf-8"))
        if len(codes) != manifest["count"] or corpus_hash != manifest["code_description_hash"]:
            raise RuntimeError(f"authoritative terminology manifest mismatch for {direction}")
        dump_json(path.with_suffix(".json"), {"model_id": spec.model_id, "revision": spec.revision, "license": spec.license, "direction": direction, "terminology_version": "terminology_universe_v2", "terminology_release": manifest["version"], "code_hash": manifest["code_hash"], "code_description_hash": manifest["code_description_hash"], "retrieval_corpus_hash": manifest["corpus_sha256"], "corpus_hash": corpus_hash, "target_count": len(codes), "embedding_dim": int(matrix.shape[1]), "dtype": str(matrix.dtype), "normalized": True, "similarity": "normalized_dot_product_exact", "row_order_sha256": hashlib.sha256("\n".join(codes).encode()).hexdigest(), "codes": codes, "array_sha256": sha256(path), "token_length": {"max": max(result.token_lengths, default=0), "p95": float(np.percentile(result.token_lengths, 95)) if result.token_lengths else 0.0, "p99": float(np.percentile(result.token_lengths, 99)) if result.token_lengths else 0.0, "truncation_count": result.truncation_count}})
        matrices[direction] = matrix
        metadata[direction] = {"codes": codes, "path": str(path), "cache_sha256": sha256(path), "encode_seconds": result.elapsed_seconds, "token_lengths": result.token_lengths, "truncation_count": result.truncation_count}
    return matrices, metadata


def dense_rank(query: np.ndarray, matrix: np.ndarray, codes: list[str], top_k: int = 100) -> tuple[list[str], list[float]]:
    scores = matrix @ query
    return rank_scores(scores, codes, top_k)


def rank_scores(scores: np.ndarray, codes: list[str], top_k: int = 100) -> tuple[list[str], list[float]]:
    count = min(top_k, len(scores))
    candidates = np.argpartition(-scores, count - 1)[:count]
    ranked = sorted(((codes[int(index)], float(scores[int(index)])) for index in candidates), key=lambda item: (-item[1], item[0]))
    return [code for code, _score in ranked], [score for _code, score in ranked]


def evaluate_rows(examples: list[BenchmarkExample], encoder: Any, matrices: dict[str, np.ndarray], code_lists: dict[str, list[str]], max_length: int, batch_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries = [source_text(example) for example in examples]
    query_result = encoder.encode(queries, max_length, batch_size, is_query=True)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), 256):
        batch_examples = examples[start : start + 256]
        batch_queries = query_result.embeddings[start : start + 256]
        matrix = matrices[batch_examples[0].direction]
        scores_batch = batch_queries @ matrix.T
        codes = code_lists[batch_examples[0].direction]
        for example, scores in zip(batch_examples, scores_batch, strict=True):
            ranked, scores_list = rank_scores(scores, codes)
            valid = set(example.valid_target_codes)
            row: dict[str, Any] = {"benchmark_id": example.benchmark_id, "direction": example.direction, "mapping_kind": example.mapping_kind, "lexical_difficulty": example.lexical_metadata.get("lexical_difficulty", "UNKNOWN"), "no_map": example.no_map, "ranked_codes": ranked, "scores": scores_list}
            row.update(metric_values(valid, ranked, no_map=example.no_map, k_values=K_VALUES))
            for k in K_VALUES:
                row[f"ChoiceListRecall@{k}"] = choice_list_recall_at_k(example, ranked, k)
                row[f"CompleteScenarioRetrieval@{k}"] = complete_scenario_retrieval_at_k(example, ranked, k)
            row["population"] = classify_population(example)
            rows.append(row)
    return rows, {"query_seconds": query_result.elapsed_seconds, "query_token_lengths": query_result.token_lengths, "query_truncation_count": query_result.truncation_count}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    noncomb = [row for row in rows if row["population"] == "P_ORDINARY_ANSWERABLE"]
    complex_rows = [row for row in rows if row["population"] in {"P_COMBINATION", "P_COMBINATION_WITH_ALTERNATIVES", "P_COMPLEX"}]
    result: dict[str, Any] = {"n": len(rows), "P_ORDINARY_ANSWERABLE_n": len(noncomb), "P_COMPLEX_n": len(complex_rows)}
    for k in K_VALUES:
        result[f"Hit@{k}"] = statistics.mean([row[f"Hit@{k}"] for row in noncomb]) if noncomb else None
        result[f"ChoiceListRecall@{k}"] = statistics.mean([row[f"ChoiceListRecall@{k}"] for row in complex_rows]) if complex_rows else None
        result[f"CompleteScenarioRetrieval@{k}"] = statistics.mean([row[f"CompleteScenarioRetrieval@{k}"] for row in complex_rows]) if complex_rows else None
    result["MRR"] = statistics.mean([row["MRR"] for row in noncomb]) if noncomb else None
    return result


def summarize_populations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    populations = {
        "P_ORDINARY_ANSWERABLE": ["P_ORDINARY_ANSWERABLE"],
        "P_COMBINATION": ["P_COMBINATION"],
        "P_COMBINATION_WITH_ALTERNATIVES": ["P_COMBINATION_WITH_ALTERNATIVES"],
        "P_COMPLEX": ["P_COMBINATION", "P_COMBINATION_WITH_ALTERNATIVES", "P_COMPLEX"],
        "P_NO_MAP": ["P_NO_MAP"],
    }
    return {name: summarize([row for row in rows if row["population"] in members]) for name, members in populations.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    EMBED.mkdir(parents=True, exist_ok=True)
    rows = load_examples()
    corpora = build_corpora(pd.DataFrame())
    dump_json(OUT / "config.json", {"experiment": "dense_full_universe_v2", "benchmark_version": "1.0", "models": [spec.__dict__ for spec in SPECS], "top_k": 100, "seed": 20260830, "batch_size": {"transformers": 64, "sentence_transformer": 32}, "selection_criterion": ["noncombination_answerable_Hit@100", "CompleteScenarioRetrieval@100", "Hit@10", "MRR"]})
    git_commit = os.environ.get("DENSE_GIT_COMMIT")
    if not git_commit:
        raise RuntimeError("DENSE_GIT_COMMIT is required because packaged runs have no .git metadata")
    dump_json(OUT / "environment.json", {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "git_commit": git_commit, "benchmark_manifest_sha256": sha256(BENCHMARK / "manifest.json")})
    dump_json(OUT / "models.json", {"models": [spec.__dict__ for spec in SPECS], "provenance_document": "docs/models/dense_retrieval_v1.md"})
    dev_metrics: dict[str, Any] = {}
    requested = os.environ.get("DENSE_MODEL")
    selected_specs = [spec for spec in SPECS if requested is None or spec.name == requested]
    if len(selected_specs) != 1:
        raise ValueError(f"DENSE_MODEL must identify exactly one frozen model; got {requested!r}")
    for spec in selected_specs:
        started = time.perf_counter()
        encoder = load_encoder(spec, "cuda" if torch.cuda.is_available() else "cpu")
        all_texts = [source_text(example) for example in rows] + [text for docs in corpora.values() for text in docs.values()]
        raw_lengths = [len(encoder.tokenizer.encode(text, add_special_tokens=True, truncation=False)) for text in all_texts]
        max_length = choose_max_length(raw_lengths)
        batch_size = 32 if spec.kind != "transformers_cls" else 64
        matrices, cache_meta = encode_targets(encoder, spec, corpora, max_length, batch_size)
        code_lists = {direction: sorted(docs) for direction, docs in corpora.items()}
        dev_examples = population(rows, FORWARD, "stratified_dev")
        dev_rows, query_meta = evaluate_rows(dev_examples, encoder, matrices, code_lists, max_length, batch_size)
        rows_path = OUT / "dev_rows.jsonl"
        rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in dev_rows), encoding="utf-8")
        dev_metrics[spec.name] = {"summary": summarize(dev_rows), "summary_by_population": summarize_populations(dev_rows), "max_length": max_length, "token_audit": {"max": max(raw_lengths, default=0), "p95": float(np.percentile(raw_lengths, 95)) if raw_lengths else 0.0, "p99": float(np.percentile(raw_lengths, 99)) if raw_lengths else 0.0, "truncation_count": sum(length > max_length for length in raw_lengths)}, "cache": cache_meta, "query": query_meta, "mean_query_latency_ms": 1000.0 * query_meta["query_seconds"] / len(dev_examples), "p95_query_latency_ms": None, "elapsed_seconds": time.perf_counter() - started}
        encoder.close()
        dump_json(OUT / "dev_metrics.json", dev_metrics)
    print(json.dumps({name: value["summary"] for name, value in dev_metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
