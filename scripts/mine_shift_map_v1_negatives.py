# ruff: noqa: E501
from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.dense.corpus import FORWARD, build_target_corpus
from shift_icd.dense.models import ModelSpec, load_encoder
from shift_icd.dense.text import clean_dense_text
from shift_icd.retrieval.bm25 import BM25Index, tokenize
from shift_icd.shift_map.mining import (
    audit_negative_collisions,
    mine_random_negatives,
    mine_ranked_negatives,
    mine_same_family_negatives,
    negative_manifest_hash,
)
from shift_icd.shift_map.training import TrainingExample

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/benchmarks/cms_track_a/v1.0"
OUT = ROOT / "artifacts/experiments/shift_map_v1"
TARGET_CACHE = ROOT / "artifacts/embeddings/dense_v1/biolord_2023_icd9cm_to_icd10cm_targets.npy"
TARGET_META = TARGET_CACHE.with_suffix(".json")
SPEC = ModelSpec(
    "BioLORD-2023", "FremyCompany/BioLORD-2023", "167aab527b238a50ca65224e6319215d2ff4fc9f", "sentence_transformer", "other", 768
)


def exact_top_bm25(index: Any, query: str, limit: int = 100) -> list[str]:
    query_terms = set(tokenize(query))
    scores: dict[str, float] = {}
    for term in query_terms:
        posting = index.postings.get(term)
        if not posting:
            continue
        idf = index._idf(index.document_frequency[term])
        for code, frequency in posting.items():
            length = len(index.tokens[code])
            norm = (
                index.k1 * (1 - index.b + index.b * length / index.average_document_length) if index.average_document_length else index.k1
            )
            scores[code] = scores.get(code, 0.0) + idf * frequency * (index.k1 + 1) / (frequency + norm)
    ranked = heapq.nsmallest(limit, scores.items(), key=lambda item: (-item[1], item[0]))
    result = [code for code, _score in ranked]
    if len(result) < limit:
        result.extend(code for code in sorted(index.documents) if code not in scores and code not in result[:limit])
    return result[:limit]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_rows() -> list[TrainingExample]:
    result = []
    with (BENCHMARK / "all_examples.jsonl").open(encoding="utf-8") as f:
        for line in f:
            x = BenchmarkExample.model_validate_json(line)
            if x.direction == "ICD9CM_TO_ICD10CM" and x.split == "train" and x.mapping_kind in {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"} and x.valid_target_codes:
                result.append(
                    TrainingExample(
                        x.benchmark_id,
                        x.source_code,
                        x.source_label or "",
                        x.source_family,
                        x.direction,
                        x.split or "",
                        x.source_family_split or "",
                        x.mapping_kind,
                        tuple(sorted(x.valid_target_codes)),
                        str(x.lexical_metadata.get("lexical_difficulty", "UNKNOWN")),
                    )
                )
    return result


def main() -> None:
    seed = 20260830
    rows = load_rows()
    normalized = pd.read_parquet(ROOT / "data/processed/cms/2018_gem/normalized_rows.parquet")
    forward_corpus = build_target_corpus(normalized, FORWARD)
    target_codes = list(forward_corpus.codes)
    target_families = {code: code.replace(".", "")[:3].upper() for code in target_codes}
    bm25 = BM25Index.from_documents(forward_corpus.as_dict(), 2.0, 0.75)

    target_matrix = np.load(TARGET_CACHE).astype(np.float32)
    target_meta = json.loads(TARGET_META.read_text(encoding="utf-8"))
    if target_meta["codes"] != target_codes:
        raise RuntimeError("frozen BioLORD target cache code order does not match canonical corpus")
    encoder = load_encoder(SPEC, "cuda" if torch.cuda.is_available() else "cpu")
    query_result = encoder.encode([clean_dense_text(row.source_label) for row in rows], 64, 32, is_query=True)
    query_embeddings = query_result.embeddings.astype(np.float32, copy=False)

    records: list[dict[str, Any]] = []
    all_negative_sets: dict[str, list[str]] = {}
    stats = {"sources": len(rows), "duplicates_removed": 0, "fallback_count": 0, "unavailable_same_family_count": 0}
    for row_index, row in enumerate(rows):
        scores = query_embeddings[row_index] @ target_matrix.T
        candidates = np.argpartition(-scores, 99)[:100]
        order = sorted(candidates.tolist(), key=lambda index: (-float(scores[index]), target_codes[index]))
        bm25_ranked = exact_top_bm25(bm25, row.source_label, 100)
        dense_ranked = [target_codes[int(i)] for i in order]
        random_neg = mine_random_negatives(row, target_codes, 4, seed)
        lexical = mine_ranked_negatives(row, bm25_ranked, 1)
        dense = mine_ranked_negatives(row, dense_ranked, 1)
        same_family, unavailable = mine_same_family_negatives(row, target_families, dense_ranked, target_codes, 1)
        if unavailable:
            stats["unavailable_same_family_count"] += 1
        mixed = []
        for candidate in lexical + dense + same_family:
            if candidate not in mixed:
                mixed.append(candidate)
        for candidate in random_neg:
            if len(mixed) >= 4:
                break
            if candidate not in mixed:
                mixed.append(candidate)
        all_negative_sets[row.benchmark_id] = mixed
        records.append(
            {
                "benchmark_id": row.benchmark_id,
                "source_code": row.source_code,
                "source_family": row.source_family,
                "valid_target_codes": list(row.valid_target_codes),
                "mapping_kind": row.mapping_kind,
                "lexical_difficulty": row.lexical_difficulty,
                "random": random_neg,
                "lexical_hard": lexical,
                "dense_hard": dense,
                "same_family_hard": same_family,
                "mixed": mixed,
                "dense_hard_score": float(scores[order[0]]) if order else None,
            }
        )
    audit = audit_negative_collisions({row.benchmark_id: row for row in rows}, all_negative_sets)
    if audit["remaining_gold_collisions"] != 0:
        raise RuntimeError(audit)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "negative_sets.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    manifest = {
        "experiment": "shift_map_v1",
        "benchmark_version": "1.0",
        "direction": "ICD9CM_TO_ICD10CM",
        "split_protocol": "stratified",
        "partition": "train",
        "source_count": len(rows),
        "source_ids_sha256": hashlib.sha256("\n".join(sorted(row.benchmark_id for row in rows)).encode()).hexdigest(),
        "negative_sets_sha256": negative_manifest_hash(all_negative_sets),
        "target_cache_sha256": sha256(TARGET_CACHE),
        "base_model_revision": SPEC.revision,
        "seed": seed,
        "strategies": ["N1_RANDOM", "N2_LEXICAL_HARD", "N3_DENSE_HARD", "N4_SAME_FAMILY_HARD", "N5_MIXED"],
        "audit": audit | stats,
        "test_data_used": False,
        "dev_data_used": False,
        "generation_script": "scripts/mine_shift_map_v1_negatives.py",
    }
    (OUT / "negative_mining_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    encoder.close()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
