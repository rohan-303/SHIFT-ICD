# ruff: noqa
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.dense.corpus import FORWARD, build_target_corpus_from_terminology
from shift_icd.terminology import TerminologyCorpus, TerminologyRecord
from shift_icd.dense.models import ModelSpec, load_encoder
from shift_icd.dense.text import clean_dense_text
from shift_icd.retrieval.bm25 import BM25Index, tokenize
from shift_icd.shift_map.mining import (
    audit_negative_collisions,
    mine_random_negatives,
    mine_ranked_negatives,
)
from shift_icd.shift_map.training import TrainingExample

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"
TARGET_CACHE = Path(__import__("os").environ["SHIFT_MAP_BIOLORD_TARGET_CACHE"])
TARGET_META = TARGET_CACHE.with_suffix(".json")
OUT = ROOT / "artifacts/experiments/shift_map_full_universe"
MODEL_SPEC = ModelSpec(
    "BioLORD-2023", "FremyCompany/BioLORD-2023", "167aab527b238a50ca65224e6319215d2ff4fc9f",
    "sentence_transformer", "other", 768,
)
UNIVERSE_HASH = "32f572abeb2ff4cb003145216fa83bfd9984fab96579205f432993aa9c550464"
TARGET_COUNT = 71704
SEED = 20260906
_BM25_INDEX: BM25Index | None = None
_BM25_NORMS: dict[str, float] = {}
_BM25_IDF: dict[str, float] = {}


def _init_bm25(index: BM25Index) -> None:
    global _BM25_INDEX, _BM25_NORMS, _BM25_IDF
    _BM25_INDEX = index
    average = index.average_document_length
    _BM25_NORMS = {
        code: index.k1 * (1 - index.b + index.b * len(tokens) / average) if average else index.k1
        for code, tokens in index.tokens.items()
    }
    _BM25_IDF = {
        term: index._idf(df) for term, df in index.document_frequency.items()
    }


def _rank_bm25(query: str) -> list[str]:
    if _BM25_INDEX is None:
        raise RuntimeError("BM25 worker was not initialized")
    return exact_top_bm25(_BM25_INDEX, query)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exact_top_bm25(index: BM25Index, query: str, limit: int = 100) -> list[str]:
    query_terms = set(tokenize(query))
    scores: dict[str, float] = {}
    norms = _BM25_NORMS
    idfs = _BM25_IDF
    for term in query_terms:
        posting = index.postings.get(term)
        if not posting:
            continue
        idf = idfs[term]
        for code, frequency in posting.items():
            norm = norms[code]
            scores[code] = scores.get(code, 0.0) + idf * frequency * (index.k1 + 1) / (frequency + norm)
    ranked = heapq.nsmallest(limit, scores.items(), key=lambda item: (-item[1], item[0]))
    result = [code for code, _score in ranked]
    result.extend(code for code in sorted(index.documents) if code not in scores and code not in result)
    return result[:limit]


def load_rows() -> list[TrainingExample]:
    rows: list[TrainingExample] = []
    with BENCHMARK.open(encoding="utf-8") as f:
        for line in f:
            x = BenchmarkExample.model_validate_json(line)
            if x.direction == "ICD9CM_TO_ICD10CM" and x.split == "train" and x.mapping_kind in {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"} and x.valid_target_codes:
                rows.append(TrainingExample(
                    x.benchmark_id, x.source_code, x.source_label or "", x.source_family,
                    x.direction, x.split or "", x.source_family_split or "", x.mapping_kind,
                    tuple(sorted(x.valid_target_codes)), str(x.lexical_metadata.get("lexical_difficulty", "UNKNOWN")),
                ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    rows = load_rows()
    terminology_path = ROOT / "artifacts/terminology_universe_v2/icd10cm_diagnosis.jsonl"
    terminology_bytes = terminology_path.read_bytes()
    if hashlib.sha256(terminology_bytes).hexdigest() != UNIVERSE_HASH:
        raise RuntimeError("canonical terminology file hash mismatch")
    target_records = [json.loads(line) for line in terminology_bytes.decode("utf-8").splitlines()]
    target_codes = [str(record["canonical_code"]) for record in target_records]
    target_texts = {str(record["canonical_code"]): clean_dense_text(record.get("long_description") or record.get("short_description") or "") for record in target_records}
    if len(target_codes) != TARGET_COUNT or target_codes != sorted(target_codes):
        raise RuntimeError("canonical target universe count/order mismatch")
    corpus = type("Corpus", (), {"codes": tuple(target_codes), "corpus_hash": UNIVERSE_HASH, "as_dict": lambda self: target_texts})()

    meta = json.loads(TARGET_META.read_text(encoding="utf-8"))
    if meta.get("codes") != target_codes or sha256(TARGET_CACHE) != "c467b7e1f2672696aa2cd0b7daaa6ffe2aa9c33b627cef086d7ba78d4d7938d4":
        raise RuntimeError("selected cache does not match corrected target order/hash")
    target_matrix = np.asarray(np.load(TARGET_CACHE, mmap_mode="r"), dtype=np.float32)
    if target_matrix.shape != (TARGET_COUNT, 768):
        raise RuntimeError(f"unexpected target cache shape: {target_matrix.shape}")
    bm25 = BM25Index.from_documents(corpus.as_dict(), 2.0, 0.75)
    worker_count = int(__import__("os").environ.get("SHIFT_MAP_BM25_WORKERS", "8"))
    with mp.Pool(processes=worker_count, initializer=_init_bm25, initargs=(bm25,)) as pool:
        bm25_ranked = pool.map(_rank_bm25, [row.source_label for row in rows], chunksize=32)
    encoder = load_encoder(MODEL_SPEC, args.device if torch.cuda.is_available() else "cpu")
    query_result = encoder.encode([clean_dense_text(row.source_label) for row in rows], 64, 64, is_query=True)
    query_embeddings = query_result.embeddings.astype(np.float32, copy=False)
    dense_scores = (torch.from_numpy(query_embeddings).to(args.device) @ torch.from_numpy(target_matrix).to(args.device).T).cpu().numpy()
    records: list[dict[str, Any]] = []
    all_negative_sets: dict[str, list[str]] = {}
    unavailable_same_family = 0
    for i, row in enumerate(rows):
        gold = set(row.valid_target_codes)
        scores = dense_scores[i]
        top_indices = np.argpartition(-scores, 99)[:100]
        lexical_ranked = bm25_ranked[i]
        dense_ranked = [target_codes[j] for j in sorted(top_indices.tolist(), key=lambda j: (-float(scores[j]), target_codes[j]))]
        families = {code.replace(".", "")[:3].upper() for code in gold}
        same_family = [code for code in dense_ranked if code not in gold and code.replace(".", "")[:3].upper() in families][:1]
        unavailable = not same_family
        if unavailable:
            unavailable_same_family += 1
        random_neg = mine_random_negatives(row, target_codes, 4, SEED)
        lexical = mine_ranked_negatives(row, lexical_ranked, 1)
        dense = mine_ranked_negatives(row, dense_ranked, 1)
        mixed: list[str] = []
        for code in lexical + dense + same_family:
            if code not in mixed:
                mixed.append(code)
        for code in random_neg:
            if len(mixed) >= 4:
                break
            if code not in mixed:
                mixed.append(code)
        all_negative_sets[row.benchmark_id] = mixed
        records.append({
            "benchmark_id": row.benchmark_id, "source_code": row.source_code,
            "source_family": row.source_family, "valid_target_codes": list(row.valid_target_codes),
            "mapping_kind": row.mapping_kind, "lexical_difficulty": row.lexical_difficulty,
            "universe_count": TARGET_COUNT, "universe_hash": UNIVERSE_HASH,
            "random": random_neg, "lexical_hard": lexical, "dense_hard": dense,
            "same_family_hard": same_family, "mixed": mixed,
            "negative_provenance": {"lexical": "BM25(k1=2.0,b=0.75)", "dense": MODEL_SPEC.revision, "same_family": "ICD10_prefix3"},
            "same_family_unavailable": unavailable,
        })
    audit = audit_negative_collisions({row.benchmark_id: row for row in rows}, all_negative_sets)
    duplicate_rows = sum(len(set(x["mixed"])) != len(x["mixed"]) for x in records)
    if audit["remaining_gold_collisions"] or duplicate_rows:
        raise RuntimeError({"audit": audit, "duplicate_rows": duplicate_rows})
    OUT.mkdir(parents=True, exist_ok=True)
    negatives_path = OUT / "negative_sets.jsonl"
    with negatives_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "shift_map_full_universe_negative_mining_v1",
        "experiment": "shift_map_full_universe", "benchmark_version": "1.0",
        "direction": "ICD9CM_TO_ICD10CM", "partition": "train", "source_count": len(rows),
        "positive_relation_count": sum(len(r.valid_target_codes) for r in rows),
        "target_universe_count": TARGET_COUNT, "target_universe_hash": UNIVERSE_HASH,
        "target_order_hash": "8ca0d0cf0213581645fe64d9c1e599552a2fb207eb740f32ca09ffeaaf9f8e26",
        "target_cache_sha256": sha256(TARGET_CACHE), "base_model_revision": MODEL_SPEC.revision,
        "seed": SEED, "strategies": ["N1_RANDOM", "N2_LEXICAL_HARD", "N3_DENSE_HARD", "N4_SAME_FAMILY_HARD", "N5_MIXED"],
        "negative_counts_per_source": {"N1_RANDOM": 4, "N2_LEXICAL_HARD": 1, "N3_DENSE_HARD": 1, "N4_SAME_FAMILY_HARD": 1, "N5_MIXED": 4},
        "negative_sets_sha256": sha256(negatives_path),
        "audit": audit | {"duplicate_mixed_rows": duplicate_rows, "unavailable_same_family_count": unavailable_same_family, "all_targets_exist": True, "legacy_universe_rejected": True},
        "test_data_used": False, "dev_data_used": False, "generation_script": "scripts/mine_shift_map_full_universe.py",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT / "negative_mining_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    encoder.close()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
