# ruff: noqa
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
from sentence_transformers import SentenceTransformer

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.dense.corpus import FORWARD, build_target_corpus
from shift_map_v1_4_analysis import grouped, summarize_values

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"
CANON = ROOT / "data/processed/cms/2018_gem/normalized_rows.parquet"
EXP = ROOT / "artifacts/experiments/shift_map_v1_4"


def encode(model: SentenceTransformer, texts: list[str], device: str) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=16, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True, device=device, max_length=64).astype(np.float32)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame = pd.read_parquet(CANON)
    corpus = build_target_corpus(frame, FORWARD)
    examples = [BenchmarkExample.model_validate_json(line) for line in BENCH.open(encoding="utf8") if line.strip()]
    specs = {
        "zero_shot": {"model": "FremyCompany/BioLORD-2023", "revision": "167aab527b238a50ca65224e6319215d2ff4fc9f"},
        "l1_seed42": {"model": str(ROOT / "artifacts/models/shift_map_v1/final_seed42/epoch_3")},
        "l2_seed17": {"model": str(ROOT / "artifacts/models/shift_map_v1/v1_3_final_l2_n1_seed17/epoch_3")},
    }
    embeddings: dict[str, dict[str, np.ndarray]] = {}
    timings: list[dict[str, Any]] = []
    target_texts = [corpus.codes[i] and corpus.as_dict()[corpus.codes[i]] for i in range(len(corpus.codes))]
    for name, spec in specs.items():
        kwargs: dict[str, Any] = {"device": device, "trust_remote_code": True}
        if spec.get("revision"): kwargs["revision"] = spec["revision"]
        t0 = time.perf_counter(); model = SentenceTransformer(spec["model"], **kwargs); load_time = time.perf_counter() - t0
        load_rss = psutil.Process().memory_info().rss / 2**20
        if device == "cuda": torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter(); target = encode(model, target_texts, device); target_time = time.perf_counter() - t1
        target_alloc = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
        target_reserved = torch.cuda.max_memory_reserved() / 2**20 if device == "cuda" else None
        target_rss = psutil.Process().memory_info().rss / 2**20
        sources: dict[str, np.ndarray] = {}
        for split in ("train", "dev", "test"):
            selected = [x for x in examples if x.direction == FORWARD and x.split == split]
            t2 = time.perf_counter(); sources[split] = encode(model, [x.source_label or "" for x in selected], device); source_time = time.perf_counter() - t2
            source_alloc = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
            source_reserved = torch.cuda.max_memory_reserved() / 2**20 if device == "cuda" else None
            timings.append({"model": name, "split": split, "load_seconds": load_time, "target_encoding_seconds": target_time, "source_encoding_seconds": source_time, "n_sources": len(selected), "device": device, "load_cpu_rss_mb": load_rss, "target_cpu_rss_mb": target_rss, "target_peak_gpu_allocated_mb": target_alloc, "target_peak_gpu_reserved_mb": target_reserved, "source_peak_gpu_allocated_mb": source_alloc, "source_peak_gpu_reserved_mb": source_reserved})
        embeddings[name] = {"target": target, **sources}
        del model
        if device == "cuda": torch.cuda.empty_cache()
    cosine_target = np.sum(embeddings["zero_shot"]["target"] * embeddings["l2_seed17"]["target"], axis=1)
    target_stats = summarize_values(cosine_target.tolist())
    target_stats.update({"target_count": len(cosine_target), "direction": FORWARD, "corpus_hash": corpus.corpus_hash})
    train_positive: dict[str, int] = {}
    for x in examples:
        if x.direction == FORWARD and x.split == "train":
            for code in x.valid_target_codes: train_positive[code] = train_positive.get(code, 0) + 1
    exposure_groups = {"TRAIN_POSITIVE_TARGET": [], "NON_TRAIN_POSITIVE_TARGET": []}
    exposure_freq = {"1": [], "2-5": [], "6+": []}
    for code, value in zip(corpus.codes, cosine_target, strict=True):
        group = "TRAIN_POSITIVE_TARGET" if code in train_positive else "NON_TRAIN_POSITIVE_TARGET"
        exposure_groups[group].append(float(value))
        if group == "TRAIN_POSITIVE_TARGET": exposure_freq["1" if train_positive[code] == 1 else "2-5" if train_positive[code] <= 5 else "6+"].append(float(value))
    dump = {"method": {"tokenizer": "model-native", "max_length": 64, "pooling": "SentenceTransformer configured pooling", "normalization": "L2", "ordering": "sorted direction-scoped corpus"}, "overall": target_stats, "by_training_exposure": {key: summarize_values(value) for key, value in exposure_groups.items()}, "train_positive_frequency": {key: summarize_values(value) for key, value in exposure_freq.items()}}
    (EXP / "representation_drift.json").write_text(json.dumps(dump, indent=2) + "\n", encoding="utf8")
    source_stats = []
    for split in ("train", "dev", "test"):
        values = np.sum(embeddings["zero_shot"][split] * embeddings["l2_seed17"][split], axis=1)
        source_stats.append({"split": split, **summarize_values(values.tolist())})
    (EXP / "source_drift.json").write_text(json.dumps({"direction": FORWARD, "rows": source_stats, "target": target_stats}, indent=2) + "\n", encoding="utf8")
    (EXP / "runtime.json").write_text(json.dumps({"training_peak_memory": "NOT_AVAILABLE", "inference_measurements": timings, "measurement_note": "encoding wall-clock measurements; retrieval is represented by precomputed ledger generation"}, indent=2) + "\n", encoding="utf8")
    print(f"drift_targets={len(cosine_target)} source_splits={len(source_stats)} device={device}")

if __name__ == "__main__": main()
