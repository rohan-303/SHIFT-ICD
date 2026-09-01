# ruff: noqa
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shift_icd.reranking.shift_map_v2 import ndcg_at_k, preserve_candidate_membership, sample_negative_ranks

ROOT = Path(__file__).resolve().parents[1]
CAND = ROOT / "artifacts/candidates/shift_map_v2"
EXP = ROOT / "artifacts/experiments/shift_map_v2"
MODEL_ID = "ncbi/MedCPT-Cross-Encoder"
MODEL_REVISION = "71caf65d4927987813984f54c284405a13fcca49"
ORDINARY = {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"}


def load_groups(split: str) -> list[dict[str, Any]]:
    path = CAND / f"forward_stratified_{split}_k100.jsonl.gz"
    groups: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf8") as handle:
        for line in handle:
            row = json.loads(line)
            group = groups.setdefault(row["benchmark_id"], {"benchmark_id": row["benchmark_id"], "split": split, "rows": []})
            group["rows"].append(row)
    result = list(groups.values())
    for group in result:
        group["rows"].sort(key=lambda row: row["retriever_rank"])
    return result


def load_model(snapshot: str, device: str) -> tuple[Any, Any, str, int]:
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(snapshot, local_files_only=True, revision=MODEL_REVISION)
    model.eval()
    model.to(device)
    params = sum(parameter.numel() for parameter in model.parameters())
    return tokenizer, model, str(next(model.parameters()).dtype), params


def score_pairs(groups: list[dict[str, Any]], tokenizer: Any, model: Any, device: str, batch_size: int, max_length: int) -> list[np.ndarray]:
    pairs: list[tuple[str, str]] = []
    spans: list[tuple[int, int]] = []
    for group in groups:
        start = len(pairs)
        pairs.extend((row["source_description"], row["target_description"]) for row in group["rows"])
        spans.append((start, len(pairs)))
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            scores.extend(logits[:, 0].float().cpu().tolist())
            if (start // batch_size + 1) % 100 == 0:
                print(f"scored_pairs={min(start + batch_size, len(pairs))}/{len(pairs)}", flush=True)
    return [np.asarray(scores[start:end], dtype=np.float64) for start, end in spans]


def ranking_metrics(groups: list[dict[str, Any]], score_arrays: list[np.ndarray], ks: tuple[int, ...] = (1, 3, 5, 10, 25, 50, 100)) -> dict[str, float | int]:
    hits = {k: [] for k in ks}
    mrr: list[float] = []
    ndcgs = {5: [], 10: []}
    ordinary = [g for g in groups if g["rows"][0]["mapping_kind"] in ORDINARY]
    ordinary_scores = [scores for group, scores in zip(groups, score_arrays, strict=True) if group["rows"][0]["mapping_kind"] in ORDINARY]
    for group, scores in zip(ordinary, ordinary_scores, strict=True):
        rows = group["rows"]
        gold = {row["target_code"] for row in rows if row["candidate_is_gold"]}
        order = np.argsort(-scores, kind="stable")
        ranked = [rows[int(index)]["target_code"] for index in order]
        first = next((index + 1 for index, code in enumerate(ranked) if code in gold), None)
        mrr.append(1.0 / first if first is not None else 0.0)
        for k in ks:
            hits[k].append(float(bool(set(ranked[:k]) & gold)))
        ndcgs[5].append(ndcg_at_k(gold, ranked, 5))
        ndcgs[10].append(ndcg_at_k(gold, ranked, 10))
    output: dict[str, float | int] = {"n_all_sources": len(groups), "n_ordinary_sources": len(ordinary)}
    for k in ks:
        output[f"Hit@{k}"] = float(np.mean(hits[k])) if hits[k] else 0.0
    output["MRR"] = float(np.mean(mrr)) if mrr else 0.0
    output["NDCG@5"] = float(np.mean(ndcgs[5])) if ndcgs[5] else 0.0
    output["NDCG@10"] = float(np.mean(ndcgs[10])) if ndcgs[10] else 0.0
    return output


def run_zero_shot(split: str, snapshot: str, device: str, batch_size: int, max_length: int) -> None:
    groups = load_groups(split)
    tokenizer, model, dtype, params = load_model(snapshot, device)
    start = time.perf_counter()
    scores = score_pairs(groups, tokenizer, model, device, batch_size, max_length)
    elapsed = time.perf_counter() - start
    out_groups = []
    for group, values in zip(groups, scores, strict=True):
        rows = group["rows"]
        before = [[row["target_code"] for row in rows]]
        order = np.argsort(-values, kind="stable")
        after_rows = [rows[int(index)] for index in order]
        after = [[row["target_code"] for row in after_rows]]
        preserve_candidate_membership(before, after)
        out_groups.append({"benchmark_id": group["benchmark_id"], "split": split, "mapping_kind": rows[0]["mapping_kind"], "ranked_codes": [row["target_code"] for row in after_rows], "scores": [float(values[int(index)]) for index in order], "gold_codes": sorted({row["target_code"] for row in rows if row["candidate_is_gold"]})})
    metrics = ranking_metrics(groups, scores)
    metrics.update({"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "dtype": dtype, "parameter_count": params, "batch_size": batch_size, "max_length": max_length, "elapsed_seconds": elapsed, "candidate_membership_preserved": True, "candidate_hit100_invariant": metrics["Hit@100"]})
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / f"zero_shot_cross_encoder_{split}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf8")
    with gzip.open(EXP / f"zero_shot_cross_encoder_{split}_rankings.jsonl.gz", "wt", encoding="utf8") as handle:
        for row in out_groups:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(metrics, indent=2))


def audit_lengths(snapshot: str, batch_size: int, max_audit_length: int = 512) -> None:
    tokenizer, _, _, _ = load_model(snapshot, "cpu")
    lengths_all: list[int] = []
    for split in ("train", "dev"):
        with gzip.open(CAND / f"forward_stratified_{split}_k100.jsonl.gz", "rt", encoding="utf8") as handle:
            batch: list[tuple[str, str]] = []
            for line in handle:
                row = json.loads(line)
                batch.append((row["source_description"], row["target_description"]))
                if len(batch) >= batch_size:
                    lengths_all.extend(int(x) for x in tokenizer(batch, truncation=False, padding=False, return_length=True)["length"])
                    batch.clear()
            if batch:
                lengths_all.extend(int(x) for x in tokenizer(batch, truncation=False, padding=False, return_length=True)["length"])
    lengths = np.asarray(lengths_all, dtype=np.int64)
    result = {
        "n": int(len(lengths)),
        "max_length": int(lengths.max()),
        "p95": float(np.quantile(lengths, 0.95)),
        "p99": float(np.quantile(lengths, 0.99)),
        "gt64": int(np.sum(lengths > 64)),
        "gt96": int(np.sum(lengths > 96)),
        "gt128": int(np.sum(lengths > 128)),
        "pct_gt64": float(np.mean(lengths > 64)),
        "pct_gt96": float(np.mean(lengths > 96)),
        "pct_gt128": float(np.mean(lengths > 128)),
    }
    (EXP / "sequence_length_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("zero-shot", "length-audit"), required=True)
    parser.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--offset-groups", type=int, default=0)
    parser.add_argument("--limit-groups", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "length-audit":
        audit_lengths(args.snapshot, args.batch_size)
        return
    groups = load_groups(args.split)
    groups = groups[args.offset_groups : args.offset_groups + args.limit_groups] if args.limit_groups else groups[args.offset_groups :]
    tokenizer, model, dtype, params = load_model(args.snapshot, args.device)
    start = time.perf_counter()
    scores = score_pairs(groups, tokenizer, model, args.device, args.batch_size, args.max_length)
    elapsed = time.perf_counter() - start
    out_groups = []
    for group, values in zip(groups, scores, strict=True):
        rows = group["rows"]
        order = np.argsort(-values, kind="stable")
        after_rows = [rows[int(index)] for index in order]
        preserve_candidate_membership([[row["target_code"] for row in rows]], [[row["target_code"] for row in after_rows]])
        out_groups.append({"benchmark_id": group["benchmark_id"], "split": args.split, "mapping_kind": rows[0]["mapping_kind"], "ranked_codes": [row["target_code"] for row in after_rows], "scores": [float(values[int(index)]) for index in order], "gold_codes": sorted({row["target_code"] for row in rows if row["candidate_is_gold"]})})
    metrics = ranking_metrics(groups, scores)
    metrics.update({"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "dtype": dtype, "parameter_count": params, "batch_size": args.batch_size, "max_length": args.max_length, "elapsed_seconds": elapsed, "candidate_membership_preserved": True, "candidate_hit100_invariant": metrics["Hit@100"]})
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / f"zero_shot_cross_encoder_{args.split}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf8")
    with gzip.open(EXP / f"zero_shot_cross_encoder_{args.split}_rankings.jsonl.gz", "wt", encoding="utf8") as handle:
        for row in out_groups:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
