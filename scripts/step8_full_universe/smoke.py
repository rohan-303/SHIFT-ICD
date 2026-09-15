# ruff: noqa: E501
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from scripts.step8_full_universe.runner import benchmark_rows, code_descriptions, load_candidate_groups
from shift_icd.reranking.step8 import construct_training_list

ROOT = Path(__file__).resolve().parents[2]
MODEL_REVISION = "71caf65d4927987813984f54c284405a13fcca49"
MAX_LENGTH = 96


def authoritative_pairs(path: Path, limit: int) -> tuple[list[list[dict[str, Any]]], list[tuple[str, str]]]:
    groups = load_candidate_groups(path)[:limit]
    benchmark = benchmark_rows()
    descriptions = code_descriptions()
    pairs: list[tuple[str, str]] = []
    for group in groups:
        source_id = str(group[0]["source_id"])
        source_description = str(benchmark[source_id]["source_label"])
        for row in group:
            target = str(row["target_code"])
            target_description = descriptions[target]
            pairs.append((source_description, target_description))
    return groups, pairs


def score_pairs(model: Any, tokenizer: Any, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
    values: list[float] = []
    for start in range(0, len(pairs), batch_size):
        encoded = tokenizer(
            pairs[start : start + batch_size], padding="longest", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        encoded = {key: value.to(next(model.parameters()).device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)
            if tuple(output.logits.shape) != (len(encoded["input_ids"]), 1):
                raise AssertionError("unexpected logits shape")
            values.extend(output.logits[:, 0].float().tolist())
    return values


def deterministic_probe(model: Any, tokenizer: Any, groups: list[list[dict[str, Any]]], pairs: list[tuple[str, str]]) -> dict[str, Any]:
    first = score_pairs(model, tokenizer, pairs, 32)
    second = score_pairs(model, tokenizer, pairs, 32)
    ranked_a: list[list[str]] = []
    ranked_b: list[list[str]] = []
    pos = 0
    for group in groups:
        chunk = list(zip(group, first[pos : pos + 100], strict=True))
        chunk2 = list(zip(group, second[pos : pos + 100], strict=True))
        pos += 100
        ranked_a.append([r["target_code"] for r, _ in sorted(chunk, key=lambda x: (-x[1], int(x[0]["candidate_rank"])))])
        ranked_b.append([r["target_code"] for r, _ in sorted(chunk2, key=lambda x: (-x[1], int(x[0]["candidate_rank"])))])
    return {
        "sources": len(groups),
        "pairs": len(pairs),
        "score_max_abs_diff": max(abs(a - b) for a, b in zip(first, second, strict=True)),
        "ranking_disagreement_count": sum(a != b for a, b in zip(ranked_a, ranked_b, strict=True)),
        "top1_disagreement_count": sum(a[0] != b[0] for a, b in zip(ranked_a, ranked_b, strict=True)),
    }


def throughput_probe(model: Any, tokenizer: Any, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for batch_size in (1, 2, 4, 8, 16, 32, 64):
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            score_pairs(model, tokenizer, pairs, batch_size)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            results.append(
                {
                    "batch_size": batch_size,
                    "pairs": len(pairs),
                    "pairs_per_second": len(pairs) / elapsed,
                    "mean_batch_latency_seconds": elapsed / ((len(pairs) + batch_size - 1) // batch_size),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                    "status": "PASS",
                }
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                results.append({"batch_size": batch_size, "status": "OOM"})
                break
            raise
    return results


def training_probe(model: Any, tokenizer: Any, train_path: Path, output: Path) -> dict[str, Any]:
    groups = load_candidate_groups(train_path)[:128]
    lists = [construct_training_list(group, seed=17) for group in groups]
    benchmark = benchmark_rows()
    descriptions = code_descriptions()
    device = next(model.parameters()).device

    def batch_logits(batch: list[list[dict[str, Any]]]) -> torch.Tensor:
        pairs = [
            (str(benchmark[str(group[0]["source_id"])]["source_label"]), descriptions[str(r["target_code"])])
            for group in batch
            for r in group
        ]
        encoded = tokenizer(pairs, padding="longest", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        return model(**encoded).logits[:, 0].float().reshape(len(batch), 8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    microbatch = 4
    bce_values: list[float] = []
    for start in range(0, len(lists), microbatch):
        batch = lists[start : start + microbatch]
        logits = batch_logits(batch)
        labels = torch.tensor([[1.0 if r["candidate_is_gold"] else 0.0 for r in group] for group in batch], device=device)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        (bce / math.ceil(len(lists) / microbatch)).backward()
        bce_values.append(float(bce.detach().cpu()))
    finite_bce = all(math.isfinite(value) for value in bce_values) and all(
        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
    )
    optimizer.step()
    changed = any(p.grad is not None and bool(torch.any(p.grad != 0)) for p in model.parameters())
    optimizer.zero_grad(set_to_none=True)
    listwise_values: list[float] = []
    for start in range(0, len(lists), microbatch):
        batch = lists[start : start + microbatch]
        logits2 = batch_logits(batch)
        losses = [
            torch.logsumexp(row, dim=0)
            - torch.logsumexp(row[torch.tensor([i for i, r in enumerate(group) if r["candidate_is_gold"]], device=device)], dim=0)
            for row, group in zip(logits2, batch, strict=True)
        ]
        listwise = torch.stack(losses).mean()
        (listwise / math.ceil(len(lists) / microbatch)).backward()
        listwise_values.append(float(listwise.detach().cpu()))
    finite_listwise = all(math.isfinite(value) for value in listwise_values) and all(
        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
    )
    checkpoint = output / "smoke_checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 1, "steps": 1}, checkpoint)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reload_ok = payload["epoch"] == 1 and payload["steps"] == 1 and bool(payload["optimizer"])
    return {
        "status": "SMOKE_ONLY_NOT_SCIENTIFIC_RESULT",
        "sources": len(lists),
        "list_size": 8,
        "bce_finite": finite_bce,
        "listwise_finite": finite_listwise,
        "parameter_update": changed,
        "checkpoint": str(checkpoint),
        "checkpoint_reload": reload_ok,
        "resume_counters": {"epoch": payload["epoch"], "steps": payload["steps"]},
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(17)
    torch.manual_seed(17)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=MODEL_REVISION, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            args.model_path, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False
        )
        .float()
        .to("cuda:0")
        .eval()
    )
    dev_groups, dev_pairs = authoritative_pairs(ROOT / "artifacts/candidates/shift_map_full_universe_v2/forward_dev_k100.jsonl.gz", 100)
    output = {
        "model_revision": MODEL_REVISION,
        "precision": "FP32",
        "max_length": MAX_LENGTH,
        "deterministic": deterministic_probe(model, tokenizer, dev_groups, dev_pairs),
        "throughput": throughput_probe(model, tokenizer, dev_pairs),
    }
    train_model = (
        AutoModelForSequenceClassification.from_pretrained(
            args.model_path, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False
        )
        .float()
        .to("cuda:0")
    )
    output["training"] = training_probe(
        train_model, tokenizer, ROOT / "artifacts/candidates/shift_map_full_universe_v2/forward_train_k100.jsonl.gz", args.output
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "r1_smoke.json").write_text(json.dumps(output, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
