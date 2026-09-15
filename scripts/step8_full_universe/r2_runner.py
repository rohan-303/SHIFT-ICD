# ruff: noqa: E501
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.step8_full_universe.runner import benchmark_rows, load_candidate_groups
from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.reranking.step8 import source_balanced_bce_loss, source_balanced_listwise_loss

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_ROOT = ROOT / "artifacts/candidates/shift_map_full_universe_v2"
MODEL_REVISION = "71caf65d4927987813984f54c284405a13fcca49"
MAX_LENGTH = 96
LIST_SIZE = 8
SEED = 17
EPOCHS = 3
BATCH_SIZE = 32
MICRO_SOURCE_BATCH = 4
STRATEGIES = ("TOP_RANK_HARD", "MIXED_RANK", "RANDOM_WITHIN_CANDIDATE")
OBJECTIVES = ("BCE", "SET_POSITIVE_LISTWISE")
LEARNING_RATES = (1e-5, 2e-5, 3e-5)
SELECTION_FIELDS = ("Hit@1", "MRR", "Hit@10", "NDCG@10", "P_COMPLEX_CompleteScenarioRetrieval@10")


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(float(row.get(field) if row.get(field) is not None else -1.0) for field in SELECTION_FIELDS)


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("valid")]
    if not valid:
        raise RuntimeError("no valid R2 runs available for selection")
    return max(valid, key=lambda r: (selection_key(r), -int(r["epoch"])))


def prepare_groups(path: Path, benchmark: dict[str, dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups = load_candidate_groups(path)
    result = []
    for group in groups:
        source = benchmark[str(group[0]["source_id"])]
        gold = set(str(x) for x in source.get("valid_target_codes", []))
        for row in group:
            row["_gold"] = str(row["target_code"]) in gold
        result.append(group)
    return result


def training_groups(groups: list[list[dict[str, Any]]], benchmark: dict[str, dict[str, Any]]) -> list[list[dict[str, Any]]]:
    eligible = []
    for group in groups:
        meta = benchmark[str(group[0]["source_id"])]
        if meta.get("mapping_kind") in {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"} and any(r["_gold"] for r in group):
            eligible.append(group)
    return eligible


def make_list(group: list[dict[str, Any]], strategy: str, seed: int, epoch: int) -> list[dict[str, Any]]:
    positives = [r for r in group if r["_gold"]]
    negatives = [r for r in group if not r["_gold"]]
    rng = random.Random(f"{seed}:{epoch}:{group[0]['source_id']}:{strategy}")
    if strategy == "TOP_RANK_HARD":
        negatives.sort(key=lambda r: int(r["candidate_rank"]))
    elif strategy == "MIXED_RANK":
        buckets = [
            [r for r in negatives if 1 <= int(r["candidate_rank"]) <= 10],
            [r for r in negatives if 11 <= int(r["candidate_rank"]) <= 25],
            [r for r in negatives if 26 <= int(r["candidate_rank"]) <= 100],
        ]
        for bucket in buckets:
            rng.shuffle(bucket)
        negatives = [r for bucket in buckets for r in bucket]
    elif strategy == "RANDOM_WITHIN_CANDIDATE":
        rng.shuffle(negatives)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    if not negatives:
        raise ValueError("SOURCE_HAS_NO_VALID_NEGATIVE_IN_FROZEN_CANDIDATE_SET")
    negative_count = max(1, LIST_SIZE - len(positives))
    return sorted(positives + negatives[:negative_count], key=lambda r: int(r["candidate_rank"]))


def encode(tokenizer: Any, pairs: list[tuple[str, str]], device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(pairs, padding="longest", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}


def score_groups(model: Any, tokenizer: Any, groups: list[list[dict[str, Any]]], benchmark: dict[str, dict[str, Any]], device: torch.device) -> list[list[str]]:
    model.eval()
    ranked: list[list[str]] = []
    pairs: list[tuple[str, str]] = []
    for group in groups:
        source = str(benchmark[str(group[0]["source_id"])]["source_label"])
        pairs.extend((source, str(row["target_description"])) for row in group)
    scores: list[float] = []
    for start in range(0, len(pairs), BATCH_SIZE):
        with torch.inference_mode():
            out = model(**encode(tokenizer, pairs[start:start + BATCH_SIZE], device))
            if tuple(out.logits.shape) != (min(BATCH_SIZE, len(pairs) - start), 1):
                raise AssertionError("unexpected MedCPT logits shape")
            scores.extend(out.logits[:, 0].float().cpu().tolist())
    offset = 0
    for group in groups:
        chunk = list(zip(group, scores[offset:offset + len(group)], strict=True))
        offset += len(group)
        ranked.append([str(row["target_code"]) for row, _ in sorted(chunk, key=lambda x: (-float(x[1]), int(x[0]["candidate_rank"])))])
    return ranked


def ndcg(gold: set[str], ranked: list[str], k: int) -> float:
    gains = [1.0 if code in gold else 0.0 for code in ranked[:k]]
    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
    ideal = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg else 0.0


def structural(example: BenchmarkExample, ranked: list[str], k: int) -> tuple[float, float]:
    retrieved = set(ranked[:k])
    if not example.scenarios:
        covered = bool(retrieved & set(example.valid_target_codes))
        return float(covered), float(covered)
    choice_total = sum(len(s.choice_lists) for s in example.scenarios)
    choice_hit = sum(sum(any(a.target_code in retrieved for a in c.alternatives) for c in s.choice_lists) for s in example.scenarios)
    complete = any(all(any(a.target_code in retrieved for a in c.alternatives) for c in s.choice_lists) for s in example.scenarios)
    return (choice_hit / choice_total if choice_total else 0.0), float(complete)


def metrics(groups: list[list[dict[str, Any]]], ranked: list[list[str]], benchmark: dict[str, dict[str, Any]]) -> dict[str, Any]:
    examples = [BenchmarkExample.model_validate(benchmark[str(g[0]["source_id"])]) for g in groups]
    ordinary = [(e, r) for e, r in zip(examples, ranked, strict=True) if e.mapping_kind in {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"} and e.valid_target_codes]
    result: dict[str, Any] = {}
    for k in (1, 5, 10, 25, 50, 100):
        result[f"Hit@{k}"] = sum(bool(set(r[:k]) & set(e.valid_target_codes)) for e, r in ordinary) / len(ordinary)
    result["MRR"] = sum((1.0 / (next((i + 1 for i, c in enumerate(r) if c in set(e.valid_target_codes)), len(r) + 1))) for e, r in ordinary) / len(ordinary)
    result["NDCG@10"] = sum(ndcg(set(e.valid_target_codes), r, 10) for e, r in ordinary) / len(ordinary)
    populations = {
        "P_COMBINATION": lambda e: e.mapping_kind == "COMBINATION",
        "P_COMBINATION_WITH_ALTERNATIVES": lambda e: e.mapping_kind == "COMBINATION_WITH_ALTERNATIVES",
        "P_COMPLEX": lambda e: "HIGH_MAPPING_COMPLEXITY" in e.difficulty_slices,
    }
    for name, predicate in populations.items():
        selected = [(e, r) for e, r in zip(examples, ranked, strict=True) if predicate(e)]
        for k in (1, 5, 10, 25, 50, 100):
            choices = [structural(e, r, k)[0] for e, r in selected]
            completes = [structural(e, r, k)[1] for e, r in selected]
            result[f"{name}_ChoiceListRecall@{k}"] = sum(choices) / len(choices) if choices else None
            result[f"{name}_CompleteScenarioRetrieval@{k}"] = sum(completes) / len(completes) if completes else None
    result["candidate_mutation_count"] = 0
    result["test_scoring_count"] = 0
    result["test_training_count"] = 0
    return result


def train_epoch(model: Any, tokenizer: Any, lists: list[list[dict[str, Any]]], benchmark: dict[str, dict[str, Any]], objective: str, optimizer: Any, device: torch.device) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    steps = math.ceil(len(lists) / MICRO_SOURCE_BATCH)
    for start in range(0, len(lists), MICRO_SOURCE_BATCH):
        batch = lists[start:start + MICRO_SOURCE_BATCH]
        pairs = [(str(benchmark[str(g[0]["source_id"])]["source_label"]), str(r["target_description"])) for g in batch for r in g]
        logits = model(**encode(tokenizer, pairs, device)).logits[:, 0].float()
        offsets = []
        cursor = 0
        for group in batch:
            offsets.append((cursor, cursor + len(group)))
            cursor += len(group)
        if objective == "BCE":
            terms = []
            for group, (left, right) in zip(batch, offsets, strict=True):
                labels = torch.tensor([1.0 if r["_gold"] else 0.0 for r in group], device=device)
                terms.append((logits[left:right], labels))
            loss = source_balanced_bce_loss(terms)
        elif objective == "SET_POSITIVE_LISTWISE":
            source_logits = [logits[left:right] for left, right in offsets]
            positives = [[i for i, r in enumerate(group) if r["_gold"]] for group in batch]
            loss = source_balanced_listwise_loss(source_logits, positives)
        else:
            raise ValueError(objective)
        (loss / steps).backward()
        losses.append(float(loss.detach().cpu()))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError("non-finite gradient")
    optimizer.step()
    return sum(losses) / len(losses)


def run_config(objective: str, strategy: str, lr: float, output: Path, train: list[list[dict[str, Any]]], dev: list[list[dict[str, Any]]], benchmark: dict[str, dict[str, Any]], model_path: Path) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=MODEL_REVISION, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False).float().to("cuda:0")
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rows: list[dict[str, Any]] = []
    best_state: bytes | None = None
    best_row: dict[str, Any] | None = None
    run_id = f"{objective.lower()}_{strategy.lower()}_{lr:g}".replace(".", "p")
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        lists = [make_list(g, strategy, SEED, epoch) for g in train]
        loss = train_epoch(model, tokenizer, lists, benchmark, objective, optimizer, device)
        ranked = score_groups(model, tokenizer, dev, benchmark, device)
        metric = metrics(dev, ranked, benchmark)
        row = {"run_id": run_id, "stage": "R2", "objective": objective, "list_strategy": strategy, "learning_rate": lr, "seed": SEED, "epoch": epoch, "model_revision": MODEL_REVISION, "candidate_train_hash": "d90dcb937748f1c0b173416d6b8f23f1c97ca8e8caf8934efdf8120993c1cb10", "candidate_dev_hash": "c7240c1f38dd87f54d63c97999cba71d91f6d0887d89bc2d4094f9ba38ad6a6f", "training_source_count": len(train), "training_list_count": len(lists), "positive_candidate_count": sum(sum(r["_gold"] for r in g) for g in lists), "negative_candidate_count": sum(sum(not r["_gold"] for r in g) for g in lists), "training_loss": loss, "runtime": time.perf_counter() - started, "valid": True, **metric}
        rows.append(row)
        if best_row is None or (selection_key(row), -epoch) > (selection_key(best_row), -int(best_row["epoch"])):
            torch.save(model.state_dict(), "_r2_state.pt")
            best_row = row
            best_state = Path("_r2_state.pt").read_bytes()
            Path("_r2_state.pt").unlink()
    if best_state is None or best_row is None:
        raise RuntimeError("no checkpoint")
    ckpt = output / "selected_checkpoints" / f"{run_id}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(best_state)
    for row in rows:
        row["selected_epoch"] = int(best_row["epoch"])
        row["selected_configuration"] = row["epoch"] == best_row["epoch"]
        row["checkpoint_sha256"] = sha256(ckpt) if row["selected_configuration"] else None
    del model
    torch.cuda.empty_cache()
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    benchmark = benchmark_rows()
    train = training_groups(prepare_groups(CANDIDATE_ROOT / "forward_train_k100.jsonl.gz", benchmark), benchmark)
    dev = prepare_groups(CANDIDATE_ROOT / "forward_dev_k100.jsonl.gz", benchmark)
    all_rows: list[dict[str, Any]] = []
    for objective in OBJECTIVES:
        all_rows.extend(run_config(objective, "MIXED_RANK", 1e-5, args.output, train, dev, benchmark, args.model_path))
    objective_winners = [select_best([r for r in all_rows if r["objective"] == o]) for o in OBJECTIVES]
    selected_objective = select_best(objective_winners)["objective"]
    for strategy in STRATEGIES:
        all_rows.extend(run_config(selected_objective, strategy, 1e-5, args.output, train, dev, benchmark, args.model_path))
    strategy_winners = [select_best([r for r in all_rows if r["objective"] == selected_objective and r["list_strategy"] == s]) for s in STRATEGIES]
    selected_strategy = select_best(strategy_winners)["list_strategy"]
    for lr in LEARNING_RATES:
        all_rows.extend(run_config(selected_objective, selected_strategy, lr, args.output, train, dev, benchmark, args.model_path))
    lr_winners = [select_best([r for r in all_rows if r["objective"] == selected_objective and r["list_strategy"] == selected_strategy and float(r["learning_rate"]) == lr]) for lr in LEARNING_RATES]
    selected_lr = select_best(lr_winners)["learning_rate"]
    write_csv(args.output / "ablation_master.csv", all_rows)
    selections = {"selected_objective": selected_objective, "objective_runner_up": next(r["objective"] for r in objective_winners if r["objective"] != selected_objective), "selected_strategy": selected_strategy, "strategy_runner_up": next(r["list_strategy"] for r in strategy_winners if r["list_strategy"] != selected_strategy), "selected_learning_rate": selected_lr, "learning_rate_runner_up": next(r["learning_rate"] for r in lr_winners if r["learning_rate"] != selected_lr), "selection_rule": list(SELECTION_FIELDS), "epoch_rule": "E2_BEST_DEV_CHECKPOINT_WITHIN_MAX_EPOCH_BUDGET", "test_scoring_count": 0, "test_training_count": 0}
    (args.output / "selection.json").write_text(json.dumps(selections, indent=2) + "\n")
    print(json.dumps({"status": "R2_DEV_ABLATIONS_COMPLETE", "train_sources": len(train), "dev_sources": len(dev), "rows": len(all_rows), **selections}, indent=2))


if __name__ == "__main__":
    main()
