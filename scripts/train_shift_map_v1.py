# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.dense.corpus import FORWARD, build_target_corpus
from shift_icd.evaluation.retrieval import choice_list_recall_at_k, complete_scenario_retrieval_at_k
from shift_icd.shift_map.losses import l1_masked_single_infonce, l2_set_positive_infonce
from shift_icd.shift_map.training import TrainingExample, eligible_examples, sample_positive, source_balanced_epoch

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"
NEGATIVE_FILE = ROOT / "artifacts/experiments/shift_map_v1/negative_sets.jsonl"
MODEL_ID = "FremyCompany/BioLORD-2023"
REVISION = "167aab527b238a50ca65224e6319215d2ff4fc9f"
TARGET_CACHE = ROOT / "artifacts/embeddings/dense_v1/biolord_2023_icd9cm_to_icd10cm_targets.json"
TARGET_EMBEDDINGS = TARGET_CACHE.with_suffix(".npy")
MODEL_ROOT = ROOT / "artifacts/models/shift_map_v1"
EXPERIMENT_ROOT = ROOT / "artifacts/experiments/shift_map_v1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_examples() -> tuple[list[TrainingExample], list[BenchmarkExample]]:
    train: list[TrainingExample] = []
    all_examples: list[BenchmarkExample] = []
    with BENCHMARK.open(encoding="utf-8") as handle:
        for line in handle:
            example = BenchmarkExample.model_validate_json(line)
            all_examples.append(example)
            if example.direction == "ICD9CM_TO_ICD10CM" and example.split == "train":
                train.append(
                    TrainingExample(
                        example.benchmark_id,
                        example.source_code,
                        example.source_label or "",
                        example.source_family,
                        example.direction,
                        example.split or "",
                        example.source_family_split or "",
                        example.mapping_kind,
                        tuple(sorted(example.valid_target_codes)),
                        str(example.lexical_metadata.get("lexical_difficulty", "UNKNOWN")),
                    )
                )
    return train, all_examples


def load_negatives() -> dict[str, dict[str, list[str]]]:
    with NEGATIVE_FILE.open(encoding="utf-8") as handle:
        return {row["benchmark_id"]: row for row in map(json.loads, handle)}


def move_features(model: SentenceTransformer, texts: list[str], device: str) -> dict[str, torch.Tensor]:
    features = model.tokenize(texts)
    return {key: value.to(device) for key, value in features.items()}


def embed_trainable(model: SentenceTransformer, texts: list[str], device: str) -> torch.Tensor:
    return model(move_features(model, texts, device))["sentence_embedding"]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if not row["no_map"]]
    ordinary = [row for row in answerable if row["mapping_kind"] not in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}]
    complex_rows = [row for row in answerable if row["mapping_kind"] in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}]
    result: dict[str, Any] = {
        "n": len(rows),
        "answerable_n": len(answerable),
        "applicable_metric_n": len(ordinary),
        "combination_n": len(complex_rows),
    }
    for k in (1, 5, 10, 25, 50, 100):
        result[f"Hit@{k}"] = float(np.mean([row[f"Hit@{k}"] for row in ordinary])) if ordinary else None
        result[f"CompleteScenarioRetrieval@{k}"] = (
            float(np.mean([row[f"CompleteScenarioRetrieval@{k}"] for row in complex_rows])) if complex_rows else None
        )
    result["MRR"] = float(np.mean([row["MRR"] for row in ordinary])) if ordinary else None
    low = [row for row in ordinary if row["lexical_difficulty"] == "LEXICAL_LOW"]
    result["LEXICAL_LOW_Hit@100"] = float(np.mean([row["Hit@100"] for row in low])) if low else None
    return result


def evaluate_dev(
    model: SentenceTransformer, dev: list[BenchmarkExample], target_texts: dict[str, str], device: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    target_codes = sorted(target_texts)
    with torch.inference_mode():
        target = model.encode(
            [target_texts[code] for code in target_codes],
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
            max_length=64,
        ).astype(np.float32)
        query = model.encode(
            [example.source_label or "" for example in dev],
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
            max_length=64,
        ).astype(np.float32)
    rows: list[dict[str, Any]] = []
    for example, vector in zip(dev, query, strict=True):
        scores = target @ vector
        candidates = np.argpartition(-scores, min(99, len(scores) - 1))[: min(100, len(scores))]
        ranked = sorted(candidates.tolist(), key=lambda index: (-float(scores[index]), target_codes[index]))
        ranked_codes = [target_codes[index] for index in ranked]
        valid = set(example.valid_target_codes)
        complex_mapping = example.mapping_kind in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}
        row: dict[str, Any] = {
            "benchmark_id": example.benchmark_id,
            "direction": example.direction,
            "split_protocol": "stratified",
            "partition": "dev",
            "sample_population": "forward_stratified_dev",
            "benchmark_version": "1.0",
            "mapping_kind": example.mapping_kind,
            "lexical_difficulty": example.lexical_metadata.get("lexical_difficulty", "UNKNOWN"),
            "no_map": example.no_map,
            "valid_target_codes": sorted(valid),
        }
        for k in (1, 5, 10, 25, 50, 100):
            row[f"Hit@{k}"] = None if example.no_map or complex_mapping else float(bool(valid & set(ranked_codes[:k])))
            row[f"ChoiceListRecall@{k}"] = choice_list_recall_at_k(example, ranked_codes, k)
            row[f"CompleteScenarioRetrieval@{k}"] = complete_scenario_retrieval_at_k(example, ranked_codes, k)
        row["MRR"] = (
            None
            if example.no_map or complex_mapping
            else next((1.0 / rank for rank, code in enumerate(ranked_codes, 1) if code in valid), 0.0)
        )
        rows.append(row)
    return summarize_rows(rows), rows


def train_epoch(
    model: SentenceTransformer,
    optimizer: torch.optim.Optimizer,
    rows: list[TrainingExample],
    negatives: dict[str, dict[str, list[str]]],
    policy: str,
    loss_name: str,
    strategy: str,
    seed: int,
    epoch: int,
    device: str,
    micro_batch: int,
    accumulation: int,
    temperature: float,
) -> float:
    eligible = source_balanced_epoch(eligible_examples(rows, policy), seed, epoch)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    optimizer_steps = 0
    for start in range(0, len(eligible), micro_batch):
        batch = eligible[start : start + micro_batch]
        selected = [sample_positive(row, seed, epoch) for row in batch]
        candidate_codes = list(dict.fromkeys(code for row, positive in zip(batch, selected, strict=True) for code in [positive, *negatives[row.benchmark_id][strategy.lower()]]))
        for row in batch:
            negative_codes = set(negatives[row.benchmark_id][strategy.lower()])
            if set(row.valid_target_codes) & negative_codes:
                raise RuntimeError(f"gold positive collision for {row.benchmark_id}")
        query = embed_trainable(model, [row.source_label for row in batch], device)
        target = embed_trainable(model, [target_texts_global[code] for code in candidate_codes], device)
        candidate_batch = target.unsqueeze(0).expand(len(batch), -1, -1)
        positive_indices = [candidate_codes.index(code) for code in selected]
        valid_indices = [[index for index, code in enumerate(candidate_codes) if code in set(row.valid_target_codes)] for row in batch]
        if loss_name == "L1":
            loss = l1_masked_single_infonce(query, candidate_batch, positive_indices, valid_indices, temperature)
        elif loss_name == "L2":
            loss = l2_set_positive_infonce(query, candidate_batch, valid_indices, temperature)
        else:
            raise ValueError(loss_name)
        (loss / accumulation).backward()
        losses.append(float(loss.detach().cpu()))
        if ((start // micro_batch) + 1) % accumulation == 0 or start + micro_batch >= len(eligible):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
    if optimizer_steps == 0:
        raise RuntimeError("training epoch produced no optimizer step")
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["P0", "P1", "P2"], required=True)
    parser.add_argument("--loss", choices=["L1", "L2"], required=True)
    parser.add_argument("--strategy", choices=["random", "lexical_hard", "dense_hard", "mixed"], required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--limit-sources", type=int, default=None)
    parser.add_argument("--limit-dev", type=int, default=None)
    args = parser.parse_args()
    global target_texts_global
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_rows, all_examples = load_examples()
    if args.limit_sources is not None:
        train_rows = train_rows[: args.limit_sources]
    negatives = load_negatives()
    target_meta = json.loads(TARGET_CACHE.read_text(encoding="utf-8"))
    normalized = __import__("pandas").read_parquet(ROOT / "data/processed/cms/2018_gem/normalized_rows.parquet")
    forward_corpus = build_target_corpus(normalized, FORWARD)
    if target_meta["codes"] != list(forward_corpus.codes):
        raise RuntimeError("frozen BioLORD target cache code order does not match authoritative forward corpus")
    target_texts_global = forward_corpus.as_dict()
    dev = [example for example in all_examples if example.direction == "ICD9CM_TO_ICD10CM" and example.split == "dev"]
    if args.limit_dev is not None:
        dev = dev[: args.limit_dev]
    model = SentenceTransformer(MODEL_ID, revision=REVISION, device=device, trust_remote_code=True)
    model.max_seq_length = 64
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    eligible_count = len(eligible_examples(train_rows, args.policy))
    steps = max(1, eligible_count // 32)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(1, int(steps * args.epochs * 0.1)))
    run_dir = MODEL_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        loss = train_epoch(
            model, optimizer, train_rows, negatives, args.policy, args.loss, args.strategy, args.seed, epoch, device, args.micro_batch, max(1, (32 + args.micro_batch - 1) // args.micro_batch), 0.05
        )
        scheduler.step()
        dev_summary, dev_rows = evaluate_dev(model, dev, target_texts_global, device)
        history.append({"epoch": epoch + 1, "training_loss": loss, **dev_summary})
        model.save(str(run_dir / f"epoch_{epoch + 1}"))
        (run_dir / f"epoch_{epoch + 1}_dev.json").write_text(
            json.dumps({"summary": dev_summary, "rows": dev_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    metadata = {
        "experiment": "shift_map_v1",
        "run_id": args.run_id,
        "base_model": MODEL_ID,
        "base_revision": REVISION,
        "policy": args.policy,
        "loss": args.loss,
        "negative_strategy": args.strategy,
        "learning_rate": args.lr,
        "seed": args.seed,
        "epochs": args.epochs,
        "device": device,
        "precision": "fp32",
        "micro_batch_size": args.micro_batch,
        "gradient_accumulation": max(1, (32 + args.micro_batch - 1) // args.micro_batch),
        "effective_source_batch_size": 32,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "train_sources": eligible_count,
        "test_data_used": False,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
