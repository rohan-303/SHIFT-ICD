# ruff: noqa
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.dense.corpus import FORWARD, build_target_corpus_from_terminology
from shift_icd.terminology import TerminologyCorpus, TerminologyRecord
from shift_icd.evaluation.retrieval import choice_list_recall_at_k, complete_scenario_retrieval_at_k
from shift_icd.shift_map.losses import l1_masked_single_infonce, l2_set_positive_infonce
from shift_icd.shift_map.training import TrainingExample, eligible_examples, sample_positive, source_balanced_epoch

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"
NEGATIVE_FILE = ROOT / "artifacts/experiments/shift_map_full_universe/negative_sets.jsonl"
OUT = ROOT / "artifacts/experiments/shift_map_full_universe/runs"
MODEL_ROOT = ROOT / "artifacts/models/shift_map_full_universe"
MODEL_ID = "FremyCompany/BioLORD-2023"
REVISION = "167aab527b238a50ca65224e6319215d2ff4fc9f"
UNIVERSE_HASH = "32f572abeb2ff4cb003145216fa83bfd9984fab96579205f432993aa9c550464"


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def load_data() -> tuple[list[TrainingExample], list[BenchmarkExample]]:
    train: list[TrainingExample] = []; all_rows: list[BenchmarkExample] = []
    with BENCHMARK.open(encoding="utf-8") as f:
        for line in f:
            x = BenchmarkExample.model_validate_json(line); all_rows.append(x)
            if x.direction == "ICD9CM_TO_ICD10CM" and x.split == "train":
                train.append(TrainingExample(x.benchmark_id, x.source_code, x.source_label or "", x.source_family,
                    x.direction, x.split or "", x.source_family_split or "", x.mapping_kind,
                    tuple(sorted(x.valid_target_codes)), str(x.lexical_metadata.get("lexical_difficulty", "UNKNOWN"))))
    return train, all_rows


def load_negatives() -> dict[str, dict[str, list[str]]]:
    with NEGATIVE_FILE.open(encoding="utf-8") as f: return {x["benchmark_id"]: x for x in map(json.loads, f)}


def embed_trainable(model: SentenceTransformer, texts: list[str], device: str) -> torch.Tensor:
    features = {k: v.to(device) for k, v in model.tokenize(texts).items()}
    return model(features)["sentence_embedding"]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordinary = [r for r in rows if not r["no_map"] and r["mapping_kind"] not in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}]
    complex_rows = [r for r in rows if not r["no_map"] and r["mapping_kind"] in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}]
    out: dict[str, Any] = {"n": len(rows), "applicable_metric_n": len(ordinary), "combination_n": len(complex_rows)}
    for k in (1, 5, 10, 25, 50, 100):
        out[f"Hit@{k}"] = float(np.mean([r[f"Hit@{k}"] for r in ordinary])) if ordinary else None
        out[f"CompleteScenarioRetrieval@{k}"] = float(np.mean([r[f"CompleteScenarioRetrieval@{k}"] for r in complex_rows])) if complex_rows else None
    out["MRR"] = float(np.mean([r["MRR"] for r in ordinary])) if ordinary else None
    low = [r for r in ordinary if r["lexical_difficulty"] == "LEXICAL_LOW"]
    out["LEXICAL_LOW_Hit@100"] = float(np.mean([r["Hit@100"] for r in low])) if low else None
    return out


def evaluate(model: SentenceTransformer, rows: list[BenchmarkExample], target_texts: dict[str, str], device: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval(); codes = sorted(target_texts)
    with torch.inference_mode():
        target = model.encode([target_texts[c] for c in codes], batch_size=256, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True, device=device, max_length=64).astype(np.float32)
        query = model.encode([r.source_label or "" for r in rows], batch_size=256, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True, device=device, max_length=64).astype(np.float32)
    result: list[dict[str, Any]] = []
    for example, vector in zip(rows, query, strict=True):
        scores = target @ vector; idx = np.argpartition(-scores, 99)[:100]; ranked = [codes[i] for i in sorted(idx.tolist(), key=lambda i: (-float(scores[i]), codes[i]))]
        valid = set(example.valid_target_codes); complex_mapping = example.mapping_kind in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}
        row: dict[str, Any] = {"benchmark_id": example.benchmark_id, "mapping_kind": example.mapping_kind, "lexical_difficulty": example.lexical_metadata.get("lexical_difficulty", "UNKNOWN"), "no_map": example.no_map, "valid_target_codes": sorted(valid)}
        for k in (1, 5, 10, 25, 50, 100):
            row[f"Hit@{k}"] = None if example.no_map or complex_mapping else float(bool(valid & set(ranked[:k])))
            row[f"ChoiceListRecall@{k}"] = choice_list_recall_at_k(example, ranked, k)
            row[f"CompleteScenarioRetrieval@{k}"] = complete_scenario_retrieval_at_k(example, ranked, k)
        row["MRR"] = None if example.no_map or complex_mapping else next((1.0 / rank for rank, code in enumerate(ranked, 1) if code in valid), 0.0)
        result.append(row)
    return summarize(result), result


def train_epoch(model: SentenceTransformer, optimizer: torch.optim.Optimizer, rows: list[TrainingExample], negatives: dict[str, dict[str, list[str]]], policy: str, loss_name: str, strategy: str, seed: int, epoch: int, target_texts: dict[str, str], device: str, micro_batch: int, temperature: float) -> float:
    eligible = source_balanced_epoch(eligible_examples(rows, policy), seed, epoch); model.train(); optimizer.zero_grad(set_to_none=True); values: list[float] = []
    accumulation = max(1, (32 + micro_batch - 1) // micro_batch)
    for start in range(0, len(eligible), micro_batch):
        batch = eligible[start:start + micro_batch]; selected = [sample_positive(r, seed, epoch) for r in batch]
        candidate_codes = list(dict.fromkeys(code for row, positive in zip(batch, selected, strict=True) for code in [*row.valid_target_codes, positive, *negatives[row.benchmark_id][strategy]]))
        for row in batch:
            if set(row.valid_target_codes) & set(negatives[row.benchmark_id][strategy]): raise RuntimeError(f"positive-negative collision: {row.benchmark_id}")
        query = embed_trainable(model, [r.source_label for r in batch], device); target = embed_trainable(model, [target_texts[c] for c in candidate_codes], device)
        candidates = target.unsqueeze(0).expand(len(batch), -1, -1)
        pos_idx = [candidate_codes.index(c) for c in selected]
        valid_idx = [[i for i, c in enumerate(candidate_codes) if c in set(r.valid_target_codes)] for r in batch]
        loss = l1_masked_single_infonce(query, candidates, pos_idx, valid_idx, temperature) if loss_name == "L1" else l2_set_positive_infonce(query, candidates, valid_idx, temperature)
        (loss / accumulation).backward(); values.append(float(loss.detach().cpu()))
        if ((start // micro_batch) + 1) % accumulation == 0 or start + micro_batch >= len(eligible):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    if not values: raise RuntimeError("no training batches")
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["P0", "P1", "P2"], required=True)
    parser.add_argument("--loss", choices=["L1", "L2"], required=True)
    parser.add_argument("--strategy", choices=["random", "lexical_hard", "dense_hard", "same_family_hard", "mixed"], required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    train, all_rows = load_data()
    negatives = load_negatives()
    terminology_path = ROOT / "artifacts/terminology_universe_v2/icd10cm_diagnosis.jsonl"
    terminology_bytes = terminology_path.read_bytes()
    if hashlib.sha256(terminology_bytes).hexdigest() != UNIVERSE_HASH:
        raise RuntimeError("canonical terminology file hash mismatch")
    target_records = [json.loads(line) for line in terminology_bytes.decode("utf-8").splitlines()]
    target_texts = {str(record["canonical_code"]): record.get("long_description") or record.get("short_description") or "" for record in target_records}
    if len(target_texts) != 71704 or [str(record["canonical_code"]) for record in target_records] != sorted(target_texts):
        raise RuntimeError("wrong corrected universe")
    dev = [x for x in all_rows if x.direction == "ICD9CM_TO_ICD10CM" and x.split == "dev"]
    model = SentenceTransformer(MODEL_ID, revision=REVISION, device=device, trust_remote_code=False)
    model.max_seq_length = 64
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    eligible_count = len(eligible_examples(train, args.policy))
    steps = max(1, (eligible_count + 31) // 32)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(1, int(steps * args.epochs * 0.1)))
    run_dir = MODEL_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        loss = train_epoch(model, optimizer, train, negatives, args.policy, args.loss, args.strategy, args.seed, epoch, target_texts, device, args.micro_batch, 0.05)
        scheduler.step()
        summary, rows = evaluate(model, dev, target_texts, device)
        history.append({"epoch": epoch + 1, "training_loss": loss, **summary})
        model.save(str(run_dir / f"epoch_{epoch + 1}"))
        (run_dir / f"epoch_{epoch + 1}_dev.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {"schema_version": "shift_map_full_universe_training_run_v1", "run_id": args.run_id, "base_model": MODEL_ID, "base_revision": REVISION, "policy": args.policy, "loss": args.loss, "negative_strategy": args.strategy, "learning_rate": args.lr, "seed": args.seed, "epochs": args.epochs, "device": device, "precision": "fp32", "micro_batch_size": args.micro_batch, "gradient_accumulation": max(1, (32 + args.micro_batch - 1) // args.micro_batch), "effective_source_batch_size": 32, "target_universe_count": 71704, "target_universe_hash": UNIVERSE_HASH, "history": history, "elapsed_seconds": time.perf_counter() - started, "train_sources": eligible_count, "test_data_used": False}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
