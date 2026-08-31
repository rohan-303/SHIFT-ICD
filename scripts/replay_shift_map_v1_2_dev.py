# ruff: noqa: E501,F841,I001
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from evaluate_shift_map_v1 import metrics
from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.dense.corpus import FORWARD, build_target_corpus

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"
CANON = ROOT / "data/processed/cms/2018_gem/normalized_rows.parquet"
MODEL_ROOT = ROOT / "artifacts/models/shift_map_v1"
OUT = ROOT / "artifacts/experiments/shift_map_v1_2"


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordinary = [
        r
        for r in rows
        if r["valid_target_codes"]
        and r["mapping_kind"] not in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}
    ]
    complex_rows = [
        r
        for r in rows
        if r["valid_target_codes"]
        and r["mapping_kind"] in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}
    ]
    result: dict[str, Any] = {"n": len(rows), "ordinary_n": len(ordinary), "complex_n": len(complex_rows)}
    for k in (1, 5, 10, 25, 50, 100):
        result[f"Hit@{k}"] = sum(r[f"Hit@{k}"] for r in ordinary) / len(ordinary) if ordinary else None
        result[f"CompleteScenarioRetrieval@{k}"] = sum(r[f"CompleteScenarioRetrieval@{k}"] for r in complex_rows) / len(complex_rows) if complex_rows else None
    result["MRR"] = sum(r["MRR"] for r in ordinary) / len(ordinary) if ordinary else None
    result["selection_key"] = [result["Hit@100"], result["CompleteScenarioRetrieval@100"], result["Hit@10"], result["MRR"]]
    return result


def metadata(checkpoint: Path) -> dict[str, Any]:
    run = checkpoint.parent.name
    path = checkpoint.parent / "metadata.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {"run_id": run, "epoch": int(checkpoint.name.split("_")[-1]), "policy": data.get("policy"), "loss": data.get("loss"), "negative_strategy": data.get("negative_strategy"), "learning_rate": data.get("learning_rate"), "seed": data.get("seed")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    rows = pd.read_parquet(CANON)
    corpus = build_target_corpus(rows, FORWARD).as_dict()
    examples = [BenchmarkExample.model_validate_json(line) for line in BENCH.open(encoding="utf-8")]
    dev = [e for e in examples if e.direction == FORWARD and e.split == "dev"]
    if args.checkpoint:
        checkpoints = [ROOT / args.checkpoint]
    else:
        checkpoints = sorted(p for p in MODEL_ROOT.glob("*/epoch_*") if p.is_dir() and p.parent.name.startswith(("dev_", "final_seed")))
    output_path = OUT / "dev_replay.json"
    if args.checkpoint:
        checkpoint = checkpoints[0]
        print(f"replaying {checkpoint}", flush=True)
        info = metadata(checkpoint)
        model = SentenceTransformer(str(checkpoint), device=device, trust_remote_code=True)
        result_rows = metrics(dev, model, corpus, device, batch_size=args.batch_size)
        output = {**info, "model_dir": str(checkpoint.relative_to(ROOT)), "summary": summary(result_rows)}
        if info["run_id"].startswith("final_seed"):
            output["rows"] = result_rows
        part = OUT / "dev_replay_parts" / (checkpoint.parent.name + "_" + checkpoint.name + ".json")
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"part": str(part), "run_id": info["run_id"], "epoch": info["epoch"]}), flush=True)
        return


if __name__ == "__main__":
    main()
 # ruff: noqa: E501,F841,I001
