# ruff: noqa: E501,F401,I001
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from evaluate_shift_map_v1 import metrics
from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.dense.corpus import BACKWARD, FORWARD, build_target_corpus

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"
CANON = ROOT / "data/processed/cms/2018_gem/normalized_rows.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    device = args.device
    model_dir = ROOT / "artifacts/models/shift_map_v1" / f"final_seed{args.seed}" / f"epoch_{args.epoch}"
    rows = pd.read_parquet(CANON)
    corpora = {d: build_target_corpus(rows, d) for d in (FORWARD, BACKWARD)}
    examples = [BenchmarkExample.model_validate_json(line) for line in BENCH.open(encoding="utf-8")]
    model = SentenceTransformer(str(model_dir), device=device, trust_remote_code=True)
    result_rows = []
    for direction, corpus in corpora.items():
        selected = [e for e in examples if e.direction == direction and e.split == "test"]
        result_rows.extend(metrics(selected, model, corpus.as_dict(), device, batch_size=args.batch_size))
    output = {
        "experiment": "shift_map_v1_2",
        "evaluator_version": "2.0",
        "model_dir": str(model_dir.relative_to(ROOT)),
        "seed": args.seed,
        "epoch": args.epoch,
        "configuration_status": "DIAGNOSTIC_ONLY_CONFIGURATION_INVALIDATED_BY_CORRECTED_DEV_LOSS_SELECTION",
        "prior_test_exposure": True,
        "test_data_used_for_selection": False,
        "training_occurred": False,
        "target_corpora": {d: {"count": len(c.codes), "terminology_version": c.terminology_version, "corpus_hash": c.corpus_hash} for d, c in corpora.items()},
        "rows_sha256": hashlib.sha256(json.dumps(result_rows, sort_keys=True).encode()).hexdigest(),
        "rows": result_rows,
    }
    out = ROOT / "artifacts/experiments/shift_map_v1_2" / f"test_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"seed": args.seed, "rows": len(result_rows), "output": str(out), "rows_sha256": output["rows_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
 # ruff: noqa: E501,F401,I001
