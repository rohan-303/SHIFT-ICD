from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from evaluate_shift_map_v1 import metrics
from sentence_transformers import SentenceTransformer

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.dense.corpus import BACKWARD, FORWARD, build_target_corpus

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"
CANONICAL = ROOT / "data/processed/cms/2018_gem/normalized_rows.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default=None)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = pd.read_parquet(CANONICAL)
    corpora = {direction: build_target_corpus(rows, direction) for direction in (FORWARD, BACKWARD)}
    examples = [BenchmarkExample.model_validate_json(line) for line in BENCHMARK.open(encoding="utf-8") if line.strip()]
    model_kwargs = {"device": device, "trust_remote_code": True}
    if args.revision:
        model_kwargs["revision"] = args.revision
    model = SentenceTransformer(args.model, **model_kwargs)
    output = {
        "experiment": "shift_map_v1_2",
        "evaluator_version": "2.0",
        "model": args.model,
        "revision": args.revision,
        "split": args.split,
        "target_corpora": {
            direction: {
                "count": len(corpus.codes),
                "terminology_version": corpus.terminology_version,
                "corpus_hash": corpus.corpus_hash,
            }
            for direction, corpus in corpora.items()
        },
        "populations": {},
    }
    for direction, corpus in corpora.items():
        selected = [
            example
            for example in examples
            if example.direction == direction and (args.split is None or example.split == args.split)
        ]
        output["populations"][direction] = {"n": len(selected), "rows": metrics(selected, model, corpus.as_dict(), device)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
