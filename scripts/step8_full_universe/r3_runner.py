from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch

from scripts.step8_full_universe import r2_runner as r2

OBJECTIVE = "SET_POSITIVE_LISTWISE"
STRATEGY = "RANDOM_WITHIN_CANDIDATE"
LEARNING_RATE = 3e-5
SEEDS = (17, 42, 2026)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark = r2.benchmark_rows()
    train = r2.training_groups(
        r2.prepare_groups(r2.CANDIDATE_ROOT / "forward_train_k100.jsonl.gz", benchmark), benchmark
    )
    dev = r2.prepare_groups(r2.CANDIDATE_ROOT / "forward_dev_k100.jsonl.gz", benchmark)
    if len(train) != 9224:
        raise AssertionError(f"unexpected supervised source count: {len(train)}")
    all_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        random.seed(seed)
        torch.manual_seed(seed)
        r2.SEED = seed
        seed_output = args.output / f"seed_{seed}"
        all_rows.extend(
            r2.run_config(
                "FINAL_SEED",
                OBJECTIVE,
                STRATEGY,
                LEARNING_RATE,
                seed_output,
                train,
                dev,
                benchmark,
                args.model_path,
            )
        )
    write_csv(args.output / "final_seed_dev.csv", all_rows)
    manifest = {
        "schema": "step8_final_seed_execution_manifest_v1",
        "status": "FINAL_SEEDS_DEV_COMPLETE",
        "seeds": list(SEEDS),
        "canonical_seed": 17,
        "objective": OBJECTIVE,
        "list_strategy": STRATEGY,
        "learning_rate": LEARNING_RATE,
        "model_id": "ncbi/MedCPT-Cross-Encoder",
        "model_revision": r2.MODEL_REVISION,
        "tokenizer_revision": r2.MODEL_REVISION,
        "training_source_count": len(train),
        "expanded_source_count": 75,
        "maximum_list_length": 50,
        "dropped_positive_count": 0,
        "positive_negative_collision_count": 0,
        "candidate_mutation_count": 0,
        "contract_v2_sha256": "17c3455e2ca2414f32fc11bfbbfb186dbc710b420d8b940042e5a5b390612d46",
        "protocol_v2_sha256": "5f1a9d34249acf15b451ac8434eccb57e41ff4702027359fcbf036bd156d600d",
        "config_freeze_sha256": "b82c0cba4b672726f0c3911fac00d5abe36263b0c20a36e0d261f3c14f5b4064",
        "test_scoring_count": 0,
        "test_training_count": 0,
    }
    (args.output / "final_seed_execution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "rows": len(all_rows), **manifest}, indent=2))


if __name__ == "__main__":
    main()
