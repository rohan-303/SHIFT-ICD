# ruff: noqa: E501,E702
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "artifacts/experiments/shift_map_v1_2/dev_replay_parts"
OUT = ROOT / "artifacts/experiments/shift_map_v1_2"
TABLES = ROOT / "reports/tables/shift_map_v1_2"
parts = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(PARTS.glob("*.json"))]
replay = {"experiment": "shift_map_v1_2", "evaluator_version": "2.0", "split": "dev", "training_occurred": False, "optimizer_created": False, "target_corpus_count": 17513, "checkpoint_count": len(parts), "checkpoints": sorted(parts, key=lambda x: x["model_dir"])}
OUT.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
(OUT / "dev_replay.json").write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
fields = ["run_id", "epoch", "policy", "loss", "negative_strategy", "learning_rate", "seed", "model_dir", "n", "ordinary_n", "complex_n", "Hit@1", "Hit@5", "Hit@10", "Hit@25", "Hit@50", "Hit@100", "MRR", "CompleteScenarioRetrieval@100"]
with (TABLES / "final_seed_dev_replay.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
    for item in replay["checkpoints"]:
        row = {k: item.get(k) for k in fields}; row.update({k: item["summary"].get(k) for k in fields if k in item["summary"]}); writer.writerow(row)
print(json.dumps({"checkpoint_count": len(parts), "table": str(TABLES / 'final_seed_dev_replay.csv')}, indent=2))