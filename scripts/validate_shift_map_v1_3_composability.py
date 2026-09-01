from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/experiments/shift_map_v1_3"

rows = json.loads((OUT / "checkpoint_provenance.json").read_text())["rows"]
by_run = {}
for row in rows:
    by_run.setdefault(row["run_id"], row)

strategies = {by_run[k]["negative_strategy"] for k in by_run if k.startswith("dev_p2_l1_")}
learning_rates = {by_run[k]["learning_rate"] for k in by_run if k.startswith("dev_p2_l1_random_lr")}
result = {
    "experiment_version": "1.3",
    "train_only": True,
    "positive_policy": "P2",
    "historical_negative_strategy_runs": sorted(strategies),
    "historical_negative_strategy_loss": sorted({by_run[k]["loss"] for k in by_run if k.startswith("dev_p2_l1_")}),
    "historical_learning_rate_values": sorted(learning_rates),
    "historical_learning_rate_losses": sorted({by_run[k]["loss"] for k in by_run if k.startswith("dev_p2_l1_random_lr")}),
    "negative_strategy_bridge_required": True,
    "learning_rate_bridge_required": True,
    "selection_criterion": ["Hit@100", "CompleteScenarioRetrieval@100", "Hit@10", "MRR"],
}
(OUT / "configuration_composability.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
