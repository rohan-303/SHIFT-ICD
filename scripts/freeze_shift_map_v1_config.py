from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/experiments/shift_map_v1"

CRITERION = ["Hit@100", "CompleteScenarioRetrieval@100", "Hit@10", "MRR", "LEXICAL_LOW_Hit@100"]


def score(x):
    return tuple(x.get(k, -1) if x.get(k) is not None else -1 for k in CRITERION)


def best(meta):
    return max(meta["history"], key=score)


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def main():
    entries = []
    for p in sorted((ROOT / "artifacts/models/shift_map_v1").glob("dev_p2_l1_*")):
        m = p / "metadata.json"
        if m.exists():
            x = json.loads(m.read_text())
            b = best(x)
            entries.append(
                {
                    "run_id": x["run_id"],
                    "strategy": x["negative_strategy"],
                    "loss": x["loss"],
                    "policy": x["policy"],
                    "learning_rate": x["learning_rate"],
                    "checkpoint_epoch": b["epoch"],
                    "dev": b,
                    "test_data_used": x.get("test_data_used", True),
                }
            )
    chosen = max(entries, key=lambda x: score(x["dev"])) if entries else None
    result = {
        "experiment": "shift_map_v1",
        "selection_population": "forward_stratified_dev",
        "criterion": CRITERION,
        "test_data_used": False,
        "runs": entries,
        "selected": chosen,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "negative_strategy_ablation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
