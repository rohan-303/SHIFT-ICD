from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts/models/shift_map_v1"
OUT = ROOT / "artifacts/experiments/shift_map_v1"


def main():
    entries = []
    for p in sorted(MODELS.glob("dev_*")):
        meta = p / "metadata.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text())
        if m.get("train_sources", 0) < 100:
            continue
        hist = m["history"]
        best = max(
            hist,
            key=lambda x: (
                x.get("Hit@100", -1),
                x.get("CompleteScenarioRetrieval@100") or -1,
                x.get("Hit@10", -1),
                x.get("MRR", -1),
                x.get("LEXICAL_LOW_Hit@100", -1),
            ),
        )
        entries.append(
            {
                "run_id": m["run_id"],
                "policy": m["policy"],
                "loss": m["loss"],
                "negative_strategy": m["negative_strategy"],
                "learning_rate": m["learning_rate"],
                "seed": m["seed"],
                "checkpoint_epoch": best["epoch"],
                "dev": best,
                "test_data_used": m.get("test_data_used", True),
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "positive_policy_ablation.json").write_text(
        json.dumps(
            {
                "experiment": "shift_map_v1",
                "selection_population": "forward_stratified_dev",
                "test_data_used": False,
                "criterion": ["Hit@100", "CompleteScenarioRetrieval@100", "Hit@10", "MRR", "LEXICAL_LOW_Hit@100"],
                "runs": entries,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(entries, indent=2))


if __name__ == "__main__":
    main()
