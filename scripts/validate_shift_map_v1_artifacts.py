from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/experiments/shift_map_v1"


def main():
    lock = OUT / "test_lock.json"
    assert lock.exists()
    x = json.loads((OUT / "manifest.json").read_text())
    assert hashlib.sha256(lock.read_bytes()).hexdigest() == x["test_lock_sha256"]
    assert set(x["tables"]) >= {
        "dev_positive_policy_ablation.csv",
        "dev_negative_strategy_ablation.csv",
        "dev_learning_rate_ablation.csv",
        "final_forward_overall.csv",
        "final_three_seed_results.csv",
        "final_lexical_difficulty.csv",
        "final_mapping_kind.csv",
        "final_alternative_size.csv",
        "final_combination_transfer.csv",
        "final_family_held_out.csv",
        "final_backward_transfer.csv",
        "final_paired_comparisons.csv",
        "final_no_map_diagnostics.csv",
        "final_complementarity.csv",
        "training_runtime.csv",
        "training_population.csv",
    }
    assert len(x["figures"]) >= 10
    seeds = json.loads((OUT / "seed_results.json").read_text())
    assert {s["seed"] for s in seeds} == {17, 42, 2026}
    assert json.loads(lock.read_text())["config"]["canonical_seed"] == 42
    assert json.loads((OUT / "negative_mining_manifest.json").read_text())["audit"]["remaining_gold_collisions"] == 0
    for p in ROOT.glob("artifacts/models/shift_map_v1/**/*"):
        if p.is_file():
            assert subprocess.run(["git", "check-ignore", "-q", str(p)], cwd=ROOT).returncode == 0, p
    print("SHIFT-MAP v1 artifact validation passed")


if __name__ == "__main__":
    main()
