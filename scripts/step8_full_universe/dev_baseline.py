from __future__ import annotations

import json
from pathlib import Path

from scripts.step8_full_universe.r2_runner import CANDIDATE_ROOT, benchmark_rows, metrics, prepare_groups

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/experiments/step8_full_universe/dev_baseline_frozen_order.json"


def main() -> None:
    benchmark = benchmark_rows()
    groups = prepare_groups(CANDIDATE_ROOT / "forward_dev_k100.jsonl.gz", benchmark)
    ranked = [[str(row["target_code"]) for row in group] for group in groups]
    result = {
        "schema": "step8_r2b_frozen_order_dev_baseline_v1",
        "status": "DESCRIPTIVE_ONLY",
        "candidate_dev_sha256": "c7240c1f38dd87f54d63c97999cba71d91f6d0887d89bc2d4094f9ba38ad6a6f",
        "source_count": len(groups),
        "candidate_k": 100,
        "medcpt_accessed": False,
        "metrics": metrics(groups, ranked, benchmark),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
