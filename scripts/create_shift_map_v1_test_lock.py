from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/experiments/shift_map_v1/test_lock.json"


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"17", "42", "2026"}:
        raise SystemExit("usage: create_shift_map_v1_test_lock.py CANONICAL_SEED")
    canonical_seed = int(sys.argv[1])
    if OUT.exists():
        raise SystemExit("test_lock.json already exists; refusing to overwrite")
    config = json.loads((ROOT / "artifacts/experiments/shift_map_v1/protocol.json").read_text())
    config.update(
        {
            "positive_policy": "P2",
            "loss": "L1_masked_single_positive_infonce",
            "negative_strategy": "N1_RANDOM",
            "learning_rate": 2e-5,
            "temperature": 0.05,
            "optimizer": "AdamW",
            "epochs": 3,
            "checkpoint_policy": "best forward stratified DEV by preregistered lexicographic criterion",
            "seeds": [17, 42, 2026],
            "canonical_seed": canonical_seed,
        }
    )
    cfg_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    payload = {
        "experiment": "shift_map_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_model_revision": "167aab527b238a50ca65224e6319215d2ff4fc9f",
        "config": config,
        "config_sha256": cfg_hash,
        "git_head": head,
        "git_status_porcelain": status,
        "test_selection_locked": True,
        "test_data_used_before_lock": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
