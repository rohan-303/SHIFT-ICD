from __future__ import annotations

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
art = root / "artifacts/experiments/shift_map_v1_2"
items = []
for path in sorted(art.glob("*.json")):
    if path.name in {"manifest.json", "zero_shot_test.json", "zero_shot_dev.json", "dev_replay.json"} or path.name.startswith("test_seed"):
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    items.append({"local_filename": path.name, "sha256": digest})
manifest = {
    "experiment": "shift_map_v1_2",
    "experiment_version": "1.2",
    "evaluator_version": "2.0",
    "benchmark_version": "1.0",
    "training_occurred": False,
    "optimizer_created": False,
    "prior_test_exposure": True,
    "artifacts": items,
}
(art / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote checksum manifest with {len(items)} artifacts")
