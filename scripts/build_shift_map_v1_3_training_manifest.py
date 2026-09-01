from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/experiments/shift_map_v1_3"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "experiment_version": "1.3",
    "benchmark": "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl",
    "benchmark_sha256": sha(ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"),
    "canonical": "data/processed/cms/2018_gem/normalized_rows.parquet",
    "canonical_sha256": sha(ROOT / "data/processed/cms/2018_gem/normalized_rows.parquet"),
    "negative_sets": "artifacts/experiments/shift_map_v1/negative_sets.jsonl",
    "negative_sets_sha256": sha(ROOT / "artifacts/experiments/shift_map_v1/negative_sets.jsonl"),
    "target_corpus_hash": "a394fe6a0055b13e2df4eaeb4ac4b739e7d8d54277adb02f9bdecff67ad80a3a",
    "evaluator_version": "2.0",
    "train_direction": "ICD9CM_TO_ICD10CM",
    "test_used_for_selection": False,
    "training_occurred": True,
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
