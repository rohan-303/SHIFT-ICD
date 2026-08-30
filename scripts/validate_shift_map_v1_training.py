# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path

from shift_icd.benchmark.schemas import BenchmarkExample

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"
NEGATIVE_MANIFEST = ROOT / "artifacts/experiments/shift_map_v1/negative_mining_manifest.json"
NEGATIVE_SETS = ROOT / "artifacts/experiments/shift_map_v1/negative_sets.jsonl"


def main() -> None:
    examples = {}
    for line in BENCHMARK.open(encoding="utf-8"):
        example = BenchmarkExample.model_validate_json(line)
        examples[example.benchmark_id] = example
    rows = [json.loads(line) for line in NEGATIVE_SETS.open(encoding="utf-8")]
    violations: list[str] = []
    for row in rows:
        example = examples.get(row["benchmark_id"])
        if example is None:
            violations.append(f"unknown benchmark id: {row['benchmark_id']}")
            continue
        if not (example.direction == "ICD9CM_TO_ICD10CM" and example.split == "train"):
            violations.append(f"non-forward-train source: {example.benchmark_id}")
        if example.mapping_kind in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}:
            violations.append(f"complex source in pairwise training: {example.benchmark_id}")
        if example.no_map or not example.valid_target_codes:
            violations.append(f"NO_MAP/empty-positive source in training: {example.benchmark_id}")
        gold = set(example.valid_target_codes)
        for strategy in ("random", "lexical_hard", "dense_hard", "same_family_hard", "mixed"):
            collisions = gold.intersection(row.get(strategy, []))
            if collisions:
                violations.append(f"gold collision {example.benchmark_id} {strategy}: {sorted(collisions)}")
    manifest = json.loads(NEGATIVE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("dev_data_used") or manifest.get("test_data_used"):
        violations.append("negative mining manifest claims DEV/TEST use")
    if manifest.get("audit", {}).get("remaining_gold_collisions") != 0:
        violations.append("manifest does not report zero remaining collisions")
    if violations:
        raise SystemExit("TRAINING CONTAMINATION VALIDATION FAILED\n" + "\n".join(violations[:20]))
    print(json.dumps({"status": "passed", "training_sources": len(rows), "dev_ids_in_training": 0, "test_ids_in_training": 0, "backward_ids_in_training": 0, "complex_sources": 0, "no_map_sources": 0, "remaining_gold_collisions": 0}, indent=2))


if __name__ == "__main__":
    main()
