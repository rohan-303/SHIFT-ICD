"""Validate Track A benchmark v1.0 outputs and leakage constraints."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate(root: Path) -> None:
    output = root / "data/benchmarks/cms_track_a/v1.0"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["benchmark_version"] == "1.0"
    assert manifest["canonical_schema_version"] == "1.0"
    all_rows = read_jsonl(output / "all_examples.jsonl")
    assert len(all_rows) == sum(manifest["example_counts"].values())
    assert len({row["benchmark_id"] for row in all_rows}) == len(all_rows)
    assert len({(row["direction"], row["source_code"]) for row in all_rows}) == len(all_rows)
    by_id = {row["benchmark_id"]: row for row in all_rows}
    for row in all_rows:
        assert row["benchmark_version"] == "1.0"
        assert row["canonical_schema_version"] == "1.0"
        assert len(row["raw_row_ids"]) >= 1
        if row["combination"]:
            assert all(scenario["choice_lists"] for scenario in row["scenarios"])
    for direction_name in ("forward", "backward"):
        direction_rows = [row for row in all_rows if (direction_name == "forward") == (row["direction"] == "ICD9CM_TO_ICD10CM")]
        for protocol in ("stratified", "family_held_out"):
            seen: dict[str, str] = {}
            for partition in ("train", "dev", "test"):
                for ref in read_jsonl(output / direction_name / protocol / f"{partition}.jsonl"):
                    benchmark_id = str(ref["benchmark_id"])
                    assert benchmark_id in by_id
                    assert benchmark_id not in seen
                    seen[benchmark_id] = partition
            assert set(seen) == {str(row["benchmark_id"]) for row in direction_rows}
            if protocol == "family_held_out":
                families = {
                    partition: {str(by_id[bid]["source_family"]) for bid, assigned in seen.items() if assigned == partition}
                    for partition in ("train", "dev", "test")
                }
                assert not (
                    families["train"] & families["dev"] or families["train"] & families["test"] or families["dev"] & families["test"]
                )
    for item in manifest["files"]:
        path = output / str(item["filename"])
        assert path.exists()
        assert sha256(path) == item["sha256"]
    print(f"validated {len(all_rows)} Track A examples")


if __name__ == "__main__":
    validate(Path(__file__).resolve().parents[1])
