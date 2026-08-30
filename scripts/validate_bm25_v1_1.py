# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts/experiments/bm25_v1_1"
BENCHMARK = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    benchmark = [json.loads(line) for line in BENCHMARK.open(encoding="utf-8") if line.strip()]
    rows = [json.loads(line) for line in (EXP / "scoped_rows.jsonl").open(encoding="utf-8") if line.strip()]
    expected = {("ICD9CM_TO_ICD10CM", "stratified", "test"): 2913, ("ICD9CM_TO_ICD10CM", "family_held_out", "test"): 2908, ("ICD10CM_TO_ICD9CM", "stratified", "test"): 14341, ("ICD10CM_TO_ICD9CM", "family_held_out", "test"): 14332}
    for scope, count in expected.items():
        scoped = [row for row in rows if (row["direction"], row["split_protocol"], row["partition"]) == scope]
        assert len(scoped) == count, (scope, len(scoped), count)
        assert {row["sample_population"] for row in scoped} == {next(name for name in ["forward_stratified_test", "forward_family_held_out_test", "backward_stratified_test", "backward_family_held_out_test"] if name.startswith("forward" if scope[0].startswith("ICD9") else "backward") and ("family" in name) == (scope[1] == "family_held_out") and ("stratified" in name) == (scope[1] == "stratified"))}
    forward = [row for row in rows if row["sample_population"] == "forward_stratified_test"]
    assert len({row["benchmark_id"] for row in forward}) == len(forward)
    kinds = {row["benchmark_id"] for row in forward if row["mapping_kind"] in {"ALTERNATIVE", "COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "NO_MAP", "SINGLE_APPROXIMATE", "SINGLE_EXACT"}}
    assert len(kinds) == len(forward)
    lexical = {row["benchmark_id"] for row in forward if row["lexical_difficulty"] in {"LEXICAL_EXACT", "LEXICAL_HIGH", "LEXICAL_MEDIUM", "LEXICAL_LOW", "LEXICAL_CONFUSABLE"}}
    assert len(lexical) == len(forward)
    benchmark_ids = {row["benchmark_id"] for row in benchmark if row["direction"] == "ICD9CM_TO_ICD10CM" and row["split"] == "test"}
    assert {row["benchmark_id"] for row in forward} == benchmark_ids
    manifest = json.loads((EXP / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected_hash in manifest["files"].items():
        assert sha256(EXP / relative) == expected_hash, relative
    print(json.dumps({"validated_scoped_rows": len(rows), "forward_mapping_partition": len(kinds), "forward_lexical_partition": len(lexical), "manifest_files": manifest["file_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
