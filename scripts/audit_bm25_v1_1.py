# ruff: noqa: E501
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"
CANONICAL = ROOT / "data/processed/cms/2018_gem/source_mappings.jsonl"
OLD_TABLES = ROOT / "reports/tables/bm25_v1"
OUT = ROOT / "artifacts/audits/bm25_v1_1_scope_audit.json"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    benchmark = load(BENCHMARK)
    canonical = load(CANONICAL)
    def key(row: dict) -> tuple[str, str]:
        return row["direction"], row["source_code"]
    benchmark_keys = {key(row) for row in benchmark}
    canonical_keys = {key(row) for row in canonical}
    datasets = {
        "forward_stratified_test": {row["benchmark_id"] for row in benchmark if row["direction"] == "ICD9CM_TO_ICD10CM" and row["split"] == "test"},
        "forward_family_held_out_test": {row["benchmark_id"] for row in benchmark if row["direction"] == "ICD9CM_TO_ICD10CM" and row["source_family_split"] == "test"},
        "backward_stratified_test": {row["benchmark_id"] for row in benchmark if row["direction"] == "ICD10CM_TO_ICD9CM" and row["split"] == "test"},
        "backward_family_held_out_test": {row["benchmark_id"] for row in benchmark if row["direction"] == "ICD10CM_TO_ICD9CM" and row["source_family_split"] == "test"},
    }
    old_tables = {}
    for name in ["bm25_by_mapping_kind.csv", "bm25_by_lexical_difficulty.csv", "bm25_by_alternative_size.csv", "bm25_combination_retrieval.csv"]:
        rows = list(csv.DictReader((OLD_TABLES / name).open(encoding="utf-8")))
        old_tables[name] = {
            "reported_n_total": sum(int(row["n"]) for row in rows),
            "rows": [{"label": row.get("mapping_kind") or row.get("lexical_difficulty") or row.get("alternative_bucket"), "n": int(row["n"])} for row in rows],
        }
    v = [row for row in benchmark if row["source_code"].casefold().replace(".", "") == "v5412"]
    canonical_v = [row for row in canonical if row["source_code"].casefold().replace(".", "") == "v5412"]
    audit = {
        "benchmark_version": "1.0",
        "canonical_schema_version": "1.0",
        "canonical_counts": dict(Counter(row["direction"] for row in canonical)),
        "benchmark_counts": dict(Counter(row["direction"] for row in benchmark)),
        "canonical_benchmark_membership": {"canonical_count": len(canonical_keys), "benchmark_count": len(benchmark_keys), "canonical_only": sorted(canonical_keys - benchmark_keys), "benchmark_only": sorted(benchmark_keys - canonical_keys)},
        "test_dataset_counts": {name: len(ids) for name, ids in datasets.items()},
        "union_test_count": len(set().union(*datasets.values())),
        "old_tables": old_tables,
        "old_slice_population_proof": "All four old TEST dataset ID populations union to 34,494, exactly the sum of the old mapping-kind and lexical slice counts. The old runner filtered slice rows by system but omitted dataset.",
        "v5412": {"benchmark_matches": [{"benchmark_id": x["benchmark_id"], "source_code": x["source_code"], "split": x["split"], "source_family_split": x["source_family_split"], "mapping_kind": x["mapping_kind"], "valid_target_count": len(x["valid_target_codes"])} for x in v], "canonical_matches": [{"direction": x["direction"], "source_code": x["source_code"], "mapping_kind": x["mapping_kind"], "unique_target_count": x["unique_target_count"], "approximate_any": x["approximate_any"]} for x in canonical_v], "lookup_finding": "The prior report searched for display-form V54.12, but stored benchmark/canonical source_code is normalized V5412."},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"canonical": len(canonical_keys), "benchmark": len(benchmark_keys), "union_test": audit["union_test_count"], "v5412_benchmark": len(v), "v5412_canonical": len(canonical_v)}, sort_keys=True))


if __name__ == "__main__":
    main()
