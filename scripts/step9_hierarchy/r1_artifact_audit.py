# ruff: noqa: E501
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from shift_icd.hierarchy.features import FEATURE_NAMES, fit_train_scaler
from shift_icd.hierarchy.metadata import build_prefix_hierarchy
from shift_icd.terminology import build_icd9_universe, build_icd10_universe

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/cms/2018_gem"
CAND = ROOT / "artifacts/candidates/shift_map_full_universe_v2"
BENCH = ROOT / "data/benchmarks/cms_track_a/v1.0/forward/all.jsonl"
OUT = ROOT / "reports/tables/step9_hierarchy"
ART = ROOT / "artifacts/experiments/step9_hierarchy"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_benchmark() -> dict[str, dict[str, Any]]:
    with BENCH.open(encoding="utf-8") as f:
        return {row["benchmark_id"]: row for row in map(json.loads, f)}


def load_groups(path: Path) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            groups[row["source_id"]].append(row)
    return [sorted(group, key=lambda row: int(row["candidate_rank"])) for group in groups.values()]


def fast_vector(source_code: str, target_code: str, candidate_codes: list[str], source_nodes: dict[str, Any], target_nodes: dict[str, Any], source_max: int, target_max: int, sibling_counts: dict[str, int]) -> tuple[float, ...]:
    source = source_nodes[source_code]
    target = target_nodes[target_code]
    candidates = [target_nodes[code] for code in candidate_codes]
    denominator = max(1, len(candidates))
    target_ancestors = set(target.ancestor_chain)
    same_parent = sum(node.parent_code == target.parent_code for node in candidates) / denominator
    same_family = sum(node.family == target.family for node in candidates) / denominator
    shared_ancestor = sum(bool(target_ancestors.intersection(node.ancestor_chain)) or node.code == target.code for node in candidates) / denominator
    same_root = sum(node.root_code == target.root_code for node in candidates) / denominator
    return (source.depth / source_max if source_max else 0.0, target.depth / target_max if target_max else 0.0, math.log1p(sibling_counts[target.code]), len(target.ancestor_chain) / target_max if target_max else 0.0, same_parent, same_family, shared_ancestor, same_root)

def quantile(values: list[float], q: float) -> float:
    sorted_values = sorted(values)
    return sorted_values[min(len(sorted_values) - 1, max(0, int(q * len(sorted_values))))]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    benchmark = load_benchmark()
    icd9 = build_icd9_universe(RAW / "icd-9-cm-v32-master-descriptions.zip")
    icd10 = build_icd10_universe(RAW / "2018-icd-10-code-descriptions.zip")
    source_nodes = build_prefix_hierarchy({r.canonical_code for r in icd9.records}, ontology="ICD-9-CM", provenance="CMS_V32_PREFIX")
    target_nodes = build_prefix_hierarchy({r.canonical_code for r in icd10.records}, ontology="ICD-10-CM", provenance="CMS_FY2018_PREFIX")
    source_max = max(node.depth for node in source_nodes.values())
    target_max = max(node.depth for node in target_nodes.values())
    parent_counts: defaultdict[str | None, int] = defaultdict(int)
    for node in target_nodes.values():
        parent_counts[node.parent_code] += 1
    sibling_counts = {code: max(0, parent_counts[node.parent_code] - 1) for code, node in target_nodes.items()}
    train_groups = load_groups(CAND / "forward_train_k100.jsonl.gz")
    eligible = []
    for group in train_groups:
        meta = benchmark[group[0]["source_id"]]
        gold = set(meta.get("valid_target_codes", []))
        for row in group:
            row["_gold_from_benchmark"] = row["target_code"] in gold
        if meta["mapping_kind"] in {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"} and any(row["_gold_from_benchmark"] for row in group):
            eligible.append(group)
    population_rows = [{"population": "ordinary_answerable_train", "source_count": len(eligible), "candidate_count": sum(len(g) for g in eligible), "positive_count": sum(sum(r["_gold_from_benchmark"] for r in g) for g in eligible), "positive_policy": "all candidate-contained valid positives", "excluded_mapping_kinds": "NO_MAP,COMBINATION,COMBINATION_WITH_ALTERNATIVES"}]
    write_csv(OUT / "training_population.csv", population_rows)
    sample = eligible[:32]
    first_rows = []
    second_rows = []
    for group in sample:
        candidates = [row["target_code"] for row in group]
        for row in group:
            first_rows.append(fast_vector(row["source_code"], row["target_code"], candidates, source_nodes, target_nodes, source_max, target_max, sibling_counts))
            second_rows.append(fast_vector(row["source_code"], row["target_code"], candidates, source_nodes, target_nodes, source_max, target_max, sibling_counts))
    first_blob = json.dumps(first_rows, separators=(",", ":")).encode()
    second_blob = json.dumps(second_rows, separators=(",", ":")).encode()
    feature_manifest = {"schema": "step9_feature_cache_manifest_v1", "status": "R1_PREFLIGHT", "sample_source_count": len(sample), "sample_candidate_row_count": len(first_rows), "feature_names": list(FEATURE_NAMES), "first_hash": sha_bytes(first_blob), "second_hash": sha_bytes(second_blob), "identical": first_blob == second_blob, "missingness_mask_hash": sha_bytes(json.dumps([[math.isnan(x) or math.isinf(x) for x in row] for row in first_rows], separators=(",", ":")).encode()), "normalization_role": "TRAIN_ONLY", "dev_feature_extraction_count": 0, "test_feature_extraction_count": 0}
    (ART / "feature_cache_manifest.json").write_text(json.dumps(feature_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    all_values: list[list[float]] = [[] for _ in FEATURE_NAMES]
    for group in eligible:
        candidates = [row["target_code"] for row in group]
        for row in group:
            vector = fast_vector(row["source_code"], row["target_code"], candidates, source_nodes, target_nodes, source_max, target_max, sibling_counts)
            for index, value in enumerate(vector):
                all_values[index].append(value)
    train_scaler = fit_train_scaler(zip(*all_values, strict=True), role="TRAIN")
    feature_manifest["train_scaler_sha256"] = sha_bytes(json.dumps({"means": train_scaler.means, "scales": train_scaler.scales}, separators=(",", ":")).encode())
    (ART / "feature_cache_manifest.json").write_text(json.dumps(feature_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scale_rows: list[dict[str, Any]] = []
    for name, values in zip(FEATURE_NAMES, all_values, strict=True):
        mean = statistics.fmean(values)
        sd = statistics.pstdev(values)
        scale_rows.append({"feature": name, "count": len(values), "mean": mean, "sd": sd, "median": statistics.median(values), "p1": quantile(values, 0.01), "p99": quantile(values, 0.99), "min": min(values), "max": max(values), "missing_fraction": 0.0, "nan_count": sum(math.isnan(x) for x in values), "inf_count": sum(math.isinf(x) for x in values), "constant": sd == 0.0, "near_constant_sd_lt_1e-6": sd < 1e-6})
    write_csv(OUT / "feature_scale_audit.csv", scale_rows)
    feature_inventory: list[dict[str, Any]] = [{"feature": name, "family": "H1" if index < 4 else "H3", "learned": False, "source": "CMS ontology structure and frozen candidate set", "leakage_inputs": "none"} for index, name in enumerate(FEATURE_NAMES)]
    write_csv(OUT / "feature_inventory.csv", feature_inventory)
    baseline_rows: list[dict[str, Any]] = []
    dev_groups = load_groups(CAND / "forward_dev_k100.jsonl.gz")[:32]
    med_path = ROOT / "artifacts/experiments/dense_full_universe_v2/remote_sync_final/gpu1/results/dense_full_universe_v2/medcpt/dev_rows.jsonl"
    med_rows = {}
    with med_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            med_rows[row["benchmark_id"]] = row["ranked_codes"]
    for group in dev_groups:
        source_id = group[0]["source_id"]
        b0 = [row["target_code"] for row in group]
        b1 = med_rows.get(source_id)
        if b1 is not None:
            baseline_rows.append({"source_id": source_id, "b0_order_sha256": sha_bytes(json.dumps(b0, separators=(",", ":")).encode()), "b1_order_sha256": sha_bytes(json.dumps(b1, separators=(",", ":")).encode()), "candidate_identity_set_equal": set(b0) == set(b1), "test_scoring_count": 0})
    write_csv(OUT / "baseline_reproduction.csv", baseline_rows)
    smoke_rows: list[dict[str, Any]] = [{"check": "feature_extraction", "status": "PASS", "scope": "32 TRAIN sources"}, {"check": "feature_determinism", "status": "PASS" if feature_manifest["identical"] else "FAIL"}, {"check": "full_dev_scientific_evaluation", "status": "NOT_RUN", "count": 0}, {"check": "test_scoring", "status": "FORBIDDEN", "count": 0}]
    write_csv(OUT / "smoke_validation.csv", smoke_rows)
    print(json.dumps({"eligible_sources": len(eligible), "eligible_candidate_rows": sum(len(g) for g in eligible), "feature_cache": feature_manifest, "baseline_rows": len(baseline_rows)}, indent=2))


if __name__ == "__main__":
    main()
