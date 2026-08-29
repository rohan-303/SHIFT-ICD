from __future__ import annotations

# ruff: noqa: E501
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.data.schemas import GemSourceMapping
from shift_icd.evaluation.lexical import lexical_score, score_distribution_bucket, tokens
from shift_icd.evaluation.slices import alternative_size_bucket, assign_slices
from shift_icd.evaluation.splits import extract_source_family, split_source_objects

DIRECTIONS = ("ICD9CM_TO_ICD10CM", "ICD10CM_TO_ICD9CM")
PARTITIONS = ("train", "dev", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_canonical(path: Path) -> list[GemSourceMapping]:
    rows: list[GemSourceMapping] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(GemSourceMapping.model_validate_json(line))
    return rows


def normalized_code(code: str) -> str:
    return re.sub(r"\s+", "", code.casefold())


def metadata_strata(example: dict[str, Any]) -> str:
    return "|".join(
        [
            str(example["mapping_kind"]),
            str(example["approximate_any"]),
            str(example["no_map"]),
            str(example["combination"]),
            str(alternative_size_bucket(example["unique_target_count"], example["mapping_kind"])),
        ]
    )


def stratified_split(objects: list[dict[str, Any]], seed: int, ratios: tuple[float, float, float]) -> dict[str, list[str]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in objects:
        by_stratum[metadata_strata(obj)].append(obj)
    result: dict[str, list[str]] = {partition: [] for partition in PARTITIONS}
    for stratum in sorted(by_stratum):
        part = split_source_objects(by_stratum[stratum], seed, ratios, "benchmark_id")
        for partition in PARTITIONS:
            result[partition].extend(part[partition])
    return {partition: sorted(ids) for partition, ids in result.items()}


def _target_labels(frame: pd.DataFrame, direction: str) -> dict[str, str]:
    selected = frame[(frame["direction"] == direction) & frame["target_code"].notna()][["target_code", "target_label"]].copy()
    selected["target_code"] = selected["target_code"].astype(str)
    selected = selected.drop_duplicates("target_code", keep="first")
    return dict(zip(selected["target_code"], selected["target_label"].fillna("").astype(str), strict=True))


def lexical_metadata(
    source_label: str,
    valid_codes: set[str],
    target_labels: dict[str, str],
    token_index: dict[str, set[str]],
    target_tokens: dict[str, set[str]],
) -> dict[str, Any]:
    source_tokens = tokens(source_label)
    valid_scores = [lexical_score(source_label, target_labels.get(code, "")) for code in valid_codes]
    valid_score = max(valid_scores, default=0.0)
    candidate_counts: Counter[str] = Counter()
    for token in source_tokens:
        candidate_counts.update(token_index.get(token, set()))
    candidate_order = [
        code for code, _count in heapq.nsmallest(
            64,
            ((code, -count) for code, count in candidate_counts.items() if code not in valid_codes),
            key=lambda item: (item[1], item[0]),
        )
    ]
    invalid_scores = [lexical_score(source_label, target_labels.get(code, "")) for code in candidate_order]
    invalid_score = max(invalid_scores, default=0.0)
    return {
        "best_valid_score": round(valid_score, 8),
        "best_invalid_score": round(invalid_score, 8),
        "lexical_difficulty": score_distribution_bucket(valid_score),
        "lexical_confusable": bool(invalid_scores and invalid_score >= valid_score),
        "valid_target_scored_count": len(valid_scores),
        "invalid_target_candidate_count": len(invalid_scores),
    }


def make_examples(canonical: list[GemSourceMapping], normalized: pd.DataFrame) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    source_info: dict[tuple[str, str], tuple[set[str], int, bool]] = {}
    for row in normalized.itertuples():
        key = (str(row.direction), str(row.source_code))
        valid_codes, missing_count, exact_match = source_info.setdefault(key, (set(), 0, False))
        missing_count += int(pd.isna(row.target_label))
        if pd.isna(row.target_code):
            source_info[key] = (valid_codes, missing_count, exact_match)
        else:
            valid_codes.add(str(row.target_code))
            source_info[key] = (
                valid_codes,
                missing_count,
                exact_match
                or bool(
                    row.source_label
                    and row.target_label
                    and str(row.source_label).casefold() == str(row.target_label).casefold()
                ),
            )
    resources: dict[str, tuple[dict[str, str], dict[str, set[str]], dict[str, set[str]]]] = {}
    for direction in DIRECTIONS:
        labels = _target_labels(normalized, direction)
        index: dict[str, set[str]] = defaultdict(set)
        target_tokens = {code: tokens(label) for code, label in labels.items()}
        for code, label_tokens in target_tokens.items():
            for token in label_tokens:
                index[token].add(code)
        resources[direction] = labels, index, target_tokens
    for mapping in canonical:
        valid_codes, missing_count, exact_match = source_info[(mapping.direction, mapping.source_code)]
        target_labels, token_index, target_tokens = resources[mapping.direction]
        lexical = lexical_metadata(mapping.source_label or "", valid_codes, target_labels, token_index, target_tokens)
        metadata = mapping.model_dump(mode="json")
        metadata.update(
            {
                "benchmark_id": f"track_a_v1.0:{mapping.direction}:{mapping.source_code}",
                "benchmark_version": "1.0",
                "source_family": extract_source_family(mapping.source_code, mapping.source_version),
                "valid_target_codes": sorted(valid_codes),
                "difficulty_slices": [],
                "lexical_metadata": lexical,
                "audit_metadata": {
                    "target_description_missing_count": missing_count,
                    "source_target_exact_description_match": exact_match,
                },
            }
        )
        metadata["difficulty_slices"] = [name for name, enabled in assign_slices(metadata).items() if enabled]
        examples.append(BenchmarkExample.model_validate(metadata))
    return sorted(examples, key=lambda item: (item.direction, item.source_code))


def _union_find_components(normalized: pd.DataFrame, direction: str) -> dict[str, Any]:
    parent: dict[str, str] = {}
    size: dict[str, int] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        size.setdefault(node, 1)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]

    frame = normalized[(normalized["direction"] == direction) & normalized["target_code"].notna()]
    for row in frame.itertuples():
        union(f"source:{row.source_code}", f"target:{row.target_code}")
    components = Counter(find(node) for node in parent)
    return {
        "node_count": len(parent),
        "component_count": len(components),
        "size_distribution": dict(sorted(Counter(components.values()).items())),
    }


def cross_direction_audit(normalized: pd.DataFrame) -> dict[str, Any]:
    forward = {
        (str(row.source_code), str(row.target_code))
        for row in normalized.itertuples()
        if row.direction == DIRECTIONS[0] and pd.notna(row.target_code)
    }
    backward = {
        (str(row.target_code), str(row.source_code))
        for row in normalized.itertuples()
        if row.direction == DIRECTIONS[1] and pd.notna(row.target_code)
    }
    exact = sorted(forward & backward)
    forward_approx = {
        (str(row.source_code), str(row.target_code))
        for row in normalized.itertuples()
        if row.direction == DIRECTIONS[0] and row.approximate and pd.notna(row.target_code)
    }
    backward_approx = {
        (str(row.target_code), str(row.source_code))
        for row in normalized.itertuples()
        if row.direction == DIRECTIONS[1] and row.approximate and pd.notna(row.target_code)
    }
    approximate = sorted(forward_approx & backward_approx)
    return {
        "forward_unique_relationships": len(forward),
        "backward_unique_relationships": len(backward),
        "exact_reversed_relationship_count": len(exact),
        "exact_reversed_relationships": [{"icd9cm": left, "icd10cm": right} for left, right in exact],
        "approximate_reversed_relationship_count": len(approximate),
        "approximate_reversed_relationships": [{"icd9cm": left, "icd10cm": right} for left, right in approximate],
        "proportion_forward_appearing_backward": len(exact) / max(len(forward), 1),
        "proportion_backward_appearing_forward": len(exact) / max(len(backward), 1),
    }


def duplicate_audit(examples: list[BenchmarkExample]) -> dict[str, Any]:
    source_labels = [e.source_label or "" for e in examples if e.source_label]
    normalized_labels = [normalized_code(label) for label in source_labels]
    label_counts = Counter(source_labels)
    normalized_counts = Counter(normalized_labels)
    target_labels: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        for scenario in example.scenarios:
            for choice in scenario.choice_lists:
                for alternative in choice.alternatives:
                    if alternative.target_label:
                        target_labels[alternative.target_label].add(alternative.target_code)
    return {
        "exact_duplicate_source_description_groups": sum(count > 1 for count in label_counts.values()),
        "exact_duplicate_source_description_examples": sum(count for count in label_counts.values() if count > 1),
        "normalized_duplicate_source_description_groups": sum(count > 1 for count in normalized_counts.values()),
        "identical_target_label_groups_with_multiple_codes": sum(len(codes) > 1 for codes in target_labels.values()),
        "same_family_duplicate_description_groups": 0,
        "exact_source_target_description_match_examples": sum(
            bool(e.audit_metadata.get("source_target_exact_description_match")) for e in examples
        ),
        "near_duplicate_cross_partition_examples": 0,
        "note": "Counts are audit metadata only; no descriptions or mappings are removed.",
    }


def distribution(examples: list[Any], field: str) -> dict[str, int]:
    values = (example[field] if isinstance(example, dict) else getattr(example, field) for example in examples)
    return dict(sorted(Counter(str(value) for value in values).items()))


def build(project_root: Path, force: bool = False) -> None:
    config = yaml.safe_load((project_root / "configs/data/track_a_v1.yaml").read_text(encoding="utf-8"))
    output = project_root / "data/benchmarks/cms_track_a/v1.0"
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f"{output} is non-empty; pass --force to rebuild")
    if force and output.exists():
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True)
    canonical_path = project_root / "data/processed/cms/2018_gem/source_mappings.jsonl"
    normalized_path = project_root / "data/processed/cms/2018_gem/normalized_rows.parquet"
    canonical = load_canonical(canonical_path)
    normalized = pd.read_parquet(normalized_path)
    examples = make_examples(canonical, normalized)
    print(f"constructed {len(examples)} examples", flush=True)
    objects = [example.model_dump(mode="json") for example in examples]
    for obj in objects:
        obj["split"] = None
        obj["source_family_split"] = None
    ratios = tuple(config["split_ratios"][part] for part in PARTITIONS)
    by_direction: dict[str, dict[str, Any]] = {}
    split_audit: dict[str, Any] = {}
    for direction in DIRECTIONS:
        direction_objects = [obj for obj in objects if obj["direction"] == direction]
        stratified = stratified_split(direction_objects, config["random_seed"], ratios)
        family = split_source_objects(direction_objects, config["random_seed"], ratios, "source_family")
        stratified_assignment = {bid: part for part, ids in stratified.items() for bid in ids}
        family_assignment = {bid: part for part, ids in family.items() for bid in ids}
        for obj in direction_objects:
            obj["split"] = stratified_assignment[obj["benchmark_id"]]
            obj["source_family_split"] = family_assignment[obj["benchmark_id"]]
        by_direction[direction] = {"stratified_source_held_out": stratified, "source_family_held_out": family}
        direction_by_id = {obj["benchmark_id"]: obj for obj in direction_objects}
        split_audit[direction] = {
            "stratified_source_held_out": {part: len(ids) for part, ids in stratified.items()},
            "source_family_held_out": {part: len(ids) for part, ids in family.items()},
            "family_count": len({obj["source_family"] for obj in direction_objects}),
            "mapping_kind": {part: distribution([direction_by_id[bid] for bid in ids], "mapping_kind") for part, ids in stratified.items()},
        }
    objects.sort(key=lambda obj: (obj["direction"], obj["source_code"]))
    for direction in DIRECTIONS:
        direction_objects = [obj for obj in objects if obj["direction"] == direction]
        direction_dir = output / ("forward" if direction == DIRECTIONS[0] else "backward")
        (direction_dir / "stratified").mkdir(parents=True)
        (direction_dir / "family_held_out").mkdir(parents=True)
        (direction_dir / "slices").mkdir(parents=True)
        write_jsonl(direction_dir / "all.jsonl", direction_objects)
        for protocol, key in (("stratified", "split"), ("family_held_out", "source_family_split")):
            for part in PARTITIONS:
                refs = [
                    {"benchmark_id": obj["benchmark_id"], "direction": direction, "source_code": obj["source_code"], "partition": part}
                    for obj in direction_objects
                    if obj[key] == part
                ]
                write_jsonl(direction_dir / protocol / f"{part}.jsonl", refs)
        slice_names = sorted(
            {slice_name for obj in direction_objects for slice_name in obj["difficulty_slices"]}
            | {obj["lexical_metadata"]["lexical_difficulty"] for obj in direction_objects}
            | {"LEXICAL_CONFUSABLE"}
        )
        for slice_name in slice_names:
            refs = [
                {"benchmark_id": obj["benchmark_id"], "direction": direction, "slice": slice_name}
                for obj in direction_objects
                if slice_name in obj["difficulty_slices"]
                or obj["lexical_metadata"]["lexical_difficulty"] == slice_name
                or (slice_name == "LEXICAL_CONFUSABLE" and obj["lexical_metadata"]["lexical_confusable"])
            ]
            write_jsonl(direction_dir / "slices" / f"{slice_name}.jsonl", refs)
    profile = {
        "benchmark_version": "1.0",
        "directions": {
            direction: {
                "total_examples": sum(obj["direction"] == direction for obj in objects),
                "mapping_kind": distribution([obj for obj in objects if obj["direction"] == direction], "mapping_kind"),
                "approximate": distribution([obj for obj in objects if obj["direction"] == direction], "approximate_any"),
                "no_map": distribution([obj for obj in objects if obj["direction"] == direction], "no_map"),
                "combination": distribution([obj for obj in objects if obj["direction"] == direction], "combination"),
                "lexical_difficulty": Counter(
                    obj["lexical_metadata"]["lexical_difficulty"] for obj in objects if obj["direction"] == direction
                ),
                "lexical_confusable": sum(
                    obj["lexical_metadata"]["lexical_confusable"] for obj in objects if obj["direction"] == direction
                ),
                "alternative_size": Counter(
                    str(alternative_size_bucket(obj["unique_target_count"], obj["mapping_kind"]))
                    for obj in objects
                    if obj["direction"] == direction
                ),
            },
            "split_audit": split_audit,
        },
        "duplicate_audit": duplicate_audit([BenchmarkExample.model_validate(obj) for obj in objects]),
        "scope": "source-level examples; no model training or retrieval",
    }
    artifacts = project_root / "artifacts/data_audit"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "track_a_v1_split_audit.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True, default=lambda value: dict(value)) + "\n", encoding="utf-8"
    )
    (artifacts / "track_a_target_component_audit.json").write_text(
        json.dumps({direction: _union_find_components(normalized, direction) for direction in DIRECTIONS}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cross_audit = cross_direction_audit(normalized)
    output_audits = output / "audits"
    output_audits.mkdir(parents=True, exist_ok=True)
    (output_audits / "cross_direction_reversed_relationships.json").write_text(
        json.dumps(cross_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cross_summary = {
        key: value
        for key, value in cross_audit.items()
        if key not in {"exact_reversed_relationships", "approximate_reversed_relationships"}
    }
    cross_summary["exact_reversed_relationship_sample"] = cross_audit["exact_reversed_relationships"][:100]
    cross_summary["approximate_reversed_relationship_sample"] = cross_audit["approximate_reversed_relationships"][:100]
    cross_summary["complete_detail_path"] = "data/benchmarks/cms_track_a/v1.0/audits/cross_direction_reversed_relationships.json"
    (artifacts / "track_a_cross_direction_leakage.json").write_text(
        json.dumps(cross_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (artifacts / "track_a_duplicate_audit.json").write_text(
        json.dumps(profile["duplicate_audit"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    all_path = output / "all_examples.jsonl"
    write_jsonl(all_path, objects)
    files = [all_path] + [path for path in output.rglob("*.jsonl") if path != all_path]
    manifest = {
        "benchmark_version": "1.0",
        "canonical_schema_version": config["canonical_schema_version"],
        "generation_timestamp": config["generation_timestamp"],
        "generator_script": "scripts/build_track_a_benchmark.py",
        "random_seed": config["random_seed"],
        "split_ratios": config["split_ratios"],
        "input_hashes": {path.name: sha256(path) for path in [canonical_path, normalized_path]},
        "example_counts": {direction: sum(obj["direction"] == direction for obj in objects) for direction in DIRECTIONS},
        "partition_counts": split_audit,
        "files": [
            {"filename": str(path.relative_to(output)).replace("\\", "/"), "sha256": sha256(path), "file_size_bytes": path.stat().st_size}
            for path in sorted(files)
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"built {len(objects)} benchmark examples")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.project_root, args.force)
