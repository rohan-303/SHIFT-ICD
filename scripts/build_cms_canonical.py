"""Build deterministic canonical Track A objects from immutable CMS archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from shift_icd.data.canonical import canonicalize_rows
from shift_icd.data.descriptions import (
    merge_descriptions,
    parse_icd9_long_titles,
    parse_icd9_short_titles,
    parse_icd10_codes,
    parse_icd10_order_short_titles,
)
from shift_icd.data.gem_parser import parse_gem_lines
from shift_icd.data.schemas import GemSourceMapping

SCHEMA_VERSION = "1.0"
DIRECTIONS = {
    "ICD9CM_TO_ICD10CM": ("2018_I9gem.txt", "ICD-9-CM", "ICD-10-CM"),
    "ICD10CM_TO_ICD9CM": ("2018_I10gem.txt", "ICD-10-CM", "ICD-9-CM"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_member_text(archive: ZipFile, member: str) -> list[str]:
    return archive.read(member).decode("ascii").splitlines()


def source_summary(source: GemSourceMapping) -> dict[str, object]:
    return source.model_dump(exclude={"scenarios", "raw_row_ids"})


def profile_sources(sources: list[GemSourceMapping], direction: str, enumeration_limit: int) -> dict[str, object]:
    def count(predicate):
        return sum(predicate(source) for source in sources)

    scenario_counts = Counter(source.scenario_count for source in sources)
    choice_counts = Counter(source.choice_list_count for source in sources)
    target_counts = Counter(source.unique_target_count for source in sources)
    set_counts = Counter(source.valid_mapping_set_count for source in sources)
    kinds = Counter(source.mapping_kind for source in sources)
    max_source = max(sources, key=lambda source: source.target_row_count)
    return {
        "direction": direction,
        "source_concepts": len(sources),
        "no_map_concepts": count(lambda s: s.no_map),
        "single_target_concepts": count(lambda s: s.unique_target_count == 1 and not s.combination),
        "multi_row_concepts": count(lambda s: s.target_row_count > 1),
        "multi_scenario_concepts": count(lambda s: s.scenario_count > 1),
        "combination_concepts": count(lambda s: s.combination),
        "combination_with_alternative_concepts": count(lambda s: s.mapping_kind == "COMBINATION_WITH_ALTERNATIVES"),
        "approximate_only_concepts": count(lambda s: s.approximate_all and s.approximate_any),
        "exact_only_concepts": count(lambda s: not s.approximate_any),
        "mixed_exact_approximate_concepts": count(lambda s: s.approximate_any and not s.approximate_all),
        "scenario_count_distribution": {str(k): v for k, v in sorted(scenario_counts.items())},
        "choice_list_count_distribution": {str(k): v for k, v in sorted(choice_counts.items())},
        "unique_target_count_distribution": {str(k): v for k, v in sorted(target_counts.items())},
        "theoretical_mapping_set_count_distribution": {str(k): v for k, v in sorted(set_counts.items())},
        "maximum_theoretical_mapping_set_count": max(source.valid_mapping_set_count for source in sources),
        "enumeration_safety_limit": enumeration_limit,
        "cases_exceeding_enumeration_safety_threshold": count(lambda s: s.valid_mapping_set_count > enumeration_limit),
        "mapping_kind_counts": dict(sorted(kinds.items())),
        "extreme_raw_row_case": {
            "source_code": max_source.source_code,
            "target_row_count": max_source.target_row_count,
            "scenario_count": max_source.scenario_count,
            "choice_list_count": max_source.choice_list_count,
            "valid_mapping_set_count": max_source.valid_mapping_set_count,
            "mapping_kind": max_source.mapping_kind,
        },
    }


def build(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    raw_root = project_root / "data/raw/cms/2018_gem"
    output = project_root / "data/processed/cms/2018_gem"
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"derived output exists; use --force to rebuild: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for path in output.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    gem_archive_path = raw_root / "2018-icd-10-cm-general-equivalence-mappings.zip"
    icd10_archive_path = raw_root / "2018-icd-10-code-descriptions.zip"
    icd9_archive_path = raw_root / "icd-9-cm-v32-master-descriptions.zip"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        with ZipFile(icd10_archive_path) as archive:
            icd10_long_path = temp / "icd10cm_codes_2018.txt"
            icd10_order_path = temp / "icd10cm_order_2018.txt"
            icd10_long_path.write_bytes(archive.read("icd10cm_codes_2018.txt"))
            icd10_order_path.write_bytes(archive.read("icd10cm_order_2018.txt"))
        with ZipFile(icd9_archive_path) as archive:
            icd9_long_path = temp / "CMS32_DESC_LONG_DX.txt"
            icd9_short_path = temp / "CMS32_DESC_SHORT_DX.txt"
            icd9_long_path.write_bytes(archive.read("CMS32_DESC_LONG_DX.txt"))
            icd9_short_path.write_bytes(archive.read("CMS32_DESC_SHORT_DX.txt"))
        icd10 = merge_descriptions(parse_icd10_codes(icd10_long_path), parse_icd10_order_short_titles(icd10_order_path))
        icd9 = merge_descriptions(parse_icd9_long_titles(icd9_long_path), parse_icd9_short_titles(icd9_short_path))

        all_normalized: list[dict[str, object]] = []
        all_sources: list[GemSourceMapping] = []
        profiles: list[dict[str, object]] = []
        with ZipFile(gem_archive_path) as archive:
            for direction, (member, _source_version, _target_version) in DIRECTIONS.items():
                raw_rows = parse_gem_lines(load_member_text(archive, member), direction)
                source_descriptions = icd9 if direction == "ICD9CM_TO_ICD10CM" else icd10
                target_descriptions = icd10 if direction == "ICD9CM_TO_ICD10CM" else icd9
                normalized, sources = canonicalize_rows(raw_rows, source_descriptions, target_descriptions, args.enumeration_limit)
                all_normalized.extend(row.model_dump() for row in normalized)
                all_sources.extend(sources)
                profiles.append(profile_sources(sources, direction, args.enumeration_limit))

    rows_df = pd.DataFrame(all_normalized)
    rows_df.to_parquet(output / "normalized_rows.parquet", index=False)
    with (output / "source_mappings.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for source in all_sources:
            handle.write(source.model_dump_json() + "\n")
    pd.DataFrame([source_summary(source) for source in all_sources]).to_parquet(output / "source_mappings.parquet", index=False)

    profile = {
        "canonical_schema_version": SCHEMA_VERSION,
        "profiling_scope": "canonical source-level representation; no train/dev/test splits",
        "directions": {item["direction"]: item for item in profiles},
        "description_coverage": {
            direction: {
                "source": description_coverage(all_normalized, direction, "source"),
                "target": description_coverage(all_normalized, direction, "target"),
            }
            for direction in DIRECTIONS
        },
    }
    audit_dir = project_root / "artifacts/data_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "cms_2018_canonical_profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outputs = [output / name for name in ["normalized_rows.parquet", "source_mappings.jsonl", "source_mappings.parquet"]]
    manifest = {
        "canonical_schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "processing_script": "scripts/build_cms_canonical.py",
        "raw_archive_hashes": {path.name: sha256_file(path) for path in [gem_archive_path, icd10_archive_path, icd9_archive_path]},
        "row_counts": {"normalized_rows": len(all_normalized), "source_mappings": len(all_sources)},
        "outputs": [
            {
                "filename": path.name,
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in outputs
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"built {len(all_normalized)} normalized rows and {len(all_sources)} source mappings")


def description_coverage(rows: list[dict[str, object]], direction: str, side: str) -> dict[str, object]:
    selected = [row for row in rows if row["direction"] == direction]
    key = f"{side}_label"
    short_key = f"{side}_short_description"
    codes = {str(row[f"{side}_code"]) for row in selected if row[f"{side}_code"]}
    available = {str(row[f"{side}_code"]) for row in selected if row[f"{side}_code"] and row[key]}
    short_available = {str(row[f"{side}_code"]) for row in selected if row[f"{side}_code"] and row[short_key]}
    return {
        "unique_codes": len(codes),
        "with_long_description": len(available),
        "missing_long_description": len(codes - available),
        "with_short_description": len(short_available),
        "missing_short_description": len(codes - short_available),
        "missing_long_codes": sorted(codes - available),
        "missing_short_codes": sorted(codes - short_available),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--enumeration-limit", type=int, default=1000)
    build(parser.parse_args())
