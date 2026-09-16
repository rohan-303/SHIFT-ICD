# ruff: noqa: E501
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from shift_icd.hierarchy.metadata import audit_hierarchy, build_prefix_hierarchy
from shift_icd.terminology import build_icd9_universe, build_icd10_universe

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/cms/2018_gem"
OUT_TABLE = ROOT / "reports/tables/step9_hierarchy/hierarchy_integrity.csv"
OUT_MANIFEST = ROOT / "artifacts/experiments/step9_hierarchy/hierarchy_metadata_manifest.json"
BENCHMARK = ROOT / "data/benchmarks/cms_track_a/v1.0/forward/all.jsonl"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def official_icd10_order_codes(path: Path) -> set[str]:
    with ZipFile(path) as z:
        lines = z.read("icd10cm_order_2018.txt").decode("cp1252").splitlines()
    return {line[6:14].strip().upper().replace(".", "") for line in lines if len(line) >= 14 and line[6:14].strip()}


def main() -> None:
    icd9 = build_icd9_universe(RAW / "icd-9-cm-v32-master-descriptions.zip")
    icd10 = build_icd10_universe(RAW / "2018-icd-10-code-descriptions.zip")
    source_codes = {r.canonical_code for r in icd9.records}
    target_codes = {r.canonical_code for r in icd10.records}
    source = build_prefix_hierarchy(source_codes, ontology="ICD-9-CM", provenance="CMS_V32_DIAGNOSIS_TITLES_PREFIX_STRUCTURE")
    target = build_prefix_hierarchy(target_codes, ontology="ICD-10-CM", provenance="CMS_FY2018_CODE_DESCRIPTIONS_PREFIX_STRUCTURE")
    order_codes = official_icd10_order_codes(RAW / "2018-icd-10-code-descriptions.zip")
    benchmark_sources: set[str] = set()
    with BENCHMARK.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            benchmark_sources.add(str(row["source_code"]).upper().replace(".", ""))
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    fields = ["ontology", "code", "parent_code", "depth", "root_code", "chapter", "block", "family", "provenance", "audit_status", "official_order_present"]
    with OUT_TABLE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ontology, nodes, official, observed_codes in (("ICD-9-CM", source, set(), source_codes), ("ICD-10-CM", target, order_codes, target_codes)):
            for code in sorted(observed_codes):
                node = nodes[code]
                status = "PASS" if node.parent_code is None or node.parent_code in nodes else "MISSING_PARENT"
                writer.writerow({"ontology": ontology, "code": code, "parent_code": node.parent_code or "", "depth": node.depth, "root_code": node.root_code, "chapter": node.chapter, "block": node.block, "family": node.family, "provenance": node.provenance, "audit_status": status, "official_order_present": "true" if (not official or code in official) else "false"})
    manifest = {
        "schema": "step9_hierarchy_metadata_manifest_v1",
        "status": "R1_PREFLIGHT_METADATA_COMPLETE",
        "source": {"ontology": "ICD-9-CM", "version": "Version 32", "node_count": len(source), "terminology_code_count": len(source_codes), "benchmark_source_code_count": len(benchmark_sources), "benchmark_source_coverage_count": len(benchmark_sources & source_codes), "benchmark_source_missing_count": len(benchmark_sources - source_codes), "audit": audit_hierarchy(source), "provenance": "CMS diagnosis titles; prefix structure only; no GEM"},
        "target": {"ontology": "ICD-10-CM", "version": "FY 2018", "node_count": len(target), "terminology_code_count": len(target_codes), "official_order_code_count": len(order_codes), "official_order_missing_count": len(target_codes - order_codes), "audit": audit_hierarchy(target), "provenance": "CMS code descriptions plus official tabular-order membership audit; prefix structure only; no GEM"},
        "candidate_target_universe_expected": 71704,
        "candidate_target_universe_coverage_count": len(target_codes),
        "integrity_table": str(OUT_TABLE.relative_to(ROOT)),
        "integrity_table_sha256": sha256(OUT_TABLE),
        "source_archive_sha256": sha256(RAW / "icd-9-cm-v32-master-descriptions.zip"),
        "target_archive_sha256": sha256(RAW / "2018-icd-10-code-descriptions.zip"),
        "gem_dependency": "NONE",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
