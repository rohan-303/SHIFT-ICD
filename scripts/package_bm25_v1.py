# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts/experiments/bm25_v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    selection = json.loads((EXP / "selection.json").read_text())
    tables = ROOT / "reports/tables/bm25_v1"
    def write(name: str, value: object) -> None:
        (EXP / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write("dev_metrics.json", {"selection_metric": selection["metric"], "grid": selection["grid"], "winner": selection["winner"]})
    write("slice_metrics.json", {path.name: path.read_text(encoding="utf-8") for path in tables.glob("bm25_by_*.csv")})
    errors = ROOT / "artifacts/error_analysis/bm25_v1_forward_errors.jsonl"
    error_count = sum(1 for line in errors.open(encoding="utf-8") if line.strip()) if errors.exists() else 0
    write("error_summary.json", {"forward_top10_failure_sample_size": error_count, "failure_category_policy": "placeholder left null; no unsupported clinical judgment"})
    files = {}
    for path in EXP.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(EXP)).replace("\\", "/")] = digest(path)
    for path in tables.glob("*.csv"):
        files[str(path.relative_to(ROOT)).replace("\\", "/")] = digest(path)
    for path in (ROOT / "reports/figures/bm25_v1").glob("*"):
        files[str(path.relative_to(ROOT)).replace("\\", "/")] = digest(path)
    report = ROOT / "reports/bm25_v1_baseline_report.md"
    files[str(report.relative_to(ROOT)).replace("\\", "/")] = digest(report)
    write("manifest.json", {"experiment": "bm25_v1", "benchmark_version": "1.0", "canonical_schema_version": "1.0", "selection_frozen": True, "files": files})
    print(json.dumps({"manifest_files": len(files), "error_sample": error_count}, sort_keys=True))


if __name__ == "__main__":
    main()
