"""Validate canonical Track A outputs against raw CMS row counts and manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from shift_icd.data.gem_parser import parse_gem_lines

EXPECTED = {
    "ICD9CM_TO_ICD10CM": ("2018_I9gem.txt", 24860, 14567),
    "ICD10CM_TO_ICD9CM": ("2018_I10gem.txt", 81593, 71704),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = root / "data/raw/cms/2018_gem/2018-icd-10-cm-general-equivalence-mappings.zip"
    output = root / "data/processed/cms/2018_gem"
    with ZipFile(raw) as archive:
        raw_rows = []
        for direction, (member, expected_rows, expected_sources) in EXPECTED.items():
            rows = parse_gem_lines(archive.read(member).decode("ascii").splitlines(), direction)
            assert len(rows) == expected_rows
            assert len({row.source_code_raw.strip() for row in rows}) == expected_sources
            raw_rows.extend(rows)
    normalized = pd.read_parquet(output / "normalized_rows.parquet")
    assert len(normalized) == len(raw_rows)
    assert set(normalized["direction"]) == set(EXPECTED)
    assert normalized["row_id"].nunique() == len(raw_rows)
    with (output / "source_mappings.jsonl").open(encoding="utf-8") as handle:
        sources = [json.loads(line) for line in handle if line.strip()]
    assert len(sources) == sum(item[2] for item in EXPECTED.values())
    assert all(source["canonical_schema_version"] == "1.0" for source in sources)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_counts"]["normalized_rows"] == len(raw_rows)
    for item in manifest["outputs"]:
        path = output / item["filename"]
        assert sha256(path) == item["sha256"]
    print(f"validated {len(raw_rows)} rows, {len(sources)} source mappings")


if __name__ == "__main__":
    main()
