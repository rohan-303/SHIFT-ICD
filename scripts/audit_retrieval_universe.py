from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from shift_icd.data.descriptions import (
    merge_descriptions,
    parse_icd9_long_titles,
    parse_icd9_short_titles,
    parse_icd10_codes,
    parse_icd10_order_short_titles,
)
from shift_icd.dense.corpus import BACKWARD, FORWARD

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/cms/2018_gem"
OUT = ROOT / "artifacts/data_audit/retrieval_universe_publication_audit.json"

def digest(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()

def independent_sets() -> tuple[dict[str, str], dict[str, str]]:
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        with zipfile.ZipFile(RAW / "2018-icd-10-code-descriptions.zip") as z:
            (t / "icd10_long").write_bytes(z.read("icd10cm_codes_2018.txt"))
            # The complete diagnosis universe is the authoritative code-description member.
            # The order file contains hierarchy rows, not a membership source.
            (t / "icd10_short").write_bytes(b"")
        with zipfile.ZipFile(RAW / "icd-9-cm-v32-master-descriptions.zip") as z:
            (t / "icd9_long").write_bytes(z.read("CMS32_DESC_LONG_DX.txt"))
            (t / "icd9_short").write_bytes(z.read("CMS32_DESC_SHORT_DX.txt"))
        icd10 = merge_descriptions(parse_icd10_codes(t / "icd10_long"), parse_icd10_order_short_titles(t / "icd10_short"))
        icd9 = merge_descriptions(parse_icd9_long_titles(t / "icd9_long"), parse_icd9_short_titles(t / "icd9_short"))
    return (
        {k: (v.long_description or v.short_description or "") for k, v in icd10.items()},
        {k: (v.long_description or v.short_description or "") for k, v in icd9.items()},
    )

def main() -> None:
    icd10, icd9 = independent_sets()
    rows = pd.read_parquet(ROOT / "data/processed/cms/2018_gem/normalized_rows.parquet")
    gem = {
        FORWARD: set(rows.loc[rows.direction == FORWARD, "target_code"].dropna().astype(str)),
        BACKWARD: set(rows.loc[rows.direction == BACKWARD, "target_code"].dropna().astype(str)),
    }
    full = {FORWARD: icd10, BACKWARD: icd9}
    result = {"schema": "retrieval-universe-publication-audit-v1", "source_files": {
        "icd10": "data/raw/cms/2018_gem/2018-icd-10-code-descriptions.zip",
        "icd9": "data/raw/cms/2018_gem/icd-9-cm-v32-master-descriptions.zip"}, "directions": {}}
    for direction in (FORWARD, BACKWARD):
        fs = set(full[direction])
        gs = gem[direction]
        inter = fs & gs
        result["directions"][direction] = {
            "n_full": len(fs), "n_gem_target": len(gs), "n_intersection": len(inter),
            "n_full_minus_gem": len(fs-gs), "n_gem_minus_full": len(gs-fs),
            "full_code_hash": digest(sorted(fs)),
            "full_code_description_hash": digest([f"{c}\t{full[direction][c]}" for c in sorted(fs)]),
            "full_minus_gem_sample": sorted(fs-gs)[:20], "gem_minus_full_sample": sorted(gs-fs)[:20],
            "historical_size": 17513 if direction == FORWARD else 11690,
            "historical_matches_full": len(fs) == (17513 if direction == FORWARD else 11690),
            "historical_matches_gem": len(gs) == (17513 if direction == FORWARD else 11690),
        }
    result["classification"] = "CU3" if all(
        v["n_full_minus_gem"] > 0 and v["historical_matches_gem"] for v in result["directions"].values()
    ) else "CU4"
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
