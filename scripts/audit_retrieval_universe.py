from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from shift_icd.dense.corpus import BACKWARD, FORWARD
from shift_icd.terminology import build_icd9_universe, build_icd10_universe

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/cms/2018_gem"
OUT = ROOT / "artifacts/data_audit/retrieval_universe_publication_audit.json"

def digest(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()

def main() -> None:
    icd10_corpus = build_icd10_universe(RAW / "2018-icd-10-code-descriptions.zip")
    icd9_corpus = build_icd9_universe(RAW / "icd-9-cm-v32-master-descriptions.zip")
    full = {
        FORWARD: {r.canonical_code: (r.long_description or r.short_description or "") for r in icd10_corpus.records},
        BACKWARD: {r.canonical_code: (r.long_description or r.short_description or "") for r in icd9_corpus.records},
    }
    rows = pd.read_parquet(ROOT / "data/processed/cms/2018_gem/normalized_rows.parquet")
    gem_raw = {
        FORWARD: {str(code) for code in rows.loc[rows.direction == FORWARD, "target_code"].dropna()},
        BACKWARD: {str(code) for code in rows.loc[rows.direction == BACKWARD, "target_code"].dropna()},
    }
    gem = {
        direction: {code.upper().replace(".", "") for code in codes}
        for direction, codes in gem_raw.items()
    }
    result = {"schema": "retrieval-universe-publication-audit-v1", "source_files": {
        "icd10": "data/raw/cms/2018_gem/2018-icd-10-code-descriptions.zip",
        "icd9": "data/raw/cms/2018_gem/icd-9-cm-v32-master-descriptions.zip"}, "directions": {}}
    for direction in (FORWARD, BACKWARD):
        fs = set(full[direction])
        gs = gem[direction]
        inter = fs & gs
        result["directions"][direction] = {
            "n_full": len(fs), "n_gem_target": len(gs), "n_gem_target_raw": len(gem_raw[direction]), "n_intersection": len(inter),
            "n_full_minus_gem": len(fs-gs), "n_gem_minus_full": len(gs-fs),
            "full_code_hash": digest(sorted(fs)),
            "full_code_description_hash": digest([f"{c}\t{full[direction][c]}" for c in sorted(fs)]),
            "full_minus_gem_sample": sorted(fs-gs)[:20], "gem_minus_full_sample": sorted(gs-fs)[:20],
            "historical_size": 17513 if direction == FORWARD else 11690,
            "historical_matches_full": len(fs) == (17513 if direction == FORWARD else 11690),
            "historical_matches_gem": len(gem_raw[direction]) == (17513 if direction == FORWARD else 11690),
        }
    result["classification"] = "CU3" if all(
        v["n_full_minus_gem"] > 0 and v["historical_matches_gem"] for v in result["directions"].values()
    ) else "CU4"
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
