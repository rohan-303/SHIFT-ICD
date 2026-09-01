from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts/experiments/shift_map_v1_3"
ROWS = json.loads((EXP / "test_seed17.json").read_text())["populations"]["ICD9CM_TO_ICD10CM"]["rows"]
METRICS = ["Hit@1", "Hit@5", "Hit@10", "Hit@25", "Hit@50", "Hit@100", "MRR"]

def summarize(rows: list[dict]) -> list[object]:
    valid = [r for r in rows if r.get("valid_target_codes") and all(r.get(k) is not None for k in METRICS)]
    return [len(valid)] + [sum(float(r[k]) for r in valid) / len(valid) for k in METRICS] if valid else [0] + [""] * len(METRICS)

def write(name: str, groups: dict[str, list[dict]]) -> None:
    with (EXP / name).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "n", *METRICS])
        for group, rows in groups.items():
            writer.writerow([group, *summarize(rows)])

write("mapping_kind.csv", {g: [r for r in ROWS if r["mapping_kind"] == g] for g in sorted({r["mapping_kind"] for r in ROWS})})
write("lexical_slices.csv", {g: [r for r in ROWS if r["lexical_difficulty"] == g] for g in sorted({r["lexical_difficulty"] for r in ROWS})})
write(
    "family_held_out.csv",
    {g: [r for r in ROWS if r["source_family_split"] == g] for g in sorted({r["source_family_split"] for r in ROWS})},
)

def bucket(size: int) -> str:
    if size == 1:
        return "1"
    if size <= 5:
        return "2-5"
    if size <= 20:
        return "6-20"
    if size <= 100:
        return "21-100"
    return ">100"

write(
    "alternative_analysis.csv",
    {
        g: [r for r in ROWS if bucket(len(r.get("valid_target_codes", []))) == g]
        for g in ["1", "2-5", "6-20", "21-100", ">100"]
    },
)
print("recomputed canonical slices")
