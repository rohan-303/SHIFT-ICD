# ruff: noqa
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from shift_map_v1_4_analysis import bootstrap_delta

ROOT = Path(__file__).resolve().parents[1]
LED = ROOT / "artifacts/experiments/shift_map_v1_4/ledgers"
EXP = ROOT / "artifacts/experiments/shift_map_v1_4"
TAB = ROOT / "reports/tables/shift_map_v1_4"
METRICS = ["Hit@1", "Hit@10", "Hit@100", "MRR"]

def load(model: str, direction: str, split: str) -> dict[str, dict]:
    rows = json.loads((LED / f"{model}_{direction}_{split}.json").read_text(encoding="utf8"))["rows"]
    return {r["benchmark_id"]: r for r in rows}

def paired(name: str, l2: dict[str, dict], zs: dict[str, dict], predicate, metrics: list[str]) -> list[dict]:
    ids = sorted(i for i in l2.keys() & zs.keys() if predicate(l2[i]))
    out = []
    for metric in metrics:
        valid = [i for i in ids if l2[i].get(metric) is not None and zs[i].get(metric) is not None]
        x = np.asarray([float(l2[i][metric]) for i in valid], dtype=float)
        y = np.asarray([float(zs[i][metric]) for i in valid], dtype=float)
        result = bootstrap_delta(x, y, seed=2026, reps=5000)
        out.append({"population": name, "metric": metric, "n": len(valid), "delta_l2_minus_zero_shot": result["delta"], "ci95_low": result["ci95_low"], "ci95_high": result["ci95_high"], "bootstrap_seed": 2026, "bootstrap_reps": 5000})
    return out

def main() -> None:
    ft_l2 = load("l2_seed17", "forward", "test"); ft_zs = load("zero_shot", "forward", "test")
    bt_l2 = load("l2_seed17", "backward", "test"); bt_zs = load("zero_shot", "backward", "test")
    rows = []
    rows += paired("forward_family_held_out", ft_l2, ft_zs, lambda r: r.get("source_family_split") == "test", ["Hit@10", "Hit@100", "MRR"])
    rows += paired("backward_stratified", bt_l2, bt_zs, lambda r: True, ["Hit@10", "Hit@100", "MRR"])
    rows += paired("backward_family_held_out", bt_l2, bt_zs, lambda r: r.get("source_family_split") == "test", ["Hit@10", "Hit@100", "MRR"])
    rows += paired("lexical_low", ft_l2, ft_zs, lambda r: r.get("lexical_difficulty") == "LEXICAL_LOW", METRICS)
    (EXP / "paired_bootstrap_missing.json").write_text(json.dumps({"method": "paired benchmark-id bootstrap", "rows": rows}, indent=2) + "\n", encoding="utf8")
    for name, population in [("family_held_out_paired.csv", "forward_family_held_out"), ("backward_transfer_paired.csv", "backward_stratified"), ("backward_family_held_out_paired.csv", "backward_family_held_out"), ("lexical_low_paired.csv", "lexical_low")]:
        selected = [r for r in rows if r["population"] == population]
        with (TAB / name).open("w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=list(selected[0] if selected else {"status": "NOT_COMPUTED"})); w.writeheader(); w.writerows(selected or [{"status": "NOT_COMPUTED"}])
    print(f"paired_rows={len(rows)}")

if __name__ == "__main__":
    main()
