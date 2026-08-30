# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/experiments/shift_map_v1"
TABLE = ROOT / "reports/tables/shift_map_v1"


def load(path):
    return pd.DataFrame(json.loads(path.read_text())["rows"])


def ranking_frame(path):
    return pd.DataFrame(json.loads(line) for line in path.read_text(encoding="utf8").splitlines())


def bootstrap_delta(a, b, seed=2026, n_boot=5000):
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(n_boot, len(delta)))].mean(axis=1)
    return float(delta.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def metric(df, group, metrics):
    rows = []
    for keys, g in df.groupby(group, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        r = dict(zip(group, keys, strict=True))
        r.update(
            {
                "n": len(g),
                "protocol": "shift_map_v1",
                "experiment_version": "1.0",
                "partition": "test",
                "direction": str(g.direction.iloc[0]),
            }
        )
        for m in metrics:
            r[m] = float(g[m].mean()) if m in g else None
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    TABLE.mkdir(parents=True, exist_ok=True)
    files = sorted(OUT.glob("test_metrics*.json"))
    frames = []
    for p in files:
        d = load(p)
        d["seed"] = json.loads(p.read_text()).get("seed")
        frames.append(d)
    all_d = pd.concat(frames, ignore_index=True)
    all_d["alternative_size"] = pd.cut(all_d["n"], bins=[0, 1, 5, 20, 100, float("inf")], labels=["1", "2-5", "6-20", "21-100", ">100"])
    metrics = [
        "Hit@1",
        "Hit@5",
        "Hit@10",
        "Hit@25",
        "Hit@50",
        "Hit@100",
        "MRR",
        "ChoiceListRecall@10",
        "ChoiceListRecall@100",
        "CompleteScenarioRetrieval@10",
        "CompleteScenarioRetrieval@100",
    ]
    for name, group in [
        ("final_lexical_difficulty", ["direction", "lexical_difficulty"]),
        ("final_mapping_kind", ["direction", "mapping_kind"]),
        ("final_alternative_size", ["direction", "alternative_size"]),
        ("final_family_held_out", ["direction", "source_family_split"]),
        ("final_backward_transfer", ["direction"]),
        ("final_combination_transfer", ["direction", "mapping_kind"]),
    ]:
        metric(all_d, group, metrics).to_csv(TABLE / f"{name}.csv", index=False)
    overall = all_d[
        (all_d.direction == "ICD9CM_TO_ICD10CM")
        & (all_d.split == "test")
        & (all_d.mapping_kind != "NO_MAP")
        & (~all_d.mapping_kind.str.contains("COMBINATION"))
    ]
    rows = []
    for seed, g in overall.groupby("seed"):
        r = {
            "direction": "ICD9CM_TO_ICD10CM",
            "protocol": "shift_map_v1",
            "partition": "test",
            "experiment_version": "1.0",
            "seed": seed,
            "n": len(g),
        }
        for m in metrics:
            r[m] = float(g[m].mean())
        rows.append(r)
    pd.DataFrame(rows).to_csv(TABLE / "final_three_seed_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "direction": "ICD9CM_TO_ICD10CM",
                "protocol": "shift_map_v1",
                "partition": "test",
                "experiment_version": "1.0",
                "n": len(overall),
                **{m: float(overall[m].mean()) for m in metrics},
            }
        ]
    ).to_csv(TABLE / "final_forward_overall.csv", index=False)
    no = all_d[all_d.mapping_kind == "NO_MAP"]
    metric(no, ["direction"], ["top1_score", "top2_score", "mean_top5_similarity"]).to_csv(
        TABLE / "final_no_map_diagnostics.csv", index=False
    )
    base = ranking_frame(ROOT / "artifacts/experiments/dense_v1/rankings/biolord_2023/forward_stratified_test.jsonl")
    bm = ranking_frame(ROOT / "artifacts/experiments/bm25_v1_1/rankings/forward_stratified_test.jsonl").rename(
        columns=lambda x: "BM25_" + x if x not in {"benchmark_id"} else x
    )
    qw = ranking_frame(ROOT / "artifacts/experiments/dense_v1/rankings/qwen3_embedding_0.6b/forward_stratified_test.jsonl").rename(
        columns=lambda x: "Qwen3_" + x if x not in {"benchmark_id"} else x
    )
    _baseline_rankings = (bm, qw)
    sh = load(OUT / "test_metrics.json")
    paired = sh.merge(base[["benchmark_id", "Hit@10", "Hit@100", "MRR"]], on="benchmark_id", suffixes=("_shift", "_BioLORD"))
    paired = paired[
        (paired.direction == "ICD9CM_TO_ICD10CM")
        & (paired.split == "test")
        & (~paired.mapping_kind.str.contains("COMBINATION"))
        & (paired.mapping_kind != "NO_MAP")
    ]
    out = []
    for _system, other in [("BioLORD", "Hit@10"), ("BioLORD", "Hit@100"), ("BioLORD", "MRR")]:
        m = other
        mean, lo, hi = bootstrap_delta(paired[f"{m}_shift"], paired[f"{m}_BioLORD"])
        out.append(
            {
                "direction": "ICD9CM_TO_ICD10CM",
                "protocol": "shift_map_v1",
                "partition": "test",
                "comparison": "SHIFT-MAP minus BioLORD",
                "metric": m,
                "n": len(paired),
                "delta": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "bootstrap_seed": 2026,
                "resamples": 5000,
            }
        )
    pd.DataFrame(out).to_csv(TABLE / "final_paired_comparisons.csv", index=False)
    manifest = {
        "experiment": "shift_map_v1",
        "tables": sorted(p.name for p in TABLE.glob("*.csv")),
        "test_rows": len(all_d),
        "test_lock_sha256": hashlib.sha256((OUT / "test_lock.json").read_bytes()).hexdigest(),
        "test_data_used": True,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
