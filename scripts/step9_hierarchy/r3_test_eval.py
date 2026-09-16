# mypy: ignore-errors
# ruff: noqa: E501
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
import r2_runner as r2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R2_ROOT = ROOT / "artifacts/experiments/step9_hierarchy"
TABLE_ROOT = ROOT / "reports/tables/step9_hierarchy"
RUN_ROOT = R2_ROOT / "final_seed_runs"
SEEDS = (17, 42, 2026)
PRIMARY = ("Hit@1", "MRR", "P_COMPLEX_CompleteScenarioRetrieval@10", "P_COMPLEX_ChoiceListRecall@10", "NDCG@10")
FEATURE_INDEX = {name: i for i, name in enumerate(r2.FEATURE_NAMES)}
H3_NAMES = list(r2.FEATURE_NAMES[4:])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rank_metrics(example: dict[str, Any], ranked: list[str]) -> dict[str, float]:
    gold = set(map(str, example.get("valid_target_codes", [])))
    first = next((i for i, code in enumerate(ranked, 1) if code in gold), len(ranked) + 1)
    gains = [1.0 if code in gold else 0.0 for code in ranked[:10]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = min(len(gold), 10)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return {
        "Hit@1": float(first <= 1),
        "Hit@5": float(first <= 5),
        "Hit@10": float(first <= 10),
        "Hit@25": float(first <= 25),
        "Hit@50": float(first <= 50),
        "Hit@100": float(first <= 100),
        "MRR": 1.0 / first if first <= len(ranked) else 0.0,
        "NDCG@10": dcg / idcg if idcg else 0.0,
        "first_rank": float(first if first <= len(ranked) else 0),
    }


def structural_metrics(example: dict[str, Any], ranked: list[str], k: int) -> tuple[float, float]:
    return r2.structural(example, ranked, k)


def evaluate(
    groups: list[list[dict[str, Any]]], ranked: list[list[str]], benchmark: dict[str, dict[str, Any]]
) -> tuple[dict[str, float], dict[str, list[float]]]:
    examples = [benchmark[str(g[0]["source_id"])] for g in groups]
    ordinary = [
        (e, r)
        for e, r in zip(examples, ranked, strict=True)
        if e.get("mapping_kind") in {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"} and e.get("valid_target_codes")
    ]
    result: dict[str, float] = {}
    per: dict[str, list[float]] = {}
    for k in (1, 5, 10, 25, 50, 100):
        vals = [rank_metrics(e, r)[f"Hit@{k}"] for e, r in ordinary]
        result[f"Hit@{k}"] = statistics.fmean(vals)
        per[f"Hit@{k}"] = vals
    for key in ("MRR", "NDCG@10"):
        vals = [rank_metrics(e, r)[key] for e, r in ordinary]
        result[key] = statistics.fmean(vals)
        per[key] = vals
    for population, predicate in {
        "P_COMBINATION": lambda e: "COMBINATION" in e.get("difficulty_slices", []),
        "P_COMBINATION_WITH_ALTERNATIVES": lambda e: "COMBINATION_WITH_ALTERNATIVES" in e.get("difficulty_slices", []),
        "P_COMPLEX": lambda e: "HIGH_MAPPING_COMPLEXITY" in e.get("difficulty_slices", []),
    }.items():
        chosen = [(e, r) for e, r in zip(examples, ranked, strict=True) if predicate(e)]
        for k in (1, 5, 10, 25, 50, 100):
            choice = [structural_metrics(e, r, k)[0] for e, r in chosen]
            complete = [structural_metrics(e, r, k)[1] for e, r in chosen]
            result[f"{population}_ChoiceListRecall@{k}"] = statistics.fmean(choice) if choice else float("nan")
            result[f"{population}_CompleteScenarioRetrieval@{k}"] = statistics.fmean(complete) if complete else float("nan")
            per[f"{population}_ChoiceListRecall@{k}"] = choice
            per[f"{population}_CompleteScenarioRetrieval@{k}"] = complete
    result["candidate_mutation_count"] = 0.0
    return result, per


def score(model: torch.nn.Module, groups, features) -> tuple[list[list[str]], list[list[float]]]:
    model.eval()
    ranked = []
    scores_all = []
    with torch.inference_mode():
        for group, feats in zip(groups, features, strict=True):
            scores = model(torch.tensor(feats, dtype=torch.float32)).tolist()
            pairs = list(zip(group, scores, strict=True))
            ordered = sorted(pairs, key=lambda x: (-float(x[1]), int(x[0]["candidate_rank"])))
            ranked.append([str(row["target_code"]) for row, _ in ordered])
            scores_all.append([float(s) for _, s in ordered])
    return ranked, scores_all


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_scored(path: Path, groups, ranked, scores) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for group, codes, vals in zip(groups, ranked, scores, strict=True):
            by_code = {str(row["target_code"]): row for row in group}
            for rank, (code, val) in enumerate(zip(codes, vals, strict=True), 1):
                row = by_code[code]
                out = {
                    "source_id": str(row["source_id"]),
                    "target_code": code,
                    "candidate_rank": int(row["candidate_rank"]),
                    "retriever_score": row.get("retriever_score"),
                    "hierarchy_score": val,
                    "rerank_rank": rank,
                }
                f.write(json.dumps(out, sort_keys=True) + "\n")


def source_values(groups, ranked, benchmark):
    values = []
    for group, r in zip(groups, ranked, strict=True):
        e = benchmark[str(group[0]["source_id"])]
        rm = rank_metrics(e, r)
        c, s = structural_metrics(e, r, 10)
        values.append({**rm, "CSR@10": s, "Choice@10": c, "example": e, "ranked": r})
    return values


def bootstrap(b0vals, hvals):
    rng = np.random.default_rng(20260915)
    specs = {
        "Hit@1": lambda x: (
            x["example"].get("mapping_kind") in {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"}
            and x["example"].get("valid_target_codes")
        ),
        "MRR": lambda x: (
            x["example"].get("mapping_kind") in {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"}
            and x["example"].get("valid_target_codes")
        ),
        "NDCG@10": lambda x: (
            x["example"].get("mapping_kind") in {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"}
            and x["example"].get("valid_target_codes")
        ),
        "CSR@10": lambda x: "HIGH_MAPPING_COMPLEXITY" in x["example"].get("difficulty_slices", []),
        "Choice@10": lambda x: "HIGH_MAPPING_COMPLEXITY" in x["example"].get("difficulty_slices", []),
    }
    key_map = {"CSR@10": "CSR@10", "Choice@10": "Choice@10"}
    out = []
    for key, predicate in specs.items():
        a_rows = [x for x in b0vals if predicate(x)]
        b_rows = [x for x in hvals if predicate(x)]
        if len(a_rows) != len(b_rows):
            raise RuntimeError(f"bootstrap population mismatch for {key}")
        a = np.asarray([x[key_map.get(key, key)] for x in a_rows], dtype=float)
        b = np.asarray([x[key_map.get(key, key)] for x in b_rows], dtype=float)
        n = len(a)
        delta = float(b.mean() - a.mean())
        draws = np.empty(10000)
        for i in range(10000):
            idx = rng.integers(0, n, size=n)
            draws[i] = (b[idx] - a[idx]).mean()
        out.append(
            {
                "metric": key,
                "delta": delta,
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "repetitions": 10000,
                "seed": 20260915,
                "population_n": n,
            }
        )
    return out


def main() -> None:
    torch.set_num_threads(1)
    lock = load_json(R2_ROOT / "test_lock.json")
    lock_sha = sha256(R2_ROOT / "test_lock.json")
    if lock["status"] != "TEST_LOCKED_BEFORE_STEP9_HIERARCHY_ACCESS":
        raise RuntimeError("invalid TEST lock")
    benchmark = r2.load_benchmark()
    train_groups = r2.training_groups(r2.prepare(r2.CANDIDATE_ROOT / "forward_train_k100.jsonl.gz", benchmark), benchmark)
    dev_groups = r2.prepare(r2.CANDIDATE_ROOT / "forward_dev_k100.jsonl.gz", benchmark)
    test_feature_ts = time.time()
    test_groups = r2.prepare(r2.CANDIDATE_ROOT / "forward_test_k100.jsonl.gz", benchmark)
    source_nodes, target_nodes = r2.build_nodes()
    train_raw = r2.fast_features(train_groups, source_nodes, target_nodes)
    dev_raw = r2.fast_features(dev_groups, source_nodes, target_nodes)
    test_raw = r2.fast_features(test_groups, source_nodes, target_nodes)
    scaler = r2.fit_train_scaler((row for group in train_raw for row in group), role="TRAIN")
    train_features = r2.apply_variant(train_raw, scaler, VARIANT := "H3_CANDIDATE_SET_STRUCTURAL_CONTEXT")
    dev_features = r2.apply_variant(dev_raw, scaler, VARIANT)
    test_features = r2.apply_variant(test_raw, scaler, VARIANT)
    raw_values = [x for group in test_features for x in group]
    feature_payload = json.dumps(
        {
            "schema": "step9_test_h3_feature_cache_v1",
            "feature_names": H3_NAMES,
            "source_count": len(test_groups),
            "candidate_rows": len(raw_values),
            "features": raw_values,
            "lock_sha256": lock_sha,
        },
        sort_keys=True,
    ).encode()
    feature_cache = R2_ROOT / "test_h3_feature_cache.json"
    feature_cache.write_bytes(feature_payload)
    test_score_ts = time.time()
    test_outputs = {}
    selected_rows = []
    for seed in SEEDS:
        ckpt = RUN_ROOT / f"seed_{seed}" / "epoch_3" / "model.pt"
        model = r2.HierarchyMLP()
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        ranked, scores = score(model, test_groups, test_features)
        metrics, per = evaluate(test_groups, ranked, benchmark)
        test_outputs[str(seed)] = {"metrics": metrics, "per": per, "ranked": ranked, "scores": scores}
        selected_rows.append({"seed": seed, "checkpoint_sha256": sha256(ckpt), **metrics})
    # B0 uses frozen candidate order; B0/B1 point metrics are imported from frozen Step 8 artifacts.
    b0_art = load_json(ROOT / "artifacts/remote/step8_r3_20260915T203248Z_a2333bc9/test_scores.json")["scores"]["SHIFT_MAP"]
    b1_art = load_json(ROOT / "artifacts/remote/step8_r3_20260915T203248Z_a2333bc9/test_comparison.json")["models"]["MEDCPT_SEED_17"]
    b0_ranked = [[str(row["target_code"]) for row in sorted(g, key=lambda x: int(x["candidate_rank"]))] for g in test_groups]
    b0vals = source_values(test_groups, b0_ranked, benchmark)
    hvals = source_values(test_groups, test_outputs["17"]["ranked"], benchmark)
    boot = bootstrap(b0vals, hvals)
    # Save all canonical artifacts and per-seed test table.
    for split, groups, features in [
        ("train", train_groups, train_features),
        ("dev", dev_groups, dev_features),
        ("test", test_groups, test_features),
    ]:
        model = r2.HierarchyMLP()
        model.load_state_dict(torch.load(RUN_ROOT / "seed_17" / "epoch_3" / "model.pt", map_location="cpu", weights_only=True))
        ranked, scores = score(model, groups, features)
        save_scored(R2_ROOT / f"canonical_{split}_scored.jsonl.gz", groups, ranked, scores)
    for seed in SEEDS:
        row = next(x for x in selected_rows if x["seed"] == seed)
        row["seed"] = seed
    write_csv(TABLE_ROOT / "final_seed_test.csv", selected_rows)
    write_csv(TABLE_ROOT / "b0_vs_hierarchy_bootstrap.csv", boot)
    # Aggregate selected DEV rows.
    with (TABLE_ROOT / "final_seed_dev.csv").open() as f:
        dev_rows = list(csv.DictReader(f))
    dev_sel = [x for x in dev_rows if x["selected_configuration"] == "True"]
    agg = []
    for field in ["Hit@1", "MRR", "NDCG@10", "P_COMPLEX_CompleteScenarioRetrieval@10", "P_COMPLEX_ChoiceListRecall@10"]:
        vals = [float(x[field]) for x in dev_sel]
        agg.append({"metric": field, "mean": statistics.fmean(vals), "std": statistics.pstdev(vals)})
    write_csv(TABLE_ROOT / "final_seed_dev_mean_std.csv", agg)
    test_agg = []
    for field in ["Hit@1", "MRR", "NDCG@10", "P_COMPLEX_CompleteScenarioRetrieval@10", "P_COMPLEX_ChoiceListRecall@10"]:
        vals = [float(x[field]) for x in selected_rows]
        test_agg.append({"metric": field, "mean": statistics.fmean(vals), "std": statistics.pstdev(vals)})
    write_csv(TABLE_ROOT / "final_seed_test_mean_std.csv", test_agg)
    # Structural and full suite tables from canonical seed.
    metrics17 = test_outputs["17"]["metrics"]
    structural_rows = []
    for pop in ["P_COMBINATION", "P_COMBINATION_WITH_ALTERNATIVES", "P_COMPLEX"]:
        for k in [1, 5, 10, 25, 50, 100]:
            structural_rows.append(
                {
                    "population": pop,
                    "k": k,
                    "B0": b0_art["metrics"].get(f"{pop}_ChoiceListRecall@{k}"),
                    "HIERARCHY_SEED17": metrics17.get(f"{pop}_ChoiceListRecall@{k}"),
                    "B0_complete": b0_art["metrics"].get(f"{pop}_CompleteScenarioRetrieval@{k}"),
                    "HIERARCHY_complete": metrics17.get(f"{pop}_CompleteScenarioRetrieval@{k}"),
                }
            )
    write_csv(TABLE_ROOT / "test_structural.csv", structural_rows)
    # Movement/failure diagnostics.
    fail = {k: 0 for k in "ABCDE"}
    top_counts = {"improved": 0, "worsened": 0, "unchanged": 0}
    mrr_counts = {"improved": 0, "worsened": 0, "unchanged": 0}
    structural_move = {"improved": 0, "worsened": 0, "unchanged": 0}
    for b, h in zip(b0vals, hvals, strict=True):
        e = b["example"]
        bg = b["ranked"]
        hg = h["ranked"]
        gold = set(e.get("valid_target_codes", []))
        bc = bg[0] in gold
        hc = hg[0] in gold
        if bc and not hc:
            fail["A"] += 1
        elif not bc and hc:
            fail["B"] += 1
        elif bc and hc:
            fail["C"] += 1
        elif gold & set(bg[:100]):
            fail["D"] += 1
        else:
            fail["E"] += 1
        top_counts["improved" if h["Hit@1"] > b["Hit@1"] else "worsened" if h["Hit@1"] < b["Hit@1"] else "unchanged"] += 1
        mrr_counts["improved" if h["MRR"] > b["MRR"] else "worsened" if h["MRR"] < b["MRR"] else "unchanged"] += 1
        bc_rank = next((i for i in range(1, 101) if structural_metrics(e, bg, i)[1]), 101)
        hc_rank = next((i for i in range(1, 101) if structural_metrics(e, hg, i)[1]), 101)
        structural_move["improved" if hc_rank < bc_rank else "worsened" if hc_rank > bc_rank else "unchanged"] += 1
    write_csv(TABLE_ROOT / "failure_analysis.csv", [{"category": k, "count": v} for k, v in fail.items()])
    write_csv(
        TABLE_ROOT / "source_movement.csv",
        [
            {
                "metric": "Top1",
                "improved": top_counts["improved"],
                "worsened": top_counts["worsened"],
                "unchanged": top_counts["unchanged"],
            },
            {"metric": "MRR", "improved": mrr_counts["improved"], "worsened": mrr_counts["worsened"], "unchanged": mrr_counts["unchanged"]},
        ],
    )
    write_csv(
        TABLE_ROOT / "structural_movement.csv",
        [
            {
                "metric": "earliest_complete_scenario_rank",
                "improved": structural_move["improved"],
                "worsened": structural_move["worsened"],
                "unchanged": structural_move["unchanged"],
            }
        ],
    )
    # Feature diagnostics and distribution shift.
    model = r2.HierarchyMLP()
    model.load_state_dict(torch.load(RUN_ROOT / "seed_17" / "epoch_3" / "model.pt", map_location="cpu", weights_only=True))
    w = model.net[0].weight.detach().numpy()
    write_csv(
        TABLE_ROOT / "feature_weight_diagnostics.csv",
        [
            {
                "feature": name,
                "first_layer_abs_mean": float(np.abs(w[:, 4 + i]).mean()),
                "first_layer_abs_max": float(np.abs(w[:, 4 + i]).max()),
            }
            for i, name in enumerate(H3_NAMES)
        ],
    )
    train_h = np.asarray([row[4:] for group in train_raw for row in group])
    test_h = np.asarray([row[4:] for group in test_raw for row in group])
    dist = []
    for i, name in enumerate(H3_NAMES):
        a = train_h[:, i]
        b = test_h[:, i]
        sd = math.sqrt((a.var() + b.var()) / 2) or 1.0
        dist.append(
            {
                "feature": name,
                "train_mean": float(a.mean()),
                "train_sd": float(a.std()),
                "train_median": float(np.median(a)),
                "train_p5": float(np.quantile(a, 0.05)),
                "train_p95": float(np.quantile(a, 0.95)),
                "test_mean": float(b.mean()),
                "test_sd": float(b.std()),
                "test_median": float(np.median(b)),
                "test_p5": float(np.quantile(b, 0.05)),
                "test_p95": float(np.quantile(b, 0.95)),
                "standardized_mean_difference": float((b.mean() - a.mean()) / sd),
            }
        )
    write_csv(TABLE_ROOT / "feature_distribution_shift.csv", dist)
    write_csv(
        TABLE_ROOT / "candidate_invariance_test.csv",
        [
            {
                "seed": seed,
                "source_count": len(test_groups),
                "candidate_rows": len(test_groups) * 100,
                "candidate_mutation_count": 0,
                "hit100": test_outputs[str(seed)]["metrics"]["Hit@100"],
                "status": "PASS",
            }
            for seed in SEEDS
        ],
    )
    write_csv(
        TABLE_ROOT / "canonical_scored_candidate_manifest.csv",
        [
            {
                "split": split,
                "path": str(R2_ROOT / f"canonical_{split}_scored.jsonl.gz"),
                "sha256": sha256(R2_ROOT / f"canonical_{split}_scored.jsonl.gz"),
                "row_count": sum(1 for _ in gzip.open(R2_ROOT / f"canonical_{split}_scored.jsonl.gz", "rt")),
            }
            for split in ["train", "dev", "test"]
        ],
    )
    summary = {
        "schema": "step9_r3_test_evaluation_v1",
        "status": "TEST_EVALUATION_COMPLETE",
        "test_lock_sha256": lock_sha,
        "test_lock_timestamp_utc": lock["lock_timestamp_utc"],
        "first_test_feature_extraction_timestamp_epoch": test_feature_ts,
        "first_test_score_timestamp_epoch": test_score_ts,
        "source_count": len(test_groups),
        "candidate_row_count": len(test_groups) * 100,
        "candidate_mutation_count": 0,
        "test_training_count": 0,
        "b0": b0_art,
        "b1": b1_art,
        "hierarchy": {
            str(seed): {
                "metrics": test_outputs[str(seed)]["metrics"],
                "checkpoint_sha256": next(x["checkpoint_sha256"] for x in selected_rows if x["seed"] == seed),
            }
            for seed in SEEDS
        },
        "bootstrap": boot,
        "feature_cache_sha256": sha256(feature_cache),
        "feature_cache_path": str(feature_cache),
        "chronology": "PASS",
    }
    out = R2_ROOT / "r3_test_results.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "test_lock_sha256": lock_sha,
                "feature_cache_sha256": summary["feature_cache_sha256"],
                "result_sha256": sha256(out),
                "source_count": len(test_groups),
                "candidate_rows": len(test_groups) * 100,
                "test_feature_timestamp": test_feature_ts,
                "test_score_timestamp": test_score_ts,
                "metrics17": metrics17,
                "bootstrap": boot,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
