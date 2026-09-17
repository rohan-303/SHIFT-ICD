# mypy: ignore-errors
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / "artifacts/experiments/step9_hierarchy"
TABLES = ROOT / "reports/tables/step9_hierarchy"
CANDIDATE = ROOT / "artifacts/candidates/shift_map_full_universe_v2/forward_test_k100.jsonl.gz"
BENCHMARK = ROOT / "data/benchmarks/cms_track_a/v1.0/forward/all.jsonl"


def sha_ids(ids: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(ids))).encode()).hexdigest()


def structural(example: dict[str, Any], ranked: list[str], k: int) -> tuple[float, float]:
    retrieved = set(ranked[:k])
    scenarios = example.get("scenarios", [])
    if not scenarios:
        hit = float(bool(retrieved & set(example.get("valid_target_codes", []))))
        return hit, hit
    total = sum(len(s["choice_lists"]) for s in scenarios)
    covered = sum(
        sum(any(a["target_code"] in retrieved for a in choice["alternatives"]) for choice in s["choice_lists"]) for s in scenarios
    )
    complete = any(
        all(any(a["target_code"] in retrieved for a in choice["alternatives"]) for choice in s["choice_lists"]) for s in scenarios
    )
    return (covered / total if total else 0.0), float(complete)


def rank_metrics(example: dict[str, Any], ranked: list[str]) -> dict[str, float]:
    gold = set(example.get("valid_target_codes", []))
    ranks = [i for i, code in enumerate(ranked, 1) if code in gold]
    first = min(ranks, default=101)
    out = {f"Hit@{k}": float(first <= k) for k in (1, 5, 10, 25, 50, 100)}
    out["MRR"] = 1.0 / first if first <= 100 else 0.0
    gains = [1.0 if code in gold else 0.0 for code in ranked[:10]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    out["NDCG@10"] = dcg / idcg if idcg else 0.0
    return out


def load() -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    benchmark = {}
    for line in BENCHMARK.open(encoding="utf-8"):
        row = json.loads(line)
        if row["split"] == "test":
            benchmark[str(row["benchmark_id"])] = row
    groups: dict[str, list[dict[str, Any]]] = {}
    with gzip.open(CANDIDATE, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            groups.setdefault(str(row["source_id"]), []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda x: int(x["candidate_rank"]))
    return benchmark, groups


def scored(path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[tuple[int, str]]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            groups.setdefault(str(row["source_id"]), []).append((int(row["rerank_rank"]), str(row["target_code"])))
    return {source: [code for _, code in sorted(rows)] for source, rows in groups.items()}


def summarize(benchmark, ranked, source_ids, label):
    rows = []
    for sid in sorted(source_ids):
        ex = benchmark[sid]
        r = ranked[sid]
        c = {k: structural(ex, r, k) for k in (1, 5, 10, 25, 50, 100)}
        rows.append({"source_id": sid, "choice": c, "ordinary": rank_metrics(ex, r)})
    return rows


def mean(rows, key, k):
    return float(np.mean([row["choice"][k][0 if key == "ChoiceListRecall" else 1] for row in rows]))


def bootstrap(b0, h):
    rng = np.random.default_rng(20260915)
    out = []
    for key, idx in (("Choice@10", 0), ("CSR@10", 1)):
        a = np.asarray([row["choice"][10][idx] for row in b0], dtype=float)
        b = np.asarray([row["choice"][10][idx] for row in h], dtype=float)
        draws = np.empty(10000)
        for i in range(10000):
            ix = rng.integers(0, len(a), size=len(a))
            draws[i] = float(np.mean(b[ix] - a[ix]))
        out.append(
            {
                "metric": key,
                "population_n": len(a),
                "delta": float(np.mean(b - a)),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "repetitions": 10000,
                "seed": 20260915,
            }
        )
    return out


def ordinary_bootstrap(b0, h):
    rng = np.random.default_rng(20260915)
    out = []
    for key in ("Hit@1", "MRR", "NDCG@10"):
        a = np.asarray([row["ordinary"][key] for row in b0], dtype=float)
        b = np.asarray([row["ordinary"][key] for row in h], dtype=float)
        draws = np.empty(10000)
        for i in range(10000):
            ix = rng.integers(0, len(a), size=len(a))
            draws[i] = float(np.mean(b[ix] - a[ix]))
        out.append(
            {
                "metric": key,
                "population_n": len(a),
                "delta": float(np.mean(b - a)),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "repetitions": 10000,
                "seed": 20260915,
            }
        )
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    benchmark, groups = load()
    by_kind = {
        kind: {sid for sid, ex in benchmark.items() if ex["mapping_kind"] == kind}
        for kind in ("COMBINATION", "COMBINATION_WITH_ALTERNATIVES")
    }
    canonical = by_kind["COMBINATION"] | by_kind["COMBINATION_WITH_ALTERNATIVES"]
    r3 = {sid for sid, ex in benchmark.items() if "HIGH_MAPPING_COMPLEXITY" in ex.get("difficulty_slices", [])}
    ordinary = {
        sid
        for sid, ex in benchmark.items()
        if ex["mapping_kind"] in {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"} and ex.get("valid_target_codes")
    }
    b0 = {sid: [str(row["target_code"]) for row in groups[sid]] for sid in groups}
    h17 = scored(R2 / "canonical_test_scored.jsonl.gz")
    # The canonical scored artifact is seed17; verify complete source coverage.
    assert set(h17) == set(groups)
    b0_rows = summarize(benchmark, b0, canonical, "B0")
    h_rows = summarize(benchmark, h17, canonical, "H17")
    b0_ord = summarize(benchmark, b0, ordinary, "B0")
    h17_ord = summarize(benchmark, h17, ordinary, "H17")
    r3_rows = summarize(benchmark, b0, r3, "R3")
    populations = {
        "P_COMBINATION": {"count": len(by_kind["COMBINATION"]), "source_set_sha256": sha_ids(by_kind["COMBINATION"])},
        "P_COMBINATION_WITH_ALTERNATIVES": {
            "count": len(by_kind["COMBINATION_WITH_ALTERNATIVES"]),
            "source_set_sha256": sha_ids(by_kind["COMBINATION_WITH_ALTERNATIVES"]),
        },
        "P_COMPLEX": {"count": len(canonical), "source_set_sha256": sha_ids(canonical)},
        "R3_HIGH_MAPPING_COMPLEXITY": {"count": len(r3), "source_set_sha256": sha_ids(r3)},
        "P_ORDINARY_ANSWERABLE": {"count": len(ordinary), "source_set_sha256": sha_ids(ordinary)},
    }
    corrected = []
    for population, rows in (("P_COMPLEX", b0_rows), ("P_COMPLEX", h_rows)):
        system = "B0" if rows is b0_rows else "HIERARCHY_SEED17"
        for k in (1, 5, 10, 25, 50, 100):
            corrected.append(
                {
                    "population": population,
                    "system": system,
                    "k": k,
                    "ChoiceListRecall": mean(rows, "ChoiceListRecall", k),
                    "CompleteScenarioRetrieval": mean(rows, "CompleteScenarioRetrieval", k),
                    "n": len(rows),
                }
            )
    boot = bootstrap(b0_rows, h_rows)
    ordinary_boot = ordinary_bootstrap(b0_ord, h17_ord)
    ordinary_metrics = {
        k: float(np.mean([row["ordinary"][k] for row in b0_ord]))
        for k in ("Hit@1", "Hit@5", "Hit@10", "Hit@25", "Hit@50", "Hit@100", "MRR", "NDCG@10")
    }
    corrected_ordinary = []
    for system, rows in (("B0", b0_ord), ("HIERARCHY_SEED17", h17_ord)):
        for k in ("Hit@1", "Hit@5", "Hit@10", "Hit@25", "Hit@50", "Hit@100", "MRR", "NDCG@10"):
            corrected_ordinary.append(
                {
                    "system": system,
                    "metric": k,
                    "value": float(np.mean([row["ordinary"][k] for row in rows])),
                    "n": len(rows),
                    "mrr_missing_gold_policy": "zero",
                }
            )
    write_csv(TABLES / "r3a_ordinary_metrics_corrected.csv", corrected_ordinary)
    write_csv(TABLES / "r3a_primary_ordinary_bootstrap.csv", ordinary_boot)
    write_csv(TABLES / "r3a_structural_metrics_corrected.csv", corrected)
    write_csv(TABLES / "r3a_b0_vs_hierarchy_structural_bootstrap.csv", boot)
    write_csv(
        TABLES / "r3a_population_set_comparison.csv",
        [
            {"set": "canonical_P_COMPLEX", "count": len(canonical), "source_set_sha256": sha_ids(canonical)},
            {"set": "original_R3_HIGH_MAPPING_COMPLEXITY", "count": len(r3), "source_set_sha256": sha_ids(r3)},
            {"set": "intersection", "count": len(canonical & r3)},
            {"set": "canonical_only", "count": len(canonical - r3)},
            {"set": "r3_only", "count": len(r3 - canonical)},
        ],
    )
    write_csv(
        TABLES / "r3a_evaluator_comparison.csv",
        [
            {
                "implementation": "src/shift_icd/evaluation/metric_contract.py",
                "P_COMPLEX_definition": "COMBINATION union COMBINATION_WITH_ALTERNATIVES union MULTI_SCENARIO",
                "conditioning_on_gold_presence": False,
                "conditioning_on_complete_scenario_presence": False,
            },
            {
                "implementation": "scripts/step9_hierarchy/r2_runner.py",
                "P_COMPLEX_definition": "HIGH_MAPPING_COMPLEXITY difficulty slice",
                "conditioning_on_gold_presence": False,
                "conditioning_on_complete_scenario_presence": False,
            },
        ],
    )
    audit = {
        "schema": "step9_r3a_structural_population_audit_v1",
        "status": "CORRECTION_REQUIRED",
        "benchmark_path": str(BENCHMARK),
        "candidate_path": str(CANDIDATE),
        "candidate_sha256": "6ef38443a201b1d630148b30717d845b0785b1a4ed433eeab4b03d893b6bf1ce",
        "authoritative_evaluator": "src/shift_icd/evaluation/metric_contract.py + src/shift_icd/evaluation/retrieval.py",
        "populations": populations,
        "r3_original_population": {
            "count": len(r3),
            "source_set_sha256": sha_ids(r3),
            "predicate": "HIGH_MAPPING_COMPLEXITY in difficulty_slices",
            "intersection_with_canonical": len(canonical & r3),
            "canonical_only": len(canonical - r3),
            "r3_only": len(r3 - canonical),
        },
        "root_cause_classification": ["SP4_MAPPING_SUBTYPE_FILTER", "SP5_EVALUATOR_IMPLEMENTATION_DRIFT"],
        "conditioned_on_gold_presence": False,
        "conditioned_on_complete_scenario_presence": False,
        "canonical_definition": "P_COMBINATION union P_COMBINATION_WITH_ALTERNATIVES; MULTI_SCENARIO absent from forward TEST",
        "ordinary_population_metrics": ordinary_metrics,
        "ordinary_corrected_hierarchy_metrics": {
            k: float(np.mean([row["ordinary"][k] for row in h17_ord]))
            for k in ("Hit@1", "Hit@5", "Hit@10", "Hit@25", "Hit@50", "Hit@100", "MRR", "NDCG@10")
        },
        "ordinary_corrected_bootstrap": ordinary_boot,
        "canonical_B0_at100": {
            "ChoiceListRecall": mean(b0_rows, "ChoiceListRecall", 100),
            "CompleteScenarioRetrieval": mean(b0_rows, "CompleteScenarioRetrieval", 100),
        },
        "r3_B0_at100": {
            "ChoiceListRecall": mean(r3_rows, "ChoiceListRecall", 100),
            "CompleteScenarioRetrieval": mean(r3_rows, "CompleteScenarioRetrieval", 100),
        },
        "corrected_bootstrap": boot,
        "no_model_retraining": True,
        "no_model_selection": True,
    }
    audit_path = R2 / "structural_population_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "audit_path": str(audit_path),
                "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                "populations": populations,
                "canonical_b0_at100": audit["canonical_B0_at100"],
                "r3_b0_at100": audit["r3_B0_at100"],
                "bootstrap": boot,
                "ordinary": ordinary_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
