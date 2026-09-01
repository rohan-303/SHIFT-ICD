# ruff: noqa
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def group_by_cardinality(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {"G=1": [], "G=2-5": [], "G=6-20": [], "G=21-100": [], "G>100": []}
    for row in rows:
        n = int(row.get("n", 0))
        key = "G=1" if n == 1 else "G=2-5" if n <= 5 else "G=6-20" if n <= 20 else "G=21-100" if n <= 100 else "G>100"
        groups[key].append(row)
    return groups


def paired_counts(l2: np.ndarray, l1: np.ndarray) -> dict[str, int]:
    return {
        "l2_success": int(np.sum((l2 == 1) & (l1 == 0))),
        "l1_success": int(np.sum((l1 == 1) & (l2 == 0))),
        "both_success": int(np.sum((l2 == 1) & (l1 == 1))),
        "both_fail": int(np.sum((l2 == 0) & (l1 == 0))),
    }


def bootstrap_delta(l2: np.ndarray, l1: np.ndarray, seed: int = 2026, reps: int = 5000) -> dict[str, float | int]:
    delta = np.asarray(l2, dtype=float) - np.asarray(l1, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.array([np.mean(rng.choice(delta, len(delta), replace=True)) for _ in range(reps)])
    return {"n": int(len(delta)), "delta": float(np.mean(delta)), "ci95_low": float(np.quantile(samples, 0.025)), "ci95_high": float(np.quantile(samples, 0.975)), "seed": seed, "reps": reps}


def summarize_values(values: list[float] | np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, "min": float(np.min(x)), "p1": float(np.quantile(x, .01)), "p5": float(np.quantile(x, .05)), "q1": float(np.quantile(x, .25)), "median": float(np.median(x)), "q3": float(np.quantile(x, .75)), "p95": float(np.quantile(x, .95)), "p99": float(np.quantile(x, .99)), "max": float(np.max(x))}


def gini(values: list[float] | np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0
    return float((2 * np.sum((np.arange(1, len(x) + 1)) * x) / (len(x) * np.sum(x))) - (len(x) + 1) / len(x))


def candidate_coverage(rows: list[dict[str, Any]], k: int) -> float:
    applicable = [r for r in rows if r.get("valid_target_codes")]
    return float(np.mean([bool(set(r["valid_target_codes"]) & set(r.get("ranked_codes", [])[:k])) for r in applicable])) if applicable else 0.0


def grouped(rows: list[dict[str, Any]], field: str) -> dict[Any, list[dict[str, Any]]]:
    result: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[row.get(field)].append(row)
    return dict(result)
