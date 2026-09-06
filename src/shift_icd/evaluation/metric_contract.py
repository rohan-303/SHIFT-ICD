# ruff: noqa: E501
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

ORDINARY_KINDS = {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"}
COMPLEX_KINDS = {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES"}


def classify_population(example: Any) -> str:
    kind = str(example.mapping_kind)
    if bool(example.no_map) or kind == "NO_MAP":
        return "P_NO_MAP"
    if kind == "COMBINATION":
        return "P_COMBINATION"
    if kind == "COMBINATION_WITH_ALTERNATIVES":
        return "P_COMBINATION_WITH_ALTERNATIVES"
    if kind in ORDINARY_KINDS:
        return "P_ORDINARY_ANSWERABLE"
    if kind == "MULTI_SCENARIO":
        return "P_COMPLEX"
    raise ValueError(f"unsupported mapping kind: {kind}")


def population_counts(examples: Iterable[object]) -> dict[str, int]:
    counts = Counter(classify_population(example) for example in examples)
    counts["P_ALL"] = sum(counts.values())
    counts["P_COMPLEX"] = counts["P_COMBINATION"] + counts["P_COMBINATION_WITH_ALTERNATIVES"] + counts.get("P_COMPLEX", 0)
    return {key: counts.get(key, 0) for key in ("P_ALL", "P_ORDINARY_ANSWERABLE", "P_COMBINATION", "P_COMBINATION_WITH_ALTERNATIVES", "P_COMPLEX", "P_NO_MAP")}


def metric_values(valid_codes: set[str], ranked_codes: Sequence[str], *, no_map: bool, k_values: Sequence[int]) -> dict[str, float | None]:
    if no_map:
        return {**{f"Hit@{k}": None for k in k_values}, "MRR": None}
    ranks = [rank for rank, code in enumerate(ranked_codes, 1) if code in valid_codes]
    best = min(ranks, default=None)
    return {**{f"Hit@{k}": float(best is not None and best <= k) for k in k_values}, "MRR": 0.0 if best is None else 1.0 / best}
