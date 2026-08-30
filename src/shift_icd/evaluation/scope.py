from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationScope:
    direction: str
    split_protocol: str
    partition: str

    def filter(self, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if row.get("direction") == self.direction
            and row.get("split_protocol") == self.split_protocol
            and row.get("partition") == self.partition
        ]

    def metadata(self, n: int, answerable: int, applicable: int, sample_population: str) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "split_protocol": self.split_protocol,
            "partition": self.partition,
            "benchmark_version": "1.0",
            "sample_population": sample_population,
            "total_example_count": n,
            "answerable_count": answerable,
            "applicable_metric_count": applicable,
        }


def normalize_code_for_lookup(code: str) -> str:
    return re.sub(r"[^a-z0-9]", "", code.casefold())


def canonical_benchmark_diff(canonical_ids: set[str], benchmark_ids: set[str]) -> dict[str, list[str]]:
    return {
        "canonical_only": sorted(canonical_ids - benchmark_ids),
        "benchmark_only": sorted(benchmark_ids - canonical_ids),
    }


def validate_slice_partition(slice_ids: dict[str, set[str]], population_ids: set[str]) -> list[str]:
    errors: list[str] = []
    union: set[str] = set()
    names = sorted(slice_ids)
    for name in names:
        unknown = slice_ids[name] - population_ids
        if unknown:
            errors.append(f"{name}: IDs outside population: {sorted(unknown)}")
        overlap = union & slice_ids[name]
        if overlap:
            errors.append(f"overlap: {sorted(overlap)}")
        union.update(slice_ids[name])
    if union != population_ids:
        errors.append(f"coverage mismatch: missing={sorted(population_ids - union)} extra={sorted(union - population_ids)}")
    return errors


def random_hit_probability(candidate_count: int, gold_count: int, k: int) -> float:
    if candidate_count <= 0 or gold_count <= 0 or k <= 0:
        return 0.0
    if gold_count >= candidate_count or k >= candidate_count:
        return 1.0
    misses = 1.0
    for i in range(k):
        misses *= (candidate_count - gold_count - i) / (candidate_count - i)
    return min(1.0, max(0.0, 1.0 - misses))
