from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any


def extract_source_family(code: str, version: str) -> str:
    compact = code.replace(".", "").strip().upper()
    if version == "ICD-10-CM":
        return compact[:3]
    if compact.startswith("V") or compact.startswith("E"):
        return compact[:3]
    return compact[:3]


def _assignment_order(group: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()


def split_source_objects(
    objects: list[dict[str, Any]],
    seed: int,
    ratios: tuple[float, float, float],
    group_key: str,
) -> dict[str, list[str]]:
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to one")
    groups: dict[str, list[str]] = defaultdict(list)
    for obj in objects:
        groups[str(obj[group_key])].append(str(obj["benchmark_id"]))
    ordered = sorted(groups, key=lambda group: _assignment_order(group, seed))
    total = len(objects)
    targets = [total * ratio for ratio in ratios]
    assigned: dict[str, list[str]] = {name: [] for name in ("train", "dev", "test")}
    counts = [0, 0, 0]
    for group in ordered:
        size = len(groups[group])
        deficits = [targets[i] - counts[i] for i in range(3)]
        index = max(range(3), key=lambda i: deficits[i])
        assigned[("train", "dev", "test")[index]].extend(groups[group])
        counts[index] += size
    return {key: sorted(value) for key, value in assigned.items()}
