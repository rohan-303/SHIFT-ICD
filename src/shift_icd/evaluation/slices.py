from __future__ import annotations

from typing import cast


def alternative_size_bucket(unique_target_count: int, mapping_kind: str) -> str | None:
    if mapping_kind != "ALTERNATIVE":
        return None
    if unique_target_count <= 5:
        return "ALTERNATIVE_SMALL"
    if unique_target_count <= 20:
        return "ALTERNATIVE_MEDIUM"
    if unique_target_count <= 100:
        return "ALTERNATIVE_LARGE"
    return "ALTERNATIVE_EXTREME"


def assign_slices(metadata: dict[str, object]) -> dict[str, bool]:
    kind = str(metadata["mapping_kind"])
    targets = int(cast(int, metadata["unique_target_count"]))
    scenarios = int(cast(int, metadata["scenario_count"]))
    lists = int(cast(int, metadata["choice_list_count"]))
    valid_sets = int(cast(int, metadata["valid_mapping_set_count"]))
    alt_bucket = alternative_size_bucket(targets, kind)
    low = scenarios <= 1 and lists <= 2 and targets <= 5 and valid_sets <= 5
    return {
        "SINGLE_EXACT": kind == "SINGLE_EXACT",
        "SINGLE_APPROXIMATE": kind == "SINGLE_APPROXIMATE",
        "NO_MAP": kind == "NO_MAP",
        "ALTERNATIVE_SMALL": alt_bucket == "ALTERNATIVE_SMALL",
        "ALTERNATIVE_MEDIUM": alt_bucket == "ALTERNATIVE_MEDIUM",
        "ALTERNATIVE_LARGE": alt_bucket == "ALTERNATIVE_LARGE",
        "ALTERNATIVE_EXTREME": alt_bucket == "ALTERNATIVE_EXTREME",
        "COMBINATION": bool(metadata.get("combination", kind.startswith("COMBINATION"))),
        "COMBINATION_WITH_ALTERNATIVES": kind == "COMBINATION_WITH_ALTERNATIVES",
        "MULTI_SCENARIO": scenarios > 1,
        "LOW_MAPPING_COMPLEXITY": low,
        "HIGH_MAPPING_COMPLEXITY": not low,
    }
