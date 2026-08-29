from __future__ import annotations

from collections.abc import Iterable

from shift_icd.data.schemas import GemSourceMapping


def choice_list_recall_at_k(mapping: GemSourceMapping, ranked_codes: Iterable[str], k: int) -> float | None:
    if mapping.no_map:
        return None
    retrieved = set(list(ranked_codes)[:k])
    if not mapping.scenarios:
        targets = {str(code) for code in mapping.model_dump().get("valid_target_codes", [])}
        return float(bool(retrieved & targets)) if targets else 0.0
    covered = sum(
        sum(any(a.target_code in retrieved for a in choice.alternatives) for choice in scenario.choice_lists)
        for scenario in mapping.scenarios
    )
    total = sum(len(scenario.choice_lists) for scenario in mapping.scenarios)
    return covered / total if total else 0.0


def complete_scenario_retrieval_at_k(mapping: GemSourceMapping, ranked_codes: Iterable[str], k: int) -> bool | None:
    if mapping.no_map:
        return None
    retrieved = set(list(ranked_codes)[:k])
    if not mapping.scenarios:
        targets = set(mapping.model_dump().get("valid_target_codes", []))
        return bool(retrieved & targets)
    return any(
        all(any(a.target_code in retrieved for a in choice.alternatives) for choice in scenario.choice_lists)
        for scenario in mapping.scenarios
    )


def hit_at_k(valid_codes: set[str], ranked_codes: Iterable[str], k: int) -> bool:
    return bool(valid_codes.intersection(list(ranked_codes)[:k]))
