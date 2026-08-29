from __future__ import annotations

from collections.abc import Iterable

from shift_icd.data.schemas import GemChoiceList, GemSourceMapping


def _covered_lists(choice_lists: Iterable[GemChoiceList], predicted: set[str]) -> int:
    return sum(any(a.target_code in predicted for a in c.alternatives) for c in choice_lists)


def choice_list_coverage(mapping: GemSourceMapping, predicted_codes: set[str]) -> float:
    lists = [c for s in mapping.scenarios for c in s.choice_lists]
    if not lists:
        if mapping.no_map:
            return 0.0
        targets = {row for row in predicted_codes}
        return 1.0 if targets & _mapping_targets(mapping) else 0.0
    return sum(_covered_lists(s.choice_lists, predicted_codes) for s in mapping.scenarios) / sum(
        len(s.choice_lists) for s in mapping.scenarios
    )


def _mapping_targets(mapping: GemSourceMapping) -> set[str]:
    return {a.target_code for s in mapping.scenarios for c in s.choice_lists for a in c.alternatives}


def is_valid_complete_mapping(mapping: GemSourceMapping, predicted_codes: set[str]) -> bool:
    if mapping.no_map:
        return False
    if not mapping.scenarios:
        return bool(predicted_codes & _mapping_targets(mapping))
    return best_matching_scenario(mapping, predicted_codes) is not None


def best_matching_scenario(mapping: GemSourceMapping, predicted_codes: set[str]) -> int | None:
    for scenario in mapping.scenarios:
        if len(predicted_codes) == 0:
            continue
        if all(any(a.target_code in predicted_codes for a in choice.alternatives) for choice in scenario.choice_lists):
            return scenario.scenario_id
    return None


def mapping_completion_fraction(mapping: GemSourceMapping, predicted_codes: set[str]) -> float:
    if mapping.no_map:
        return 0.0
    if not mapping.scenarios:
        return 1.0 if is_valid_complete_mapping(mapping, predicted_codes) else 0.0
    return max(
        (_covered_lists(s.choice_lists, predicted_codes) / len(s.choice_lists) for s in mapping.scenarios),
        default=0.0,
    )


def component_precision(mapping: GemSourceMapping, predicted_codes: set[str]) -> float:
    if not predicted_codes:
        return 0.0
    valid = _mapping_targets(mapping)
    return len(predicted_codes & valid) / len(predicted_codes)


def component_recall(mapping: GemSourceMapping, predicted_codes: set[str]) -> float:
    if mapping.no_map:
        return 0.0
    if not mapping.scenarios:
        return float(bool(predicted_codes & _mapping_targets(mapping)))
    return mapping_completion_fraction(mapping, predicted_codes)


def component_f1(mapping: GemSourceMapping, predicted_codes: set[str]) -> float:
    precision = component_precision(mapping, predicted_codes)
    recall = component_recall(mapping, predicted_codes)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def exact_mapping_match(mapping: GemSourceMapping, predicted_codes: set[str]) -> bool:
    if mapping.no_map or not mapping.scenarios:
        return is_valid_complete_mapping(mapping, predicted_codes) and len(predicted_codes) == 1
    for scenario in mapping.scenarios:
        selected: set[str] = set()
        valid = True
        for choice in scenario.choice_lists:
            matches = {a.target_code for a in choice.alternatives if a.target_code in predicted_codes}
            if len(matches) != 1:
                valid = False
                break
            selected.update(matches)
        if valid and selected == predicted_codes:
            return True
    return False


def valid_top1_target(mapping: GemSourceMapping, predicted_code: str | None) -> bool:
    return predicted_code is not None and is_valid_complete_mapping(mapping, {predicted_code})
