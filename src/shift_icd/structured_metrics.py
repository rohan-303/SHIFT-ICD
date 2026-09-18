# ruff: noqa: E501
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _form(structure: Mapping[str, Any]) -> str:
    return str(structure.get("mapping_form", "NO_MAP"))


def _scenarios(structure: Mapping[str, Any]) -> tuple[tuple[tuple[str, ...], ...], ...]:
    result = []
    for scenario in structure.get("scenarios", []):
        choices = tuple(sorted(tuple(sorted(set(map(str, choice.get("alternatives", []))))) for choice in scenario.get("choice_lists", [])))
        result.append(choices)
    return tuple(sorted(result))


def _flat(structure: Mapping[str, Any]) -> frozenset[str]:
    values: set[str] = set(map(str, structure.get("flat_alternatives", [])))
    for scenario in structure.get("scenarios", []):
        for choice in scenario.get("choice_lists", []):
            values.update(map(str, choice.get("alternatives", [])))
    return frozenset(values)


def canonical_signature(structure: Mapping[str, Any]) -> tuple[Any, ...]:
    form = _form(structure)
    return (form, _scenarios(structure), tuple(sorted(_flat(structure))) if not _scenarios(structure) else ())


def exact_canonical_structure_match(predicted: Mapping[str, Any], gold: Mapping[str, Any]) -> bool:
    return canonical_signature(predicted) == canonical_signature(gold)


def flat_set_metrics(predicted: Mapping[str, Any], gold: Mapping[str, Any]) -> dict[str, float]:
    prediction = _flat(predicted)
    target = _flat(gold)
    if not prediction and not target:
        precision = recall = f1 = 1.0
    elif not prediction:
        precision = f1 = 0.0
        recall = 0.0
    elif not target:
        precision = recall = f1 = 0.0
    else:
        precision = len(prediction & target) / len(prediction)
        recall = len(prediction & target) / len(target)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def complete_scenario_success(predicted: Mapping[str, Any], gold: Mapping[str, Any]) -> bool:
    gold_scenarios = _scenarios(gold)
    if not gold_scenarios:
        return False
    predicted_scenarios = _scenarios(predicted)
    for gold_scenario in gold_scenarios:
        if any(all(any(set(pred_choice) & set(gold_choice) for pred_choice in pred_scenario) for gold_choice in gold_scenario) for pred_scenario in predicted_scenarios):
            return True
    return False


def structure_validity(predicted: Mapping[str, Any], candidate_ids: Iterable[str]) -> bool:
    candidates = set(map(str, candidate_ids))
    form = _form(predicted)
    if form == "NO_MAP":
        return not _flat(predicted) and not _scenarios(predicted)
    if form not in {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE", "COMBINATION", "COMBINATION_WITH_ALTERNATIVES"}:
        return False
    if not _flat(predicted) and not _scenarios(predicted):
        return False
    if not _flat(predicted) <= candidates:
        return False
    for scenario in predicted.get("scenarios", []):
        choices = scenario.get("choice_lists", [])
        if not choices or len(choices) > 3:
            return False
        for choice in choices:
            alternatives = list(map(str, choice.get("alternatives", [])))
            if not alternatives or len(alternatives) != len(set(alternatives)) or not set(alternatives) <= candidates:
                return False
    return True


def classify_retrieval_failure(exact_match: bool, fully_representable: bool) -> str | None:
    if exact_match:
        return None
    return "DECODER_LIMITED" if fully_representable else "RETRIEVAL_LIMITED"


def selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["exact_canonical_structure_match"]),
        -float(row["complete_scenario_success"]),
        -float(row["mapping_form_macro_f1"]),
        -float(row["no_map_f1"]),
        -float(row["flat_set_f1"]),
        -float(row["cardinality_exact_accuracy"]),
        str(row["configuration_id"]),
    )


def assert_official_dev_access_forbidden() -> None:
    raise RuntimeError("STEP11_OFFICIAL_DEV_ACCESS_FORBIDDEN")


def assert_test_access_forbidden() -> None:
    raise RuntimeError("STEP11_TEST_ACCESS_FORBIDDEN")


def verify_feature_provenance(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for field in ("feature_dimension", "canonical_checkpoint_sha256", "candidate_artifacts"):
        if actual.get(field) != expected.get(field):
            raise RuntimeError("STEP11_FEATURE_PROVENANCE_MISMATCH")
    if actual.get("normalized_retriever_score") != expected.get("normalized_retriever_score"):
        raise RuntimeError("STEP11_FEATURE_PROVENANCE_MISMATCH")


def epoch_selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (*selection_key(row)[:-1], int(row["epoch"]))


def choose_best_epoch(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("no epoch rows")
    return min(rows, key=epoch_selection_key)


def choose_best(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("no selection rows")
    return min(rows, key=selection_key)
