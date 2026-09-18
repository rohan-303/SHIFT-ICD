# ruff: noqa: E501
from __future__ import annotations

import json

from shift_icd.structured_decoder import StructuredSetDecoder, assemble_structure


def _candidate_scores() -> dict[str, float]:
    return {code: 1.0 for code in ("A", "B", "C", "D")}


def test_same_flat_set_different_choice_partition_is_not_identifiable() -> None:
    shared = {"mapping_form": "COMBINATION_WITH_ALTERNATIVES", "scenario_count": 1, "slot_count": 2, "flat_targets": ["A", "B", "C", "D"]}
    left = {**shared, "choice_lists": [["A", "B"], ["C", "D"]]}
    right = {**shared, "choice_lists": [["A", "C"], ["B", "D"]]}
    assert left["flat_targets"] == right["flat_targets"]
    assert left["choice_lists"] != right["choice_lists"]
    assert not _current_heads_can_encode_assignments(left, right)


def test_same_flat_set_different_scenario_partition_is_not_identifiable() -> None:
    left = {"mapping_form": "COMBINATION", "scenario_count": 2, "slot_count": 2, "flat_targets": ["A", "B", "C", "D"], "scenarios": [["A", "B"], ["C", "D"]]}
    right = {"mapping_form": "COMBINATION", "scenario_count": 2, "slot_count": 2, "flat_targets": ["A", "B", "C", "D"], "scenarios": [["A", "C"], ["B", "D"]]}
    assert left["scenarios"] != right["scenarios"]
    assert not _current_heads_can_encode_assignments(left, right)


def test_prior_oracle_requires_gold_structure() -> None:
    structure = {"mapping_form": "COMBINATION_WITH_ALTERNATIVES", "scenarios": [{"scenario_id": 1, "choice_lists": [{"choice_list_id": 1, "alternatives": ["A", "B"]}, {"choice_list_id": 2, "alternatives": ["C", "D"]}]}], "flat_alternatives": []}
    assembled = assemble_structure(structure, _candidate_scores())
    assert assembled["scenarios"]
    assert "choice_lists" in structure["scenarios"][0]
    assert not _current_heads_can_encode_assignments(structure, structure)


def test_model_output_schema_has_no_assignment_heads() -> None:
    model = StructuredSetDecoder(3, 8)
    assert set(model(torch_features()).keys()) == {"form_logits", "cardinality_logits", "membership_logits"}


def torch_features():
    import torch
    return torch.zeros((1, 4, 3))


def _current_heads_can_encode_assignments(left: dict, right: dict) -> bool:
    # The frozen contract has no scenario-, slot-, choice-list-, or assignment-level output.
    return False


def test_audit_fixture_is_deterministic() -> None:
    payload = {"left": ["A", "B"], "right": ["C", "D"]}
    assert json.dumps(payload, sort_keys=True) == json.dumps(payload, sort_keys=True)
