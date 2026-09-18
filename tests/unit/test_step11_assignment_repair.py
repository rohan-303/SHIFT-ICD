from __future__ import annotations

import torch

from shift_icd.structured_decoder import (
    MAX_SCENARIOS,
    MAX_SLOTS,
    StructuredAssignmentDecoder,
    assemble_from_outputs,
    build_oracle_outputs,
    hierarchical_match,
    match_structures,
    structured_assignment_loss,
)


def test_repaired_output_shapes_and_oracle_distinguish_partitions() -> None:
    model = StructuredAssignmentDecoder(feature_dim=3, hidden_dim=8)
    outputs = model(torch.zeros((1, 4, 3)))
    assert outputs["form_logits"].shape == (1, 6)
    assert outputs["scenario_count_logits"].shape == (1, MAX_SCENARIOS + 1)
    assert outputs["scenario_activity_logits"].shape == (1, MAX_SCENARIOS)
    assert outputs["slot_count_logits"].shape == (1, MAX_SCENARIOS, MAX_SLOTS + 1)
    assert outputs["slot_activity_logits"].shape == (1, MAX_SCENARIOS, MAX_SLOTS)
    assert outputs["assignment_logits"].shape == (1, 4, MAX_SCENARIOS, MAX_SLOTS)
    assert outputs["membership_logits"].shape == (1, 4)


def _cwa(scenarios: list[dict]) -> dict:
    return {"mapping_form": "COMBINATION_WITH_ALTERNATIVES", "scenarios": scenarios, "flat_alternatives": [], "no_map": False}


def _scenario(sid: int, groups: list[list[str]]) -> dict:
    return {"scenario_id": sid, "choice_lists": [{"choice_list_id": i + 1, "alternatives": group} for i, group in enumerate(groups)]}


def test_oracle_reconstructs_different_choice_partitions() -> None:
    left = _cwa([_scenario(1, [["A", "B"], ["C", "D"]])])
    right = _cwa([_scenario(1, [["A", "C"], ["B", "D"]])])
    for structure in (left, right):
        outputs = build_oracle_outputs(structure, ["A", "B", "C", "D"])
        decoded = assemble_from_outputs(outputs, ["A", "B", "C", "D"])
        assert decoded == structure


def test_oracle_reconstructs_multi_scenario_partition() -> None:
    structure = _cwa([_scenario(1, [["A"], ["B"]]), _scenario(2, [["C"], ["D"]])])
    outputs = build_oracle_outputs(structure, ["A", "B", "C", "D"])
    assert assemble_from_outputs(outputs, ["A", "B", "C", "D"]) == structure


def test_matching_is_permutation_invariant() -> None:
    first = _cwa([_scenario(1, [["A", "B"], ["C"]]), _scenario(2, [["D"]])])
    second = _cwa([_scenario(2, [["D"]]), _scenario(1, [["C"], ["B", "A"]])])
    assert match_structures(first, ["A", "B", "C", "D"]) == match_structures(second, ["A", "B", "C", "D"])
    assert hierarchical_match(first) == hierarchical_match(second)


def test_loss_is_finite_and_source_balanced() -> None:
    model = StructuredAssignmentDecoder(3, 8)
    outputs = model(torch.zeros((2, 4, 3)))
    loss = structured_assignment_loss(outputs, [
        {"mapping_form": "NO_MAP", "scenarios": [], "flat_alternatives": [], "no_map": True},
        _cwa([_scenario(1, [["A"], ["B"]])]),
    ], [["A", "B", "C", "D"], ["A", "B", "C", "D"]])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.scenario_queries.grad is not None
    assert model.slot_queries.grad is not None
    assert model.assignment_head.weight.grad is not None
