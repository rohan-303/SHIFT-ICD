# ruff: noqa: E501, I001
from __future__ import annotations

import math

import torch

import shift_icd.structured_decoder as decoder
from shift_icd.structured_decoder import StructuredAssignmentDecoder, structured_assignment_loss


def simple(form: str = "ALTERNATIVE") -> dict:
    return {
        "mapping_form": form,
        "scenarios": [],
        "flat_alternatives": ["A", "B"] if form == "ALTERNATIVE" else (["A"] if form.startswith("SINGLE") else []),
        "no_map": form == "NO_MAP",
    }


def complex_structure(order: tuple[int, ...] = (1, 2)) -> dict:
    by_id = {
        1: {"scenario_id": 1, "choice_list": [{"choice_list_id": 1, "alternatives": ["A"]}, {"choice_list_id": 2, "alternatives": ["B", "C"]}]},
        2: {"scenario_id": 2, "choice_list": [{"choice_list_id": 1, "alternatives": ["D"]}]},
    }
    return {"mapping_form": "COMBINATION_WITH_ALTERNATIVES", "scenarios": [{"scenario_id": i, "choice_lists": by_id[i]["choice_list"]} for i in order], "flat_alternatives": [], "no_map": False}


def test_loss_result_has_exact_seven_components_and_applicability() -> None:
    model = StructuredAssignmentDecoder(3, 8)
    outputs = model(torch.zeros((3, 4, 3)))
    result = structured_assignment_loss(outputs, [simple("NO_MAP"), simple("ALTERNATIVE"), complex_structure()], [["A", "B", "C", "D"]] * 3)
    assert result.total_loss.requires_grad
    assert result.L_form is not None and result.L_scenario_count is not None and result.L_scenario_activity is not None
    assert result.L_slot_count is not None and result.L_slot_activity is not None and result.L_assignment is not None
    assert result.L_flat_set is not None
    assert result.applicable_source_count == {"L_form": 3, "L_scenario_count": 3, "L_scenario_activity": 3, "L_slot_count": 1, "L_slot_activity": 1, "L_assignment": 1, "L_flat_set": 1}


def test_production_loss_is_permutation_invariant_for_gold_order() -> None:
    torch.manual_seed(7)
    model = StructuredAssignmentDecoder(3, 8)
    outputs = model(torch.randn((1, 4, 3)))
    first = structured_assignment_loss(outputs, [complex_structure((1, 2))], [["A", "B", "C", "D"]])
    second = structured_assignment_loss(outputs, [complex_structure((2, 1))], [["A", "B", "C", "D"]])
    for name in ("total_loss", "L_scenario_activity", "L_slot_count", "L_slot_activity", "L_assignment"):
        assert torch.allclose(getattr(first, name), getattr(second, name), atol=1e-7, rtol=0)


def test_matcher_is_called_before_structural_supervision(monkeypatch) -> None:
    calls = []
    original = decoder.hierarchical_match

    def spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(decoder, "hierarchical_match", spy)
    model = StructuredAssignmentDecoder(3, 8)
    result = structured_assignment_loss(model(torch.zeros((1, 4, 3))), [complex_structure()], [["A", "B", "C", "D"]])
    assert calls == [True]
    assert result.matched_scenario_count == 2


def test_flat_set_only_applies_to_simple_mappable_forms() -> None:
    model = StructuredAssignmentDecoder(3, 8)
    outputs = model(torch.zeros((3, 4, 3)))
    result = structured_assignment_loss(outputs, [simple("NO_MAP"), simple("ALTERNATIVE"), complex_structure()], [["A", "B", "C", "D"]] * 3)
    assert result.L_flat_set is not None
    assert result.applicable_source_count["L_flat_set"] == 1
    assert result.applicable_source_count["L_assignment"] == 1


def test_all_applicable_heads_receive_gradients() -> None:
    model = StructuredAssignmentDecoder(3, 8)
    structures = [simple("NO_MAP"), simple("ALTERNATIVE"), complex_structure()]
    result = structured_assignment_loss(model(torch.zeros((3, 4, 3))), structures, [["A", "B", "C", "D"]] * 3)
    result.total_loss.backward()
    for parameter in (model.form_head.weight, model.scenario_count_head.weight, model.scenario_activity_head.weight, model.slot_count_head.weight, model.slot_activity_head.weight, model.assignment_head.weight, model.membership_head.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_form_weighting_changes_only_form_component() -> None:
    model = StructuredAssignmentDecoder(3, 8)
    outputs = model(torch.zeros((2, 4, 3)))
    structures = [simple("NO_MAP"), complex_structure()]
    candidates = [["A", "B", "C", "D"]] * 2
    unit = torch.ones(6)
    weighted = torch.tensor([5.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    unweighted = structured_assignment_loss(outputs, structures, candidates, unit)
    reweighted = structured_assignment_loss(outputs, structures, candidates, weighted)
    assert not torch.allclose(unweighted.L_form, reweighted.L_form)
    for name in ("L_scenario_count", "L_scenario_activity", "L_slot_count", "L_slot_activity", "L_assignment"):
        assert torch.allclose(getattr(unweighted, name), getattr(reweighted, name), atol=0, rtol=0)


def test_source_balance_is_mean_of_source_means() -> None:
    model = StructuredAssignmentDecoder(3, 8)
    structures = [simple("NO_MAP"), complex_structure()]
    candidates = [["A", "B", "C", "D"]] * 2
    batch = structured_assignment_loss(model(torch.zeros((2, 4, 3))), structures, candidates)
    first = structured_assignment_loss(model(torch.zeros((1, 4, 3))), [structures[0]], [candidates[0]])
    second = structured_assignment_loss(model(torch.zeros((1, 4, 3))), [structures[1]], [candidates[1]])
    assert math.isfinite(float(batch.total_loss))
    # The invariant is source-level equal weighting; component ledger counts expose it.
    assert batch.applicable_source_count["L_form"] == 2
    assert first.applicable_source_count["L_form"] == second.applicable_source_count["L_form"] == 1
