# ruff: noqa: E501
from __future__ import annotations

import math

import pytest
import torch

from shift_icd.structured_decoder import (
    StructuredAssignmentDecoder,
    form_cross_entropy,
    structured_assignment_loss,
    train_derived_form_weights,
)

COUNTS = {
    "NO_MAP": 251,
    "SINGLE_EXACT": 2095,
    "SINGLE_APPROXIMATE": 4310,
    "ALTERNATIVE": 1613,
    "COMBINATION": 190,
    "COMBINATION_WITH_ALTERNATIVES": 207,
}
ORDER = tuple(COUNTS)


def test_balanced_weights_reproduce_train_formula_and_normalize() -> None:
    weights = train_derived_form_weights(COUNTS)
    assert math.isclose(weights["NO_MAP"], 5.754316069057105, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(sum(COUNTS[name] * weights[name] for name in ORDER) / sum(COUNTS.values()), 1.0, rel_tol=0, abs_tol=1e-15)


def test_unit_weights_and_zero_count_fail_closed() -> None:
    assert train_derived_form_weights({name: 1 for name in ORDER}) == {name: 1.0 for name in ORDER}
    with pytest.raises(ValueError, match="STEP11_FORM_WEIGHT_UNDEFINED_ZERO_CLASS"):
        train_derived_form_weights({**COUNTS, "NO_MAP": 0})


def test_class_order_is_explicit_not_mapping_iteration_order() -> None:
    reordered = {name: COUNTS[name] for name in reversed(ORDER)}
    assert train_derived_form_weights(reordered) == train_derived_form_weights(COUNTS)


def test_weighted_form_fixture_and_unweighted_cross_entropy() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 1.0, -0.5]])
    targets = torch.tensor([0, 2])
    weights = torch.tensor([2.0, 1.0, 0.5])
    raw = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    assert torch.allclose(form_cross_entropy(logits, targets), raw.mean())
    assert torch.allclose(form_cross_entropy(logits, targets, weights), (raw * weights[targets]).mean())


def test_weighting_changes_only_form_component() -> None:
    model = StructuredAssignmentDecoder(3, 8)
    outputs = model(torch.zeros((2, 4, 3)))
    structures = [
        {"mapping_form": "NO_MAP", "scenarios": [], "flat_alternatives": [], "no_map": True},
        {"mapping_form": "COMBINATION", "scenarios": [{"scenario_id": 1, "choice_lists": [{"choice_list_id": 1, "alternatives": ["A"]}, {"choice_list_id": 2, "alternatives": ["B"]}]}], "flat_alternatives": [], "no_map": False},
    ]
    candidates = [["A", "B", "C", "D"], ["A", "B", "C", "D"]]
    unit = torch.ones(6)
    weighted = torch.tensor([5.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    unweighted_loss = structured_assignment_loss(outputs, structures, candidates, unit)
    weighted_loss = structured_assignment_loss(outputs, structures, candidates, weighted)
    assert torch.isfinite(weighted_loss)
    assert not torch.equal(unweighted_loss, weighted_loss)
    # The combination source has the same structural terms; only its form term is scaled.
    no_map_form = form_cross_entropy(outputs["form_logits"][0:1], torch.tensor([0]), unit)
    assert weighted_loss > unweighted_loss
    assert no_map_form > 0
