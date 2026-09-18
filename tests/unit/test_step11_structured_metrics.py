# ruff: noqa: E501
from __future__ import annotations

from shift_icd.structured_metrics import (
    assert_official_dev_access_forbidden,
    assert_test_access_forbidden,
    canonical_signature,
    choose_best,
    choose_best_epoch,
    classify_retrieval_failure,
    complete_scenario_success,
    exact_canonical_structure_match,
    flat_set_metrics,
    structure_validity,
    verify_feature_provenance,
)


def _single(form: str = "SINGLE_EXACT", targets: list[str] | None = None) -> dict:
    return {"mapping_form": form, "flat_alternatives": ["A"] if targets is None else targets, "scenarios": []}


def _combo(scenarios: list[list[list[str]]]) -> dict:
    return {
        "mapping_form": "COMBINATION_WITH_ALTERNATIVES",
        "flat_alternatives": [],
        "scenarios": [{"choice_lists": [{"alternatives": values} for values in choices]} for choices in scenarios],
    }


def test_exact_match_is_order_invariant_but_grouping_sensitive() -> None:
    first = _combo([[['A', 'B'], ['C']], [['D']]])
    same = _combo([[['D']], [['C'], ['B', 'A']]])
    different = _combo([[['A', 'C'], ['B']], [['D']]])
    assert exact_canonical_structure_match(first, same)
    assert not exact_canonical_structure_match(first, different)
    assert canonical_signature(first) == canonical_signature(same)


def test_complete_scenario_allows_partial_prediction_success() -> None:
    gold = _combo([[['A'], ['B']]])
    predicted = _combo([[['A', 'X'], ['B']]])
    assert complete_scenario_success(predicted, gold)


def test_flat_empty_set_behavior() -> None:
    assert flat_set_metrics(_single("NO_MAP", []), _single("NO_MAP", []))["f1"] == 1.0
    assert flat_set_metrics(_single("SINGLE_EXACT", ["A"]), _single("NO_MAP", []))["f1"] == 0.0


def test_validity_and_failure_classification() -> None:
    assert structure_validity(_single(targets=["A"]), ["A", "B"])
    assert not structure_validity(_single(targets=["Z"]), ["A", "B"])
    assert classify_retrieval_failure(False, False) == "RETRIEVAL_LIMITED"
    assert classify_retrieval_failure(False, True) == "DECODER_LIMITED"
    assert classify_retrieval_failure(True, True) is None


def test_selection_hierarchy_and_configuration_tie_break() -> None:
    rows = [
        {"configuration_id": "lr3_weighted", "exact_canonical_structure_match": 0.2, "complete_scenario_success": 1.0, "mapping_form_macro_f1": 1.0, "no_map_f1": 1.0, "flat_set_f1": 1.0, "cardinality_exact_accuracy": 1.0},
        {"configuration_id": "lr1_unweighted", "exact_canonical_structure_match": 0.2, "complete_scenario_success": 1.0, "mapping_form_macro_f1": 1.0, "no_map_f1": 1.0, "flat_set_f1": 1.0, "cardinality_exact_accuracy": 1.0},
        {"configuration_id": "lr0", "exact_canonical_structure_match": 0.3, "complete_scenario_success": 0.0, "mapping_form_macro_f1": 0.0, "no_map_f1": 0.0, "flat_set_f1": 0.0, "cardinality_exact_accuracy": 0.0},
    ]
    assert choose_best(rows)["configuration_id"] == "lr0"
    assert choose_best(rows[:2])["configuration_id"] == "lr1_unweighted"


def test_each_selection_level_and_earliest_epoch_tie() -> None:
    keys = ["exact_canonical_structure_match", "complete_scenario_success", "mapping_form_macro_f1", "no_map_f1", "flat_set_f1", "cardinality_exact_accuracy"]
    base = {key: 0.0 for key in keys}
    base.update({"configuration_id": "b", "epoch": 2})
    for key in keys:
        candidate = dict(base)
        candidate[key] = 1.0
        assert choose_best([candidate, base]) is candidate
        base = candidate
    early = dict(base, epoch=1)
    assert choose_best_epoch([base, early])["epoch"] == 1


def test_feature_provenance_fails_closed_on_mismatch() -> None:
    expected = {"feature_dimension": 1539, "canonical_checkpoint_sha256": "x", "candidate_artifacts": {"train_sha256": "y"}, "normalized_retriever_score": {"ddof": 0}}
    verify_feature_provenance(dict(expected), expected)
    bad = dict(expected, feature_dimension=1540)
    try:
        verify_feature_provenance(bad, expected)
    except RuntimeError as error:
        assert str(error) == "STEP11_FEATURE_PROVENANCE_MISMATCH"
    else:
        raise AssertionError("feature provenance mismatch was accepted")


def test_official_dev_and_test_are_quarantined() -> None:
    for guard, message in ((assert_official_dev_access_forbidden, "STEP11_OFFICIAL_DEV_ACCESS_FORBIDDEN"), (assert_test_access_forbidden, "STEP11_TEST_ACCESS_FORBIDDEN")):
        try:
            guard()
        except RuntimeError as error:
            assert str(error) == message
        else:
            raise AssertionError("quarantine guard did not stop access")
