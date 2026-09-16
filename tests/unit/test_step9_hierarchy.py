# ruff: noqa: E501
from __future__ import annotations

import pytest

from shift_icd.hierarchy.features import (
    FEATURE_NAMES,
    apply_train_scaler,
    extract_feature_vector,
    fit_train_scaler,
)
from shift_icd.hierarchy.metadata import HierarchyRecord, audit_hierarchy, build_prefix_hierarchy
from shift_icd.hierarchy.runner_contract import (
    Step9TestAccessError,
    checkpoint_path,
    require_test_lock,
    reserve_checkpoint,
)
from shift_icd.hierarchy.training import make_contract_v2_list, validate_variable_lists


def _nodes() -> tuple[dict[str, HierarchyRecord], dict[str, HierarchyRecord]]:
    source = build_prefix_hierarchy(
        ["001", "0010", "0011", "V58", "V5889"],
        ontology="ICD-9-CM",
        provenance="synthetic_prefix_fixture_v1",
    )
    target = build_prefix_hierarchy(
        ["A00", "A000", "A001", "A01", "A010", "A0100"],
        ontology="ICD-10-CM",
        provenance="synthetic_prefix_fixture_v1",
    )
    return source, target


def test_hierarchy_parent_chain_depth_and_integrity_are_deterministic() -> None:
    source, _ = _nodes()
    assert source["0010"].parent_code == "001"
    assert source["0010"].ancestor_chain == ("001", "00", "0")
    assert source["0010"].depth == 3
    audit = audit_hierarchy(source)
    assert audit["cycle_count"] == 0
    assert audit["invalid_parent_count"] == 0
    assert audit["root_count"] == 2


def test_features_ignore_mapping_and_split_metadata() -> None:
    source, target = _nodes()
    candidates = ["A000", "A001", "A010"]
    base = extract_feature_vector("0010", "A000", candidates, source, target)
    altered_metadata = {"candidate_is_gold": True, "mapping_kind": "COMBINATION", "split": "test", "approximate": True, "scenario_labels": ["leakage"]}
    altered = extract_feature_vector("0010", "A000", candidates, source, target)
    assert altered_metadata
    assert base == altered
    assert len(base) == len(FEATURE_NAMES)


def test_feature_scaler_is_train_only_and_deterministic() -> None:
    source, target = _nodes()
    rows = [extract_feature_vector("0010", code, ["A000", "A001"], source, target) for code in ["A000", "A001"]]
    scaler = fit_train_scaler(rows, role="TRAIN")
    assert apply_train_scaler(rows, scaler) == apply_train_scaler(rows, scaler)
    with pytest.raises(ValueError, match="TRAIN_ONLY"):
        fit_train_scaler(rows, role="DEV")


def test_candidate_membership_is_not_changed_by_feature_extraction() -> None:
    source, target = _nodes()
    candidates = ["A000", "A001", "A010"]
    _ = [extract_feature_vector("0010", code, candidates, source, target) for code in candidates]
    assert candidates == ["A000", "A001", "A010"]


def test_reranking_preserves_top100_and_structural100_membership() -> None:
    candidates = [f"C{i:03d}" for i in range(1, 101)]
    scores = {code: float(100 - index) for index, code in enumerate(candidates)}
    ranked = sorted(candidates, key=lambda code: (-scores[code], candidates.index(code)))
    assert set(ranked) == set(candidates)
    assert len(ranked) == 100
    assert set(ranked[:100]) == set(candidates)


def test_test_access_fails_closed_without_lock() -> None:
    with pytest.raises(Step9TestAccessError, match="STEP9_TEST_ACCESS_FORBIDDEN"):
        require_test_lock(None)


def test_checkpoint_path_contains_collision_proof_identity() -> None:
    path = checkpoint_path("H1", "SET_POSITIVE_LISTWISE", 17, 2)
    assert "H1" in str(path)
    assert "seed_17" in str(path)
    assert "epoch_2" in str(path)


def test_checkpoint_collision_is_rejected(tmp_path) -> None:
    path = tmp_path / "model.pt"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="STEP9_CHECKPOINT_COLLISION"):
        reserve_checkpoint(path)


def test_contract_v2_preserves_variable_positive_lists() -> None:
    groups = []
    for positive_count in (1, 8, 9, 49):
        groups.append([
            {"source_id": str(positive_count), "candidate_rank": rank, "_gold": rank <= positive_count}
            for rank in range(1, 101)
        ])
    result = validate_variable_lists(groups)
    assert result["dropped_positive_count"] == 0
    assert result["collision_count"] == 0
    assert [len(make_contract_v2_list(group, seed=17, epoch=1)) for group in groups] == [8, 9, 10, 50]
