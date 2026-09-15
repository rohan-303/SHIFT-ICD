from __future__ import annotations

import json
from pathlib import Path

import pytest

from shift_icd.reranking.step8 import (
    EXCLUDED_KINDS,
    MAX_LENGTH,
    MODEL_REVISION,
    ORDINARY_KINDS,
    Step8TestPolicy,
    construct_training_list,
    evaluator_membership_invariance,
    ordinary_training_eligible,
    validate_membership,
)

ROOT = Path(__file__).resolve().parents[2]


def test_pinned_revision_and_length_contract() -> None:
    assert MODEL_REVISION == "71caf65d4927987813984f54c284405a13fcca49"
    assert MAX_LENGTH == 96


def test_frozen_candidate_hashes_are_exact() -> None:
    manifest = json.loads((ROOT / "artifacts/experiments/shift_map_full_universe/candidate_freeze_manifest.json").read_text())
    expected = {row["split"]: row["sha256"] for row in manifest["files"]}
    assert expected == {
        "train": "d90dcb937748f1c0b173416d6b8f23f1c97ca8e8caf8934efdf8120993c1cb10",
        "dev": "c7240c1f38dd87f54d63c97999cba71d91f6d0887d89bc2d4094f9ba38ad6a6f",
        "test": "6ef38443a201b1d630148b30717d845b0785b1a4ed433eeab4b03d893b6bf1ce",
    }


def test_training_population_excludes_structural_and_no_map() -> None:
    assert ORDINARY_KINDS == {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"}
    assert EXCLUDED_KINDS == {"NO_MAP", "COMBINATION", "COMBINATION_WITH_ALTERNATIVES"}
    assert ordinary_training_eligible({"mapping_kind": "SINGLE_EXACT", "gold_codes": ["A"]})
    assert not ordinary_training_eligible({"mapping_kind": "COMBINATION", "gold_codes": ["A"]})
    assert not ordinary_training_eligible({"mapping_kind": "NO_MAP", "gold_codes": []})
    assert not ordinary_training_eligible({"mapping_kind": "SINGLE_EXACT", "gold_codes": []})


def test_alternative_positive_is_never_sampled_as_negative() -> None:
    rows = [{"candidate_rank": i, "candidate_is_gold": i in {1, 3}} for i in range(1, 20)]
    selected = construct_training_list(rows, seed=17)
    assert len(selected) == 8
    assert {row["candidate_rank"] for row in selected if row["candidate_is_gold"]} == {1, 3}


def test_candidate_set_can_only_be_reordered() -> None:
    validate_membership([["a", "b", "c"]], [["c", "a", "b"]])
    with pytest.raises(ValueError):
        validate_membership([["a", "b", "c"]], [["a", "b"]])


def test_default_test_policy_is_closed() -> None:
    with pytest.raises(PermissionError, match="FORBIDDEN"):
        Step8TestPolicy().require()


def test_hit_and_structural_at_100_are_invariant_to_reranking() -> None:
    before = [["A", "B", "C"], ["D", "E", "F"]]
    after = [["C", "A", "B"], ["F", "E", "D"]]
    result = evaluator_membership_invariance(before, after, [{"B"}, {"F"}], k=100)
    assert result["hit_at_100_before"] == result["hit_at_100_after"] == 1.0
    assert result["structural_at_100_before"] == result["structural_at_100_after"] == 1.0
