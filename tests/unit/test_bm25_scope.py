import pytest

from shift_icd.evaluation.scope import (
    EvaluationScope,
    canonical_benchmark_diff,
    normalize_code_for_lookup,
    random_hit_probability,
    validate_slice_partition,
)


def test_scope_filters_direction_protocol_and_partition():
    rows = [
        {"direction": "ICD9CM_TO_ICD10CM", "split_protocol": "stratified", "partition": "test", "benchmark_id": "a"},
        {"direction": "ICD10CM_TO_ICD9CM", "split_protocol": "stratified", "partition": "test", "benchmark_id": "b"},
    ]
    scope = EvaluationScope("ICD9CM_TO_ICD10CM", "stratified", "test")
    assert scope.filter(rows) == [rows[0]]


def test_slice_partition_is_mutually_exclusive():
    ids = {"a", "b", "c"}
    assert validate_slice_partition({"ONE": {"a"}, "TWO": {"b"}, "THREE": {"c"}}, ids) == []


def test_membership_diff_is_bidirectional():
    assert canonical_benchmark_diff({"a", "b"}, {"b", "c"}) == {"canonical_only": ["a"], "benchmark_only": ["c"]}


def test_display_and_normalized_code_lookup_match():
    assert normalize_code_for_lookup("V54.12") == "v5412"
    assert normalize_code_for_lookup("V5412") == "v5412"


def test_random_hit_probability_is_stable_and_bounded():
    assert random_hit_probability(100, 1, 10) == pytest.approx(0.1)
    assert 0.0 <= random_hit_probability(17513, 533, 100) <= 1.0
