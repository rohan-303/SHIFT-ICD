import numpy as np
import pytest

from shift_icd.reranking.shift_map_v2 import (
    build_cross_encoder_text,
    candidate_coverage,
    listwise_set_positive_loss,
    preserve_candidate_membership,
    sample_negative_ranks,
    source_balanced_bce,
)


def test_primary_input_excludes_metadata_and_codes() -> None:
    text = build_cross_encoder_text("source long description", "target long description")
    assert text == ("source long description", "target long description")
    for forbidden in ("candidate_is_gold", "retriever_rank", "retriever_score", "SINGLE_EXACT", "A123", "B456"):
        assert forbidden not in " ".join(text)


def test_candidate_membership_and_hit100_are_preserved() -> None:
    original = [["A", "B", "C"], ["D", "E", "F"]]
    reranked = [["C", "A", "B"], ["F", "D", "E"]]
    assert preserve_candidate_membership(original, reranked)
    assert candidate_coverage(original, {"A", "F"}, 3) == candidate_coverage(reranked, {"A", "F"}, 3)


def test_candidate_membership_rejects_addition_or_removal() -> None:
    with pytest.raises(ValueError):
        preserve_candidate_membership([["A"]], [["B"]])


def test_source_balanced_bce_gives_each_source_equal_weight() -> None:
    logits = [[0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    labels = [[1.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    value = source_balanced_bce(logits, labels)
    assert value == pytest.approx(0.6931471805599453)


def test_set_positive_listwise_loss_is_stable_and_uses_all_positives() -> None:
    value = listwise_set_positive_loss(np.array([1000.0, 999.0, -1000.0]), [0, 1])
    assert np.isfinite(value)
    assert value < 0.7


def test_negative_sampling_excludes_positive_ranks_and_is_deterministic() -> None:
    ranks = sample_negative_ranks(list(range(1, 101)), {2, 20}, strategy="mixed", seed=42, count=7)
    assert ranks == sample_negative_ranks(list(range(1, 101)), {2, 20}, strategy="mixed", seed=42, count=7)
    assert 2 not in ranks and 20 not in ranks
    assert len(ranks) == 7


def test_coverage_requires_ordinary_sources_in_denominator() -> None:
    rows = [
        {"kind": "SINGLE_EXACT", "gold": {"A"}, "candidates": ["A"]},
        {"kind": "SINGLE_APPROXIMATE", "gold": {"B"}, "candidates": ["C"]},
        {"kind": "NO_MAP", "gold": set(), "candidates": []},
        {"kind": "COMBINATION", "gold": {"D"}, "candidates": ["D"]},
    ]
    assert candidate_coverage(rows, k=1, row_format=True) == pytest.approx(0.5)
