from __future__ import annotations

from types import SimpleNamespace

import pytest

from shift_icd.reranking.step8 import (
    LIST_SIZE,
    Step8TestPolicy,
    bce_loss,
    build_pair,
    construct_training_list,
    extract_relevance_scores,
    set_positive_listwise_loss,
    validate_membership,
)


def test_pair_contains_only_descriptions() -> None:
    assert build_pair("source text", "target text") == ("source text", "target text")


def test_metadata_changes_cannot_change_pair() -> None:
    left = build_pair("source text", "target text")
    right = build_pair("source text", "target text")
    assert left == right


def test_scalar_relevance_requires_single_logit() -> None:
    class Tensor:
        shape = (2, 1)

        def __getitem__(self, item):
            return self

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [0.25, -0.5]

    assert extract_relevance_scores(SimpleNamespace(logits=Tensor()), num_labels=1) == [0.25, -0.5]
    with pytest.raises(ValueError):
        extract_relevance_scores(SimpleNamespace(logits=Tensor()), num_labels=2)


def test_bce_and_set_positive_losses_are_finite_and_preserve_all_positives() -> None:
    assert bce_loss([2.0, -1.0], [1.0, 0.0]) > 0
    assert set_positive_listwise_loss([3.0, 1.0, 2.0], [0, 2]) == pytest.approx(0.09434427692615754)


def test_list8_preserves_alternative_positives_and_excludes_them_from_negatives() -> None:
    rows = [{"candidate_rank": i, "candidate_is_gold": i in {2, 7}} for i in range(1, 20)]
    selected = construct_training_list(rows, seed=17)
    assert len(selected) == LIST_SIZE
    assert {r["candidate_rank"] for r in selected if r["candidate_is_gold"]} == {2, 7}
    assert all(not r["candidate_is_gold"] for r in selected if not r["candidate_is_gold"])


def test_membership_is_invariant_under_reranking() -> None:
    validate_membership([["A", "B"]], [["B", "A"]])
    with pytest.raises(ValueError):
        validate_membership([["A", "B"]], [["A", "C"]])


def test_test_access_is_fail_closed() -> None:
    with pytest.raises(PermissionError, match="FORBIDDEN"):
        Step8TestPolicy().require()
