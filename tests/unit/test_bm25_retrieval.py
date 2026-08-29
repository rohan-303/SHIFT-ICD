from math import isclose, log

import pytest

from shift_icd.retrieval.baselines import exact_label_rank, token_overlap_rank
from shift_icd.retrieval.bm25 import BM25Index, normalize_text, tokenize


def test_normalization_preserves_medical_terms_and_digits():
    assert normalize_text("  Fracture, WITH 2-Digits! ") == "fracture with 2 digits"
    assert tokenize("fracture with 2 digits") == ["fracture", "with", "2", "digits"]


def test_bm25_hand_computed_score_and_tie_breaking():
    index = BM25Index.from_documents({"A": "acute fracture", "B": "chronic fracture"})
    ranked = index.rank("acute fracture", limit=2)
    assert [code for code, _ in ranked] == ["A", "B"]
    tf = 1
    n = 2
    rare_idf = log(1 + (n - 1 + 0.5) / (1 + 0.5))
    common_idf = log(1 + (n - 2 + 0.5) / (2 + 0.5))
    avgdl = 2.0
    denominator = tf + 1.5 * (1 - 0.75 + 0.75 * 2 / avgdl)
    expected = (rare_idf + common_idf) * (tf * 2.5) / denominator
    assert isclose(ranked[0][1], expected)


def test_bm25_deterministic_tie_breaking():
    index = BM25Index.from_documents({"B": "same", "A": "same"})
    assert [code for code, _ in index.rank("same")] == ["A", "B"]


def test_exact_and_overlap_baselines_are_gold_independent():
    docs = {"A": "acute fracture", "B": "fracture"}
    assert exact_label_rank("acute fracture", docs)[:2] == ["A", "B"]
    assert token_overlap_rank("acute fracture", docs)[:2] == ["A", "B"]


def test_empty_documents_remain_rankable():
    index = BM25Index.from_documents({"A": "", "B": "fracture"})
    assert {code for code, _ in index.rank("unknown")} == {"A", "B"}


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_bm25_rejects_invalid_b(bad):
    with pytest.raises(ValueError):
        BM25Index.from_documents({"A": "x"}, b=bad)
