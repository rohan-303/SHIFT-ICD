import numpy as np

from shift_icd.dense.retrieval import exact_rank, l2_normalize, rrf_fuse
from shift_icd.dense.text import clean_dense_text, format_qwen_query


def test_l2_normalize_rows():
    values = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(values[0], [0.6, 0.8])
    np.testing.assert_allclose(values[1], [0.0, 0.0])


def test_exact_rank_is_deterministic_with_code_ties():
    query = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert exact_rank(query, matrix, ["B", "A", "C"], 3) == [("A", 1.0), ("B", 1.0), ("C", 0.0)]


def test_qwen_instruction_and_text_cleaning():
    text = "  Acute\n\n  fracture  "
    assert clean_dense_text(text) == "Acute fracture"
    assert format_qwen_query("Retrieve diagnoses", "Acute fracture") == "Instruct: Retrieve diagnoses\nQuery:Acute fracture"


def test_rrf_fusion_is_deterministic():
    fused = rrf_fuse(["B", "A", "C"], ["A", "B", "D"], 60, 4)
    assert [code for code, _score in fused] == ["A", "B", "C", "D"]
