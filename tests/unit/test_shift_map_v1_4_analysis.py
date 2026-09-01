from __future__ import annotations

import numpy as np
from shift_map_v1_4_analysis import (
    bootstrap_delta,
    candidate_coverage,
    gini,
    group_by_cardinality,
    paired_counts,
    summarize_values,
)


def test_group_by_cardinality_has_five_protocol_buckets() -> None:
    rows = [{"n": n} for n in [1, 2, 6, 21, 101]]
    groups = group_by_cardinality(rows)
    assert list(groups) == ["G=1", "G=2-5", "G=6-20", "G=21-100", "G>100"]
    assert [len(groups[k]) for k in groups] == [1, 1, 1, 1, 1]


def test_paired_counts_and_bootstrap_are_deterministic() -> None:
    a = np.array([1, 0, 1, 0])
    b = np.array([0, 0, 1, 1])
    assert paired_counts(a, b) == {"l2_success": 1, "l1_success": 1, "both_success": 1, "both_fail": 1}
    assert bootstrap_delta(a, b, seed=2026, reps=100)["delta"] == 0.0
    assert bootstrap_delta(a, b, seed=2026, reps=100) == bootstrap_delta(a, b, seed=2026, reps=100)


def test_summary_gini_and_candidate_coverage() -> None:
    summary = summarize_values([1.0, 2.0, 3.0])
    assert summary["median"] == 2.0
    assert round(gini([1, 1, 2]), 6) == round(1 / 6, 6)
    rows = [{"valid_target_codes": ["a"], "ranked_codes": ["a", "b"]}, {"valid_target_codes": ["z"], "ranked_codes": ["a"]}]
    assert candidate_coverage(rows, 1) == 0.5


def test_candidate_is_gold_is_not_input_text() -> None:
    row = {
        "source_description": "source label",
        "target_description": "target label",
        "candidate_is_gold": True,
    }
    model_input = f"{row['source_description']} [SEP] {row['target_description']}"
    assert "candidate_is_gold" not in model_input
    assert "True" not in model_input
