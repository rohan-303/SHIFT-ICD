from __future__ import annotations

from scripts.external_step8.score_dev import evaluate_rankings


def test_evaluate_rankings_computes_hits_mrr_ndcg_and_membership() -> None:
    groups = [
        {"rows": [{"target_code": "A", "candidate_is_gold": False}, {"target_code": "B", "candidate_is_gold": True}]},
        {"rows": [{"target_code": "C", "candidate_is_gold": True}, {"target_code": "D", "candidate_is_gold": False}]},
    ]
    ranked = [["A", "B"], ["C", "D"]]
    result = evaluate_rankings(groups, ranked)
    assert result["sources"] == 2
    assert result["candidate_membership_valid"] is True
    assert result["hit_at_1"] == 0.5
    assert result["hit_at_3"] == 1.0
    assert result["hit_at_100"] == 1.0
    assert result["mrr"] == 0.75
    assert result["ndcg_at_5"] == 0.8154648767857288
