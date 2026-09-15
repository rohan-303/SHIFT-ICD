from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from shift_icd.reranking.step8 import (
    MINIMUM_LIST_SIZE,
    collate_variable_source_lists,
    construct_training_list_v2,
    source_balanced_bce_loss,
    source_balanced_listwise_loss,
)


def rows(positives: int, negatives: int = 100) -> list[dict[str, object]]:
    return [
        {"target_code": f"P{i:03d}", "target_description": f"target {i}", "candidate_rank": i + 1, "candidate_is_gold": i < positives}
        for i in range(min(100, positives + negatives))
    ]


@pytest.mark.parametrize(
    ("positive_count", "expected_length", "expected_negatives"),
    [(1, 8, 7), (2, 8, 6), (7, 8, 1), (8, 9, 1), (9, 10, 1), (49, 50, 1)],
)
def test_variable_list_rule_preserves_all_positives(positive_count: int, expected_length: int, expected_negatives: int) -> None:
    selected = construct_training_list_v2(rows(positive_count), seed=17)
    assert len(selected) == expected_length
    assert sum(bool(row["candidate_is_gold"]) for row in selected) == positive_count
    assert sum(not bool(row["candidate_is_gold"]) for row in selected) == expected_negatives
    assert all(row["candidate_rank"] <= 100 for row in selected)


def test_p100_fails_closed_without_in_candidate_negative() -> None:
    with pytest.raises(ValueError, match="SOURCE_HAS_NO_VALID_NEGATIVE_IN_FROZEN_CANDIDATE_SET"):
        construct_training_list_v2(rows(100), seed=17)


def test_deterministic_regeneration_and_no_positive_negative_collision() -> None:
    first = construct_training_list_v2(rows(9), seed=17)
    second = construct_training_list_v2(rows(9), seed=17)
    assert first == second
    positives = {row["target_code"] for row in first if row["candidate_is_gold"]}
    negatives = {row["target_code"] for row in first if not row["candidate_is_gold"]}
    assert positives.isdisjoint(negatives)


def test_variable_length_collation_has_source_offsets_without_padding() -> None:
    lists = [construct_training_list_v2(rows(1), seed=17), construct_training_list_v2(rows(9), seed=17)]
    pairs, offsets = collate_variable_source_lists(lists, source_description="source")
    assert len(pairs) == 18
    assert offsets == [(0, 8), (8, 18)]
    assert all(pair[0] == "source" for pair in pairs)


def test_source_balanced_bce_equalizes_source_weight() -> None:
    short_logits = torch.zeros(8)
    short_labels = torch.zeros(8)
    long_logits = torch.zeros(50)
    long_labels = torch.ones(50)
    loss = source_balanced_bce_loss([(short_logits, short_labels), (long_logits, long_labels)])
    expected = (math.log(2) + math.log(2)) / 2
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-7)


def test_source_balanced_listwise_averages_sources() -> None:
    loss = source_balanced_listwise_loss([torch.zeros(8), torch.zeros(50)], [[0], list(range(49))])
    expected = (math.log(8) + math.log(50) - math.log(49)) / 2
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-7)


def test_real_maximum_positive_train_source_is_preserved() -> None:
    root = Path("artifacts")
    source_id = "track_a_v1.0:ICD9CM_TO_ICD10CM:99811"
    benchmark = {}
    with Path("data/benchmarks/cms_track_a/v1.0/forward/all.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["benchmark_id"] == source_id:
                benchmark = row
                break
    candidates = []
    with gzip.open(root / "candidates/shift_map_full_universe_v2/forward_train_k100.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["source_id"] == source_id:
                row["candidate_is_gold"] = row["target_code"] in set(benchmark["valid_target_codes"])
                candidates.append(row)
    selected = construct_training_list_v2(candidates, seed=17)
    assert len(selected) == 50
    assert sum(row["candidate_is_gold"] for row in selected) == 49
    assert sum(not row["candidate_is_gold"] for row in selected) == 1


def test_fixed_length_bce_matches_previous_candidate_mean() -> None:
    logits = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5, 0.2, -0.2, 0.3])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    from torch.nn import functional

    expected = functional.binary_cross_entropy_with_logits(logits, labels)
    actual = source_balanced_bce_loss([(logits, labels)])
    assert torch.isclose(actual, expected, atol=1e-7)


def test_contract_constants_and_no_test_inputs() -> None:
    assert MINIMUM_LIST_SIZE == 8
    test_artifact = Path("artifacts/candidates/shift_map_full_universe_v2/forward_test_k100.jsonl.gz")
    assert test_artifact.is_file()
    assert test_artifact.stat().st_size > 0


def test_protocol_v1_is_preserved_and_protocol_v2_supersedes_it() -> None:
    old = Path("artifacts/experiments/step8_full_universe/r2_search_protocol.json")
    assert hashlib.sha256(old.read_bytes()).hexdigest() == "999119c1c9ea810dc05eb234a346f79e0f642433087faf93497c9b10812bc719"
    supersession = json.loads(Path("artifacts/experiments/step8_full_universe/r2_search_protocol_v1_supersession.json").read_text())
    assert supersession["status"] == "SUPERSEDED_BY_TRAINING_LIST_CONTRACT_AMENDMENT"
    protocol_v2 = json.loads(Path("artifacts/experiments/step8_full_universe/r2_search_protocol_v2.json").read_text())
    assert protocol_v2["training_list_contract"] == "training_list_contract_v2.json"
    assert protocol_v2["supersedes_protocol_sha256"] == supersession["superseded_sha256"]
