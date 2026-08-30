import math

import torch

from shift_icd.shift_map.losses import l1_masked_single_infonce, l2_set_positive_infonce
from shift_icd.shift_map.mining import audit_negative_collisions, mine_random_negatives
from shift_icd.shift_map.training import (
    TrainingExample,
    eligible_examples,
    sample_positive,
    source_balanced_epoch,
)


def ex(kind="SINGLE_EXACT", targets=None, split="train", direction="ICD9CM_TO_ICD10CM", source="s"):
    return TrainingExample(
        benchmark_id=f"id:{source}",
        source_code=source,
        source_label="source",
        source_family="A00",
        direction=direction,
        split=split,
        source_family_split="train",
        mapping_kind=kind,
        valid_target_codes=tuple(targets or ("T1",)),
        lexical_difficulty="LEXICAL_LOW",
    )


def test_policies_exclude_complex_and_no_map():
    rows = [
        ex("SINGLE_EXACT"),
        ex("SINGLE_APPROXIMATE", source="a"),
        ex("ALTERNATIVE", ["T1", "T2"], source="b"),
        ex("COMBINATION", source="c"),
        ex("NO_MAP", [], source="d"),
    ]
    assert [x.source_code for x in eligible_examples(rows, "P0")] == ["s"]
    assert [x.source_code for x in eligible_examples(rows, "P1")] == ["s", "a"]
    assert [x.source_code for x in eligible_examples(rows, "P2")] == ["s", "a", "b"]


def test_source_balanced_and_positive_sampling_are_deterministic():
    rows = [ex(source="a"), ex(targets=["T1", "T2", "T3"], kind="ALTERNATIVE", source="b")]
    epoch = source_balanced_epoch(rows, seed=17, epoch=2)
    assert [x.source_code for x in epoch] == ["a", "b"]
    assert sample_positive(rows[1], seed=17, epoch=0) == sample_positive(rows[1], seed=17, epoch=0)
    assert sample_positive(rows[1], seed=17, epoch=0) in rows[1].valid_target_codes


def test_full_positive_set_masks_collisions_and_miner_excludes_gold():
    item = ex("ALTERNATIVE", ["T1", "T2"], source="a")
    neg = mine_random_negatives(item, ["T1", "T2", "T3", "T4"], count=2, seed=1)
    assert not set(neg) & set(item.valid_target_codes)
    assert audit_negative_collisions({"a": item}, {"a": neg})["remaining_gold_collisions"] == 0


def test_l1_is_directional_and_matches_manual_cross_entropy():
    q = torch.tensor([[1.0, 0.0]], requires_grad=True)
    c = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = l1_masked_single_infonce(q, c, [0], [[0]], temperature=1.0)
    expected = -torch.log_softmax(torch.tensor([1.0, 0.0]), dim=0)[0]
    assert torch.allclose(loss, expected)
    loss.backward()
    assert q.grad is not None


def test_l2_is_stable_count_normalized_set_loss():
    q = torch.tensor([[1.0, 0.0]])
    c = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    loss = l2_set_positive_infonce(q, c, [[0, 1]], temperature=0.05)
    assert math.isfinite(float(loss))
    assert loss < 1.0


def test_aggregation_uses_valid_target_set_size_not_row_n():
    from aggregate_shift_map_v1 import add_positive_set_size

    rows = [{"n": 1, "valid_target_codes": ["A", "B"]}, {"n": 1, "valid_target_codes": ["C"]}]
    result = add_positive_set_size(rows)
    assert [row["alternative_size"] for row in result] == ["2-5", "1"]


def test_aggregation_reduces_slice_metrics_across_seeds():
    from aggregate_shift_map_v1 import reduce_seed_slice

    rows = [
        {"seed": 17, "mapping_kind": "SINGLE_EXACT", "Hit@10": 0.8},
        {"seed": 42, "mapping_kind": "SINGLE_EXACT", "Hit@10": 0.6},
    ]
    result = reduce_seed_slice(rows, ["mapping_kind"], ["Hit@10"])
    assert len(result) == 1
    assert result[0]["n"] == 2
    assert result[0]["Hit@10"] == 0.7
