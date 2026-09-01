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


def test_l2_full_positive_numerator_matches_manual_logmeanexp():
    q = torch.tensor([[1.0, 0.0]])
    c = torch.tensor([[1.0, 0.0], [0.5, 0.0], [-1.0, 0.0]])
    temperature = 1.0
    logits = torch.tensor([1.0, 0.5, -1.0])
    expected = -(torch.logsumexp(logits[:2], dim=0) - math.log(2.0) - torch.logsumexp(logits, dim=0))
    assert torch.allclose(l2_set_positive_infonce(q, c, [[0, 1]], temperature), expected)


def test_l2_rejects_unrepresented_or_empty_positive_sets():
    q = torch.tensor([[1.0, 0.0]])
    c = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    import pytest

    with pytest.raises(ValueError, match="at least one"):
        l2_set_positive_infonce(q, c, [[]], 0.05)


def test_l2_positive_cardinality_is_count_normalized():
    q = torch.tensor([[1.0, 0.0]])
    c = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    one = l2_set_positive_infonce(q, c, [[0]], 1.0)
    two = l2_set_positive_infonce(q, c, [[0, 1]], 1.0)
    assert torch.allclose(one, two)


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


def test_dense_target_corpus_is_direction_scoped():
    import pandas as pd

    from shift_icd.dense.corpus import build_target_corpus

    frame = pd.DataFrame(
        [
            {"direction": "ICD9CM_TO_ICD10CM", "target_code": "A", "target_label": "forward", "target_short_description": "", "row_id": 1},
            {"direction": "ICD10CM_TO_ICD9CM", "target_code": "A", "target_label": "backward", "target_short_description": "", "row_id": 2},
        ]
    )
    corpus = build_target_corpus(frame, "ICD9CM_TO_ICD10CM")
    assert corpus.codes == ("A",)
    assert corpus.descriptions == ("forward",)


def test_shift_map_v1_3_provenance_marks_l1_bridges_required():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    provenance = json.loads((root / "artifacts/experiments/shift_map_v1_3/checkpoint_provenance.json").read_text())
    composability = json.loads((root / "artifacts/experiments/shift_map_v1_3/configuration_composability.json").read_text())
    assert provenance["negative_strategy_composable"] is False
    assert provenance["learning_rate_composable"] is False
    assert composability["negative_strategy_bridge_required"] is True
    assert composability["learning_rate_bridge_required"] is True
    assert composability["historical_negative_strategy_loss"] == ["L1"]
    assert composability["historical_learning_rate_losses"] == ["L1"]


def test_training_manifest_preserves_test_selection_boundary():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "artifacts/experiments/shift_map_v1_3/training_manifest.json").read_text())
    assert manifest["evaluator_version"] == "2.0"
    assert manifest["train_direction"] == "ICD9CM_TO_ICD10CM"
    assert manifest["test_used_for_selection"] is False


    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    fix = json.loads((root / "artifacts/experiments/shift_map_v1_2/evaluator_fix.json").read_text())
    selection = json.loads((root / "artifacts/experiments/shift_map_v1_2/selection_replay.json").read_text())
    assert fix["evaluator_version"] == "2.0"
    assert selection["test_used_for_selection"] is False
    assert selection["corrected_canonical_seed"] in {17, 42, 2026}


def test_shift_map_v1_historical_artifacts_are_preserved():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    assert (root / "artifacts/experiments/shift_map_v1/test_lock.json").exists()
    assert (root / "artifacts/experiments/shift_map_v1/test_metrics.json").exists()
    assert (root / "artifacts/experiments/shift_map_v1_2/decision.json").exists()


def test_shift_map_v1_2_paired_population_identity():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    parity = json.loads(
        (root / "artifacts/experiments/shift_map_v1_2/zero_shot_parity.json").read_text()
    )
    decision = json.loads(
        (root / "artifacts/experiments/shift_map_v1_2/decision.json").read_text()
    )
    assert parity["parity"] == "PASS"
    assert decision["next_path"] == "C"
    import pandas as pd

    from shift_icd.dense.corpus import BACKWARD, FORWARD, build_target_corpus

    frame = pd.read_parquet("data/processed/cms/2018_gem/normalized_rows.parquet")
    forward = build_target_corpus(frame, FORWARD)
    backward = build_target_corpus(frame, BACKWARD)
    assert len(forward.codes) == 17513
    assert len(backward.codes) == 11690
    assert forward.terminology_version == backward.terminology_version == "CMS FY2018"
    assert len(forward.corpus_hash) == len(backward.corpus_hash) == 64
