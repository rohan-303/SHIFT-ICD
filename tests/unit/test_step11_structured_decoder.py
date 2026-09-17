# ruff: noqa: E501
from __future__ import annotations

import pytest

from shift_icd.data.schemas import GemChoiceAlternative, GemChoiceList, GemScenario, GemSourceMapping
from shift_icd.structured_decoder import (
    StructuredDecoderLock,
    StructuredSetDecoder,
    assemble_structure,
    canonicalize_gold,
    checkpoint_path,
    deserialize_structure,
    serialize_structure,
    source_balanced_loss,
)


def alt(code: str, approximate: bool = False) -> GemChoiceAlternative:
    return GemChoiceAlternative(target_code=code, raw_row_ids=[f"r-{code}"], approximate=approximate)


def mapping(kind: str, scenarios: list[GemScenario], targets: list[str] | None = None) -> GemSourceMapping:
    return GemSourceMapping(
        source_code="S1", direction="ICD9CM_TO_ICD10CM", source_version="ICD-9-CM", target_version="ICD-10-CM", mapping_kind=kind, raw_row_ids=["r"], scenarios=scenarios,
        target_row_count=len(targets or []), alternative_count=len(targets or []), required_component_count=max((len(s.choice_lists) for s in scenarios), default=0),
        scenario_count=len(scenarios), choice_list_count=sum(len(s.choice_lists) for s in scenarios), unique_target_count=len(set(targets or [])), valid_mapping_set_count=1,
        approximate_any=False, approximate_all=False, no_map=kind == "NO_MAP", combination=kind.startswith("COMBINATION"), eligible_simple_mapping=kind == "SINGLE_EXACT",
        eligible_approximate_mapping=False, eligible_no_map=kind == "NO_MAP", eligible_alternative_mapping=kind == "ALTERNATIVE", eligible_combination_mapping=kind.startswith("COMBINATION"),
        eligible_multiscenario_mapping=len(scenarios) > 1, description_available=False, unusually_large_mapping_structure=False,
    )


def test_combination_round_trip_preserves_choice_lists() -> None:
    gold = mapping("COMBINATION_WITH_ALTERNATIVES", [GemScenario(scenario_id=1, choice_lists=[GemChoiceList(choice_list_id=1, alternatives=[alt("A"), alt("B")]), GemChoiceList(choice_list_id=2, alternatives=[alt("C")])], raw_row_ids=["r"], approximate=False)])
    structure = canonicalize_gold(gold)
    assert structure["mapping_form"] == "COMBINATION_WITH_ALTERNATIVES"
    assert structure["scenarios"][0]["choice_lists"][0]["alternatives"] == ["A", "B"]
    assert deserialize_structure(serialize_structure(structure)) == structure


def test_no_map_is_empty_and_assembler_cannot_emit_targets() -> None:
    structure = canonicalize_gold(mapping("NO_MAP", [], []))
    assert structure["scenarios"] == []
    assert assemble_structure(structure, {"A": 99.0})["mapping_form"] == "NO_MAP"
    assert assemble_structure(structure, {"A": 99.0})["targets"] == []


def test_oracle_assembler_reconstructs_representable_structure() -> None:
    gold = mapping("COMBINATION", [GemScenario(scenario_id=1, choice_lists=[GemChoiceList(choice_list_id=1, alternatives=[alt("A")]), GemChoiceList(choice_list_id=2, alternatives=[alt("B")])], raw_row_ids=["r"], approximate=False)])
    structure = canonicalize_gold(gold)
    result = assemble_structure(structure, {"A": 2.0, "B": 1.0})
    assert result["targets"] == ["A", "B"]


def test_source_balanced_loss_averages_sources_not_labels() -> None:
    assert source_balanced_loss([[1.0], [3.0, 5.0]]) == pytest.approx(2.5)


def test_step11_locks_fail_closed() -> None:
    lock = StructuredDecoderLock()
    with pytest.raises(RuntimeError, match="STEP11_OFFICIAL_DEV_ACCESS_FORBIDDEN"):
        lock.assert_allowed("official_dev")
    with pytest.raises(RuntimeError, match="STEP11_TEST_ACCESS_FORBIDDEN"):
        lock.assert_allowed("test")
    lock.assert_allowed("fusion_val_smoke")


def test_decoder_forward_is_deterministic_and_has_frozen_heads() -> None:
    import torch
    torch.manual_seed(17)
    model = StructuredSetDecoder(3, 8)
    features = torch.ones((2, 4, 3))
    first = model(features)
    second = model(features)
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert set(first) == {"form_logits", "cardinality_logits", "membership_logits"}


def test_checkpoint_name_encodes_configuration_seed_and_epoch() -> None:
    assert checkpoint_path("primary", 17, 1) == "step11_primary_seed_17_epoch_1.pt"
    with pytest.raises(ValueError, match="STEP11_CHECKPOINT_COLLISION"):
        checkpoint_path("", 17, 1)
