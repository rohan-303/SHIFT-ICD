from __future__ import annotations

import json

import pytest

from shift_icd.data.schemas import (
    GemChoiceAlternative,
    GemChoiceList,
    GemScenario,
    GemSourceMapping,
)
from shift_icd.evaluation.gold import (
    best_matching_scenario,
    choice_list_coverage,
    component_f1,
    exact_mapping_match,
    is_valid_complete_mapping,
    mapping_completion_fraction,
)
from shift_icd.evaluation.lexical import lexical_features
from shift_icd.evaluation.slices import assign_slices
from shift_icd.evaluation.splits import extract_source_family, split_source_objects


def alternative(code: str, choice: int = 1) -> GemChoiceAlternative:
    return GemChoiceAlternative(target_code=code, raw_row_ids=[f"r-{code}"], approximate=False)


def source(*scenarios: GemScenario, kind: str = "COMBINATION_WITH_ALTERNATIVES") -> GemSourceMapping:
    return GemSourceMapping(
        source_code="001",
        source_label="source concept",
        direction="ICD9CM_TO_ICD10CM",
        source_version="ICD-9-CM",
        target_version="ICD-10-CM",
        mapping_kind=kind,
        raw_row_ids=["row-1"],
        scenarios=list(scenarios),
        target_row_count=4,
        alternative_count=4,
        required_component_count=2,
        scenario_count=len(scenarios),
        choice_list_count=sum(len(s.choice_lists) for s in scenarios),
        unique_target_count=4,
        valid_mapping_set_count=4,
        approximate_any=False,
        approximate_all=False,
        no_map=kind == "NO_MAP",
        combination=kind.startswith("COMBINATION"),
        eligible_simple_mapping=False,
        eligible_approximate_mapping=False,
        eligible_no_map=kind == "NO_MAP",
        eligible_alternative_mapping=False,
        eligible_combination_mapping=kind.startswith("COMBINATION"),
        eligible_multiscenario_mapping=len(scenarios) > 1,
        description_available=True,
        unusually_large_mapping_structure=False,
    )


def test_combination_validator_requires_every_choice_list():
    gold = source(
        GemScenario(
            scenario_id=1,
            choice_lists=[
                GemChoiceList(choice_list_id=1, alternatives=[alternative("A"), alternative("B")]),
                GemChoiceList(choice_list_id=2, alternatives=[alternative("C"), alternative("D")]),
            ],
            raw_row_ids=["row-1"],
            approximate=False,
        )
    )
    assert is_valid_complete_mapping(gold, {"A", "C"})
    assert not is_valid_complete_mapping(gold, {"A"})
    assert choice_list_coverage(gold, {"A"}) == 0.5
    assert component_f1(gold, {"A", "C"}) == pytest.approx(1.0)
    assert exact_mapping_match(gold, {"A", "C"})


def test_alternatives_and_multiple_scenarios_are_symbolic():
    gold = source(
        GemScenario(
            scenario_id=1,
            choice_lists=[GemChoiceList(choice_list_id=1, alternatives=[alternative("A")])],
            raw_row_ids=["r1"],
            approximate=False,
        ),
        GemScenario(
            scenario_id=2,
            choice_lists=[GemChoiceList(choice_list_id=1, alternatives=[alternative("B")])],
            raw_row_ids=["r2"],
            approximate=False,
        ),
    )
    assert best_matching_scenario(gold, {"B"}) == 2
    assert is_valid_complete_mapping(gold, {"B"})

    alternatives = source(kind="ALTERNATIVE")
    assert not is_valid_complete_mapping(alternatives, set())


def test_no_map_is_not_a_target_mapping():
    no_map = source(kind="NO_MAP")
    no_map.no_map = True
    assert not is_valid_complete_mapping(no_map, {"A"})
    assert mapping_completion_fraction(no_map, {"A"}) == 0.0


def test_family_extraction_and_deterministic_splits():
    assert extract_source_family("123.45", "ICD-9-CM") == "123"
    assert extract_source_family("V54.12", "ICD-9-CM") == "V54"
    assert extract_source_family("E1234", "ICD-9-CM") == "E12"
    assert extract_source_family("A00.0", "ICD-10-CM") == "A00"
    objects = [
        {"benchmark_id": f"b{i}", "source_code": f"{i:03d}", "source_family": f"{i:03d}", "mapping_kind": "SINGLE_EXACT"} for i in range(20)
    ]
    first = split_source_objects(objects, seed=17, ratios=(0.7, 0.1, 0.2), group_key="source_code")
    second = split_source_objects(objects, seed=17, ratios=(0.7, 0.1, 0.2), group_key="source_code")
    assert first == second
    assert set(first) == {"train", "dev", "test"}
    assert sum(len(v) for v in first.values()) == 20


def test_lexical_features_and_slices_are_deterministic():
    features = lexical_features("Fracture of lower arm", "Fracture of lower arm")
    assert features["exact_string_match"] is True
    assert features["token_overlap"] == 1.0
    assert assign_slices(
        {
            "mapping_kind": "ALTERNATIVE",
            "unique_target_count": 533,
            "scenario_count": 0,
            "choice_list_count": 0,
            "valid_mapping_set_count": 533,
        }
    )["ALTERNATIVE_EXTREME"]


def test_serialization_round_trip():
    payload = source().model_dump(mode="json")
    assert GemSourceMapping.model_validate(json.loads(json.dumps(payload))).source_code == "001"
