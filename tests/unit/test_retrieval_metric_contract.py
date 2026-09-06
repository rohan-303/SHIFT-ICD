# ruff: noqa: E501
from types import SimpleNamespace

import pytest

from shift_icd.evaluation.metric_contract import (
    ORDINARY_KINDS,
    classify_population,
    metric_values,
    population_counts,
)
from shift_icd.evaluation.retrieval import choice_list_recall_at_k, complete_scenario_retrieval_at_k


def ex(kind: str, no_map: bool = False) -> SimpleNamespace:
    return SimpleNamespace(mapping_kind=kind, no_map=no_map)


@pytest.fixture
def track_mapping():
    from shift_icd.data.schemas import GemChoiceAlternative, GemChoiceList, GemScenario, GemSourceMapping

    def alt(code: str) -> GemChoiceAlternative:
        return GemChoiceAlternative(target_code=code, raw_row_ids=[code], approximate=False)

    return GemSourceMapping(
        source_code="001", source_label="source", direction="ICD9CM_TO_ICD10CM", source_version="9", target_version="10",
        mapping_kind="COMBINATION", raw_row_ids=["r"], scenarios=[GemScenario(scenario_id=1, choice_lists=[GemChoiceList(choice_list_id=1, alternatives=[alt("A")]), GemChoiceList(choice_list_id=2, alternatives=[alt("C")])], raw_row_ids=["r"], approximate=False)],
        target_row_count=2, alternative_count=2, required_component_count=2, scenario_count=1, choice_list_count=2, unique_target_count=2, valid_mapping_set_count=1,
        approximate_any=False, approximate_all=False, no_map=False, combination=True, eligible_simple_mapping=False, eligible_approximate_mapping=False,
        eligible_no_map=False, eligible_alternative_mapping=False, eligible_combination_mapping=True, eligible_multiscenario_mapping=False,
        description_available=True, unusually_large_mapping_structure=False,
    )


def test_population_contract_is_mutually_exclusive() -> None:
    examples = [ex("SINGLE_EXACT"), ex("ALTERNATIVE"), ex("COMBINATION"), ex("COMBINATION_WITH_ALTERNATIVES"), ex("NO_MAP", True)]
    assert ORDINARY_KINDS == {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"}
    assert [classify_population(item) for item in examples] == ["P_ORDINARY_ANSWERABLE", "P_ORDINARY_ANSWERABLE", "P_COMBINATION", "P_COMBINATION_WITH_ALTERNATIVES", "P_NO_MAP"]
    assert population_counts(examples) == {"P_ALL": 5, "P_ORDINARY_ANSWERABLE": 2, "P_COMBINATION": 1, "P_COMBINATION_WITH_ALTERNATIVES": 1, "P_COMPLEX": 2, "P_NO_MAP": 1}


def test_structural_metrics_require_all_components_and_are_monotone(track_mapping) -> None:
    ranked = ["A", "noise", "C"]
    values = [choice_list_recall_at_k(track_mapping, ranked, k) for k in (1, 2, 3)]
    complete = [complete_scenario_retrieval_at_k(track_mapping, ranked, k) for k in (1, 2, 3)]
    assert values == [0.5, 0.5, 1.0]
    assert complete == [False, False, True]
    assert all(a <= b for a, b in zip(values, values[1:]))  # noqa: B905
    assert all(a <= b for a, b in zip(complete, complete[1:]))  # noqa: B905


def test_metric_values_exclude_no_map_and_keep_source_level_alternatives() -> None:
    assert metric_values({"A", "B"}, ["x", "B"], no_map=False, k_values=(1, 2)) == {"Hit@1": 0.0, "Hit@2": 1.0, "MRR": 0.5}
    assert metric_values(set(), ["x"], no_map=True, k_values=(1, 2)) == {"Hit@1": None, "Hit@2": None, "MRR": None}


def test_corrected_dense_corpora_use_full_authoritative_universes() -> None:
    import pandas as pd

    from scripts.run_dense_full_universe_v2 import BACKWARD, FORWARD, build_corpora

    corpora = build_corpora(pd.DataFrame())
    assert len(corpora[FORWARD]) == 71_704
    assert len(corpora[BACKWARD]) == 14_567


def test_exact_dense_rank_returns_deterministic_top_100() -> None:
    import numpy as np

    from scripts.run_dense_full_universe_v2 import dense_rank

    matrix = np.eye(120, dtype=np.float32)
    ranked, scores = dense_rank(matrix[0], matrix, [f"C{i:03d}" for i in range(120)])
    assert len(ranked) == 100
    assert ranked[0] == "C000"
    assert scores[0] == 1.0
