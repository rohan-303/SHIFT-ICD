from __future__ import annotations

import csv
import json
from pathlib import Path

from shift_icd.evaluation.metric_contract import classify_population, population_counts

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "artifacts/experiments/step9_hierarchy/structural_population_audit.json"
TABLES = ROOT / "reports/tables/step9_hierarchy"


def test_r3a_canonical_complex_population_is_frozen() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["populations"]["P_COMBINATION"]["count"] == 64
    assert audit["populations"]["P_COMBINATION_WITH_ALTERNATIVES"]["count"] == 69
    assert audit["populations"]["P_COMPLEX"]["count"] == 133
    assert audit["r3_original_population"]["count"] == 78
    assert audit["r3_original_population"]["intersection_with_canonical"] == 26


def test_metric_contract_does_not_filter_complex_sources_by_gold_presence() -> None:
    class Example:
        def __init__(self, kind: str) -> None:
            self.mapping_kind = kind
            self.no_map = False

    examples = [Example("COMBINATION"), Example("COMBINATION_WITH_ALTERNATIVES"), Example("SINGLE_EXACT")]
    counts = population_counts(examples)
    assert counts["P_COMPLEX"] == 2
    assert classify_population(examples[0]) == "P_COMBINATION"
    assert classify_population(examples[1]) == "P_COMBINATION_WITH_ALTERNATIVES"


def test_r3a_at100_and_bootstrap_use_canonical_population() -> None:
    with (TABLES / "r3a_structural_metrics_corrected.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    at100 = [row for row in rows if row["k"] == "100"]
    assert {row["n"] for row in at100} == {"133"}
    assert all(float(row["ChoiceListRecall"]) == 0.7323308270676692 for row in at100)
    assert all(float(row["CompleteScenarioRetrieval"]) == 0.48872180451127817 for row in at100)

    with (TABLES / "r3a_b0_vs_hierarchy_structural_bootstrap.csv").open(encoding="utf-8") as stream:
        bootstrap = list(csv.DictReader(stream))
    assert {row["population_n"] for row in bootstrap} == {"133"}
    assert {row["repetitions"] for row in bootstrap} == {"10000"}
    assert {row["seed"] for row in bootstrap} == {"20260915"}
