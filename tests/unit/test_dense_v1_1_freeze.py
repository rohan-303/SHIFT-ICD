from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/experiments/dense_v1_1"
TAB = ROOT / "reports/tables/dense_v1_1"
FIG = ROOT / "reports/figures/dense_v1_1"


def test_dense_v1_1_scope_metadata_and_direction_filtering() -> None:
    d = json.loads((ART / "scope_audit.json").read_text())
    assert not d["test_selection_used"]
    assert {x["sample_population"] for x in d["scope_audit"]} == {
        "forward_stratified_test",
        "forward_family_held_out_test",
        "backward_stratified_test",
        "backward_family_held_out_test",
    }
    assert all(x["ids_consistent"] for x in d["scope_audit"])


def test_dense_v1_1_split_partition_and_counts() -> None:
    for name, n in {
        "forward_stratified_overall.csv": 2913,
        "forward_family_held_out_overall.csv": 2908,
        "backward_stratified_overall.csv": 14341,
        "backward_family_held_out_overall.csv": 14332,
    }.items():
        d = pd.read_csv(TAB / name)
        assert set(d.n_total) == {n}
        assert set(d.benchmark_version) == {1.0}
        assert set(d.experiment_version) == {1.1}


def test_paired_delta_population_consistency_and_frozen_metric() -> None:
    d = pd.read_csv(TAB / "forward_paired_bootstrap.csv")
    assert set(d.applicable_n) == {2695}
    row = d[(d.model == "BioLORD-2023") & (d.comparison == "vs BM25") & (d.metric == "Hit@10")].iloc[0]
    assert abs(row.delta - 0.1398886827458256) < 1e-12
    assert "non-combination" in row.delta_definition


def test_prediction_ledger_serialization_and_pairing() -> None:
    d = pd.read_csv(TAB / "forward_stratified_prediction_ledger.csv")
    required = {
        "benchmark_id",
        "system",
        "top1_target",
        "top1_score",
        "best_valid_target_rank",
        "Hit@1",
        "Hit@5",
        "Hit@10",
        "Hit@25",
        "Hit@50",
        "Hit@100",
        "MRR",
        "mapping_kind",
        "lexical_difficulty",
        "alternative_size_bucket",
    }
    assert required <= set(d.columns)
    assert len(d) == 17478
    assert d.groupby("system").benchmark_id.nunique().eq(2913).all()


def test_complementarity_rrf_and_lexical_slices() -> None:
    c = pd.read_csv(TAB / "forward_bm25_dense_complementarity.csv")
    assert set(c[c.model == "BioLORD-2023"].k) >= {10, 100}
    assert set(c.slice.dropna()) >= {
        "LEXICAL_EXACT",
        "LEXICAL_HIGH",
        "LEXICAL_MEDIUM",
        "LEXICAL_LOW",
        "SINGLE_EXACT",
        "SINGLE_APPROXIMATE",
        "ALTERNATIVE",
    }
    r = pd.read_csv(TAB / "forward_rrf_failure_analysis.csv")
    assert set(r.fixed_rrf_k) == {60}


def test_combination_and_multiscenario_audit() -> None:
    c = pd.read_csv(TAB / "forward_stratified_combination.csv")
    assert {"ChoiceListRecall@1", "CompleteScenarioRetrieval@100"} <= set(c.columns)
    m = pd.read_csv(TAB / "multi_scenario_audit.csv")
    assert set(m.n_multi_scenario) == {0}


def test_dev_selection_model_revision_license_and_runtime_schema() -> None:
    d = json.loads((ART / "dev_selection_audit.json").read_text())
    assert d["selected_model"] == "BioLORD-2023"
    assert d["test_performance_used"] is False
    p = pd.read_csv(TAB / "dense_model_provenance.csv")
    assert len(p) == 4 and p.revision.str.len().eq(40).all()
    assert "license" in p.columns
    r = pd.read_csv(TAB / "dense_runtime.csv")
    assert {"model_load_seconds", "query_encoding_seconds", "gpu_peak_allocated_bytes"} <= set(r.columns)


def test_required_tables_figures_and_test_boundary() -> None:
    assert len(list(TAB.glob("*.csv"))) >= 22
    assert len(list(FIG.glob("figure_*.png"))) == 9
    boundary = (ROOT / "docs/experiments/shift_map_v1_training_boundary.md").read_text().lower()
    for marker in [
        "forward stratified train",
        "forward stratified dev",
        "forward stratified test",
        "n1",
        "n6",
        "must never use dev/test gold",
    ]:
        assert marker in boundary


def test_no_model_update_and_frozen_versions() -> None:
    manifest = json.loads((ART / "manifest.json").read_text())
    assert manifest["source_experiment"] == "dense_v1"
    assert manifest["no_model_rerun"] is True
    assert json.loads((ART / "frozen_metrics.json").read_text())["experiment_version"] == "1.1"
