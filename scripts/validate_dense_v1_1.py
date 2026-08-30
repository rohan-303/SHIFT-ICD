from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/experiments/dense_v1_1"
TAB = ROOT / "reports/tables/dense_v1_1"
EXPECTED = {
    "forward_stratified_test": 2913,
    "forward_family_held_out_test": 2908,
    "backward_stratified_test": 14341,
    "backward_family_held_out_test": 14332,
}
REQUIRED_TABLES = {
    "forward_stratified_overall.csv",
    "forward_family_held_out_overall.csv",
    "backward_stratified_overall.csv",
    "backward_family_held_out_overall.csv",
    "forward_stratified_by_mapping_kind.csv",
    "forward_stratified_by_lexical_difficulty.csv",
    "forward_stratified_by_alternative_size.csv",
    "forward_stratified_combination.csv",
    "forward_stratified_no_map_diagnostics.csv",
    "forward_bm25_dense_complementarity.csv",
    "forward_oracle_union.csv",
    "forward_paired_bootstrap.csv",
    "forward_family_comparison.csv",
    "backward_family_comparison.csv",
    "forward_backward_comparison.csv",
    "dense_runtime.csv",
    "dense_model_provenance.csv",
    "dense_dev_selection.csv",
    "forward_stratified_prediction_ledger.csv",
}


def main() -> None:
    scope = json.loads((ART / "scope_audit.json").read_text())
    assert all(x["ids_consistent"] and EXPECTED[x["sample_population"]] == x["n"] for x in scope["scope_audit"])
    assert not scope["test_selection_used"]
    assert REQUIRED_TABLES <= {p.name for p in TAB.glob("*.csv")}
    primary = pd.read_csv(TAB / "forward_stratified_overall.csv")
    bio = primary[primary.model == "BioLORD-2023"].iloc[0]
    assert bio["n_total"] == 2913 and bio["n_answerable"] == 2828 and bio["applicable_hit_k_n"] == 2695
    assert abs(bio["Hit@10"] - 0.9213358070500928) < 1e-12
    assert abs(bio["Hit@100"] - 0.9840445269016698) < 1e-12
    ledger = pd.read_csv(TAB / "forward_stratified_prediction_ledger.csv")
    assert len(ledger) == 2913 * 6 and set(ledger.system) == {
        "BM25",
        "SapBERT",
        "BioLORD-2023",
        "MedCPT",
        "Qwen3-Embedding-0.6B",
        "BM25 + BioLORD RRF",
    }
    assert set(ledger.benchmark_id[ledger.system == "BioLORD-2023"]) == set(ledger.benchmark_id[ledger.system == "BM25"])
    comp = pd.read_csv(TAB / "forward_bm25_dense_complementarity.csv")
    assert set(comp[comp.model == "BioLORD-2023"].k) == {10, 100}
    prov = pd.read_csv(TAB / "dense_model_provenance.csv")
    assert len(prov) == 4 and all(prov.revision.str.len() == 40) and all(prov.embedding_dim > 0)
    runtime = pd.read_csv(TAB / "dense_runtime.csv")
    assert len(runtime) == 4 and all(
        c in runtime.columns for c in ["model_load_seconds", "query_encoding_seconds", "gpu_peak_allocated_bytes"]
    )
    ms = pd.read_csv(TAB / "multi_scenario_audit.csv")
    assert set(ms.n_multi_scenario) == {0}
    boundary = (ROOT / "docs/experiments/shift_map_v1_training_boundary.md").read_text().lower()
    for text in [
        "forward stratified train",
        "forward stratified dev",
        "forward stratified test",
        "combination",
        "n1",
        "n6",
        "must never use dev/test gold",
    ]:
        assert text in boundary
    print(
        json.dumps(
            {
                "status": "passed",
                "tables": len(list(TAB.glob("*.csv"))),
                "figures": len(list((ROOT / "reports/figures/dense_v1_1").glob("*.png"))),
                "ledger_rows": len(ledger),
                "populations": EXPECTED,
            }
        )
    )


if __name__ == "__main__":
    main()
