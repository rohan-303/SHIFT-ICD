from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/experiments/step10_fusion"
TABLES = ROOT / "reports/tables/step10_fusion"


def test_final_training_uses_exactly_two_epochs_and_no_dev_selection() -> None:
    manifest = json.loads((OUT / "final_seed_manifest.json").read_text())
    assert all(seed["epochs_completed"] == 2 for seed in manifest["seeds"])
    assert manifest["official_dev_scoring_count"] == 0
    assert manifest["test_access_count"] == 0


def test_final_scaler_is_full_train_only() -> None:
    scaler = json.loads((OUT / "final_train_scaler_manifest.json").read_text())
    assert scaler["source_count"] == 10197
    assert scaler["candidate_row_count"] == 1019700
    assert scaler["fit_split"] == "TRAIN"
    assert scaler["dev_rows_used"] == 0
    assert scaler["test_rows_used"] == 0


def test_official_dev_lock_chronology_and_access_counts() -> None:
    audit = json.loads((OUT / "official_dev_access_audit.json").read_text())
    assert audit["lock_before_feature"] is True
    assert audit["lock_before_score"] is True
    assert audit["test_feature_extraction_count"] == 0
    assert audit["test_scoring_count"] == 0
    assert audit["test_training_count"] == 0


def test_canonical_official_dev_population_and_mrr_contract() -> None:
    results = list(__import__("csv").DictReader((TABLES / "official_dev_seed_results.csv").open()))
    structural = list(__import__("csv").DictReader((TABLES / "official_dev_structural.csv").open()))
    assert len(results) == 3
    assert all(float(row["P_COMPLEX_CompleteScenarioRetrieval@10"]) >= 0 for row in results)
    assert all(float(row["P_COMPLEX_ChoiceListRecall@100"]) <= 1 for row in structural)
    baseline = list(__import__("csv").DictReader((TABLES / "official_dev_baseline.csv").open()))[0]
    assert float(baseline["MRR"]) >= 0


def test_candidate_invariance_and_replication_classification_are_frozen() -> None:
    manifest = json.loads((OUT / "official_dev_confirmation_manifest.json").read_text())
    assert manifest["replication_classification"] in {
        "OFFICIAL_DEV_SIGNAL_REPLICATES",
        "OFFICIAL_DEV_MIXED",
        "OFFICIAL_DEV_FAILS_TO_REPLICATE",
        "INVALID_FUSION_EXPERIMENT",
    }
    assert manifest["test_access_counts"] == {"features": 0, "scoring": 0, "training": 0}
    rows = list(__import__("csv").DictReader((TABLES / "candidate_invariance_official_dev.csv").open()))
    assert all(int(row["candidate_mutation_count"]) == 0 for row in rows)
    assert all(row["Hit@100_invariant"] == "True" for row in rows)
    assert all(row["structural_at_100_invariant"] == "True" for row in rows)
