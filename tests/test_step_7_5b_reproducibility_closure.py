from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/experiments/dense_full_universe_v2"
TABLES = ROOT / "reports/tables/dense_full_universe_v2"
EXPECTED = {
    "forward": (
        71704,
        "8ca0d0cf0213581645fe64d9c1e599552a2fb207eb740f32ca09ffeaaf9f8e26",
        "32f572abeb2ff4cb003145216fa83bfd9984fab96579205f432993aa9c550464",
    ),
    "backward": (
        14567,
        "204be6e29572332f3141a18347a28e95f2b258e5bacec8cf98046974adbc2eb5",
        "dcfe26da33d422a2395081fffaaf83e29d2f557d33458ade392d83b14e171b3b",
    ),
}
REQUIRED_TABLES = [
    "model_registry.csv", "forward_dev_ordinary.csv", "neural_reselection.csv",
    "forward_test_ordinary.csv", "forward_test_structural.csv", "forward_family_ordinary.csv",
    "forward_family_structural.csv", "backward_stratified_ordinary.csv",
    "backward_stratified_structural.csv", "backward_family_ordinary.csv",
    "backward_family_structural.csv", "rrf_provenance.csv", "no_map_diagnostics.csv",
    "selected_biolord_failure_slices.csv", "bootstrap_biolord_vs_bm25.csv", "runtime_original.csv",
    "remote_code_provenance.csv", "cache_integrity.csv",
]


def test_dense_result_manifests_require_revision_corpus_and_code_provenance() -> None:
    registry = json.loads((ART / "model_registry.json").read_text())
    assert all(m.get("revision") for m in registry["models"])
    assert all(m.get("model_id") for m in registry["models"])
    assert (TABLES / "remote_code_provenance.csv").exists()
    for p in (ART / "remote_sync_corrected").glob("gpu*/results/dense_full_universe_v2/*/embeddings/*.json"):
        d = json.loads(p.read_text())
        assert d["retrieval_corpus_hash"]
        assert d["revision"]


def test_selected_biolord_reconstruction() -> None:
    selection = json.loads((ART / "neural_reselection.json").read_text())
    assert "BioLORD-2023" in json.dumps(selection)
    rows = list(csv.DictReader((TABLES / "forward_dev_ordinary.csv").open()))
    best = max(rows, key=lambda r: (float(r["Hit@10"]), float(r["MRR"]), float(r["Hit@100"])))
    assert best["Model"] == "BioLORD-2023"


def test_test_lock_chronology_and_provenance() -> None:
    lock = json.loads((ART / "test_lock.json").read_text())
    assert lock["evaluator"] == "retrieval_evaluator_v3"
    assert lock["test_used_for_selection"] is False
    assert (ART / "test_lock.json").stat().st_mtime < (ART / "remote_test_sync/results/test_results/all_metrics.json").stat().st_mtime
    assert hashlib.sha256((ART / "test_lock.json").read_bytes()).hexdigest() == (ART / "test_lock.sha256").read_text().split()[0]


def test_corrected_rrf_inputs() -> None:
    rrf = json.loads((ART / "rrf_dev.json").read_text())
    lock = json.loads((ART / "test_lock.json").read_text())
    assert lock["rrf"]["constant"] == 60
    assert set(rrf) >= {"BioLORD-2023", "SapBERT", "MedCPT", "Qwen3-Embedding-0.6B"}
    assert "legacy" not in json.dumps(rrf).lower()


def test_structural_tables_complete_and_bounded() -> None:
    names = [
        "forward_test_structural.csv",
        "forward_family_structural.csv",
        "backward_stratified_structural.csv",
        "backward_family_structural.csv",
    ]
    for name in names:
        rows = list(csv.DictReader((TABLES / name).open()))
        assert rows
        for row in rows:
            for prefix in ("ChoiceListRecall@", "CompleteScenarioRetrieval@"):
                vals = [float(row[f"{prefix}{k}"]) for k in (1, 5, 10, 25, 50, 100)]
                assert all(0 <= v <= 1 for v in vals)
                assert vals == sorted(vals)


def test_empty_structural_populations_are_not_fabricated() -> None:
    for p in TABLES.glob("*_structural.csv"):
        for row in csv.DictReader(p.open()):
            assert int(row["n"]) > 0 or all(row[f"ChoiceListRecall@{k}"] == "NOT_APPLICABLE" for k in (1, 5, 10, 25, 50, 100))


def test_bootstrap_population_alignment() -> None:
    boot = json.loads((ART / "bootstrap_biolord_vs_bm25.json").read_text())
    assert {v["n"] for v in boot.values()} == {2695}
    assert {v["seed"] for v in boot.values()} == {20260906}
    assert {v["replicates"] for v in boot.values()} == {10000}


def test_selected_biolord_cache_hash_validation() -> None:
    manifest = json.loads((ART / "selected_biolord_cache_manifest.json").read_text())
    assert manifest["model_id"] == "FremyCompany/BioLORD-2023"
    assert manifest["revision"] == "167aab527b238a50ca65224e6319215d2ff4fc9f"
    assert {e["target_count"] for e in manifest["embeddings"]} == {71704, 14567}
    assert all(len(e["array_sha256"]) == 64 for e in manifest["embeddings"])


def test_old_universe_cache_rejected() -> None:
    for p in (ART / "remote_sync_corrected").rglob("*.json"):
        text = p.read_text()
        assert "17513" not in text and "11690" not in text and "11689" not in text
    for p in (ART / "remote_sync_corrected").glob("gpu*/results/dense_full_universe_v2/*/embeddings/*.json"):
        d = json.loads(p.read_text())
        direction = "forward" if d["target_count"] == 71704 else "backward"
        assert d["target_count"] == EXPECTED[direction][0]
        assert d["code_hash"] == EXPECTED[direction][1]


def test_code_provenance_rp1_rp2_schema() -> None:
    rows = list(csv.DictReader((TABLES / "remote_code_provenance.csv").open()))
    assert rows[0]["checkout_type"] in {"RP1_EXACT_COMMITTED_CHECKOUT", "RP2_RECONSTRUCTABLE_PACKAGED_SNAPSHOT"}
    assert rows[0]["remote_head_recorded"]
    assert (ROOT.parent / "SHIFT-ICD-provenance/step_7_5b/source.tar.gz").exists()


def test_compact_export_completeness() -> None:
    assert all((TABLES / name).exists() for name in REQUIRED_TABLES)
    assert (ART / "test_lock.json").exists()
    assert (ART / "selected_biolord_cache_manifest.json").exists()


def test_blocker_closure_matrix_completeness() -> None:
    p = TABLES / "reproducibility_blocker_closure.csv"
    if p.exists():
        rows = list(csv.DictReader(p.open()))
        assert rows and all(r["final_status"] in {"CLOSED", "OPEN"} for r in rows)


def test_runtime_evidence_labels_unavailable_values() -> None:
    rows = list(csv.DictReader((TABLES / "runtime_original.csv").open()))
    assert {r["model"] for r in rows} == {"SapBERT", "BioLORD-2023", "MedCPT", "Qwen3-Embedding-0.6B"}
    assert all(r["p95_query_latency_ms"] == "NOT_RECORDED" for r in rows)


def test_core_model_cache_shapes_and_semantics() -> None:
    for p in (ART / "remote_sync_corrected").glob("gpu*/results/dense_full_universe_v2/*/embeddings/*.json"):
        d = json.loads(p.read_text())
        assert d["dtype"] == "float32"
        assert d["normalized"] is True
        assert d["similarity"] == "normalized_dot_product_exact"
        assert d["embedding_dim"] in {768, 1024}


def test_freeze_manifest_links_required_hashes_when_present() -> None:
    p = ART / "freeze_manifest.json"
    if p.exists():
        d = json.loads(p.read_text())
        for key in ("evaluator_version", "model_registry_hash", "test_lock_hash", "export_package_hash"):
            assert d.get(key)
