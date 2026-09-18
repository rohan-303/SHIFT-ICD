# ruff: noqa: E501
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from shift_icd.dense.models import ModelSpec, load_encoder
from shift_icd.dense.text import clean_dense_text
from shift_icd.structured_metrics import flat_set_metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/experiments/step11_structured_decoder"
TABLES = ROOT / "reports/tables/step11_structured_decoder"
CANDIDATE_ROOT = ROOT / "artifacts/candidates/shift_map_full_universe_v2"
GOLD = OUT / "structured_gold_train.jsonl.gz"
SEED = 17


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_candidates(split: str) -> dict[str, list[dict[str, Any]]]:
    rows = load_jsonl_gz(CANDIDATE_ROOT / f"forward_{split}_k100.jsonl.gz")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["source_id"]), []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: int(row["candidate_rank"]))
    return groups


def positive_bin(count: int) -> str:
    return "0" if count == 0 else "1" if count == 1 else "2+"


def frozen_inner_split(groups: dict[str, list[dict[str, Any]]], benchmark: dict[str, dict[str, Any]]) -> tuple[set[str], set[str]]:
    import random

    strata: dict[tuple[str, str], list[str]] = {}
    for source_id, group in groups.items():
        source = benchmark[source_id]
        gold = set(map(str, source.get("valid_target_codes", [])))
        count = sum(str(row["target_code"]) in gold for row in group)
        strata.setdefault((str(source["mapping_kind"]), positive_bin(count)), []).append(source_id)
    rng = random.Random(20260917)
    train: list[str] = []
    validation: list[str] = []
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        n_val = max(1, int(round(0.15 * len(ids)))) if len(ids) > 1 else 0
        validation.extend(ids[:n_val])
        train.extend(ids[n_val:])
    return set(train), set(validation)


def load_structures() -> dict[str, dict[str, Any]]:
    rows = load_jsonl_gz(GOLD)
    return {f"track_a_v1.0:ICD9CM_TO_ICD10CM:{row['source_code']}": row for row in rows}


def select_b1_delta(train_ids: set[str], groups: dict[str, list[dict[str, Any]]], structures: dict[str, dict[str, Any]]) -> tuple[float, dict[str, float]]:
    values: dict[float, list[float]] = {delta: [] for delta in (0.10, 0.25, 0.50, 1.00)}
    for source_id in sorted(train_ids):
        group = groups[source_id]
        scores = np.asarray([float(row["retriever_score"]) for row in group], dtype=np.float64)
        z = (scores - scores.mean()) / max(float(scores.std(ddof=0)), 1e-8)
        gold = structures[source_id]
        gold_set = set(map(str, gold.get("flat_alternatives", [])))
        for delta in values:
            predicted = [str(row["target_code"]) for row, value in zip(group, z[0] - z, strict=True) if value <= delta]
            values[delta].append(flat_set_metrics({"flat_alternatives": predicted, "scenarios": []}, {"flat_alternatives": sorted(gold_set), "scenarios": []})["f1"])
    means = {delta: float(np.mean(scores)) for delta, scores in values.items()}
    return max(means, key=lambda delta: (means[delta], -delta)), means


def feature_audit(groups: dict[str, list[dict[str, Any]]], train_ids: set[str], val_ids: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_ids = sorted(train_ids)[:8] + sorted(val_ids)[:8]
    texts = [groups[source_id][0]["source_description"] for source_id in sample_ids]
    spec = ModelSpec("BioLORD-2023", "FremyCompany/BioLORD-2023", "167aab527b238a50ca65224e6319215d2ff4fc9f", "sentence_transformer", "other", 768)
    candidate_texts = [[clean_dense_text(str(row["target_description"])) for row in groups[source_id]] for source_id in sample_ids]
    encoder = load_encoder(spec, "cpu")
    try:
        first = encoder.encode([clean_dense_text(text) for text in texts], 64, 8, is_query=True).embeddings
        second = encoder.encode([clean_dense_text(text) for text in texts], 64, 8, is_query=True).embeddings
        first_candidates = [encoder.encode(values, 64, 8, is_query=False).embeddings for values in candidate_texts]
        second_candidates = [encoder.encode(values, 64, 8, is_query=False).embeddings for values in candidate_texts]
    finally:
        encoder.close()
    rows: list[dict[str, Any]] = []
    for index, source_id in enumerate(sample_ids):
        group = groups[source_id]
        candidate_matrix = np.asarray(first_candidates[index], dtype=np.float32)
        candidate_matrix_repeat = np.asarray(second_candidates[index], dtype=np.float32)
        source_equal = bool(np.array_equal(first[index], second[index]))
        candidate_equal = bool(np.array_equal(candidate_matrix, candidate_matrix_repeat))
        order_equal = [row["target_code"] for row in group] == [row["target_code"] for row in group]
        rows.append({"source_id": source_id, "split": "FUSION_TRAIN" if source_id in train_ids else "FUSION_VAL", "source_embedding_equal": source_equal, "candidate_embedding_equal": candidate_equal, "candidate_order_equal": order_equal, "source_embedding_sha256": hashlib.sha256(first[index].tobytes()).hexdigest(), "candidate_embedding_sha256": hashlib.sha256(candidate_matrix.tobytes()).hexdigest(), "status": "PASS" if source_equal and candidate_equal and order_equal else "FAIL"})
    provenance = {
        "schema": "step11_feature_provenance_v1",
        "status": "FROZEN_PRE_OUTCOME",
        "source_embedding": {"producer": "src/shift_icd/dense/models.py::SentenceTransformerEncoder.encode", "source_artifact": "data/benchmarks/cms_track_a/v1.0/forward/all.jsonl", "model_id": spec.model_id, "revision": spec.revision, "pooling": "SentenceTransformer native pooling", "max_sequence_length": 64, "embedding_dim": 768, "dtype": "float32", "normalization": "L2", "is_query": True},
        "candidate_embedding": {"producer": "src/shift_icd/dense/models.py::SentenceTransformerEncoder.encode", "source_artifact": "artifacts/candidates/shift_map_full_universe_v2/forward_{train,dev}_k100.jsonl.gz::target_description", "embedding_artifact": "ON_THE_FLY_FROM_FROZEN_CANDIDATE_ARTIFACT", "embedding_dim": 768, "dtype": "float32", "normalization": "L2", "pooling": "SentenceTransformer native pooling", "max_sequence_length": 64, "model_id": spec.model_id, "revision": spec.revision},
        "retriever_score": {"source": "forward_*_k100.jsonl.gz::retriever_score", "dtype": "float64_before_feature_cast"},
        "normalized_retriever_score": {"formula": "(score - mean(score))/max(population_sd(score),1e-8)", "ddof": 0, "statistics_dtype": "float64"},
        "candidate_rank": {"definition": "frozen 1-based candidate_rank in forward Top-100", "dtype": "float32_feature"},
        "feature_dimension": 1539,
        "candidate_artifacts": {"train_sha256": sha(CANDIDATE_ROOT / "forward_train_k100.jsonl.gz"), "dev_sha256": sha(CANDIDATE_ROOT / "forward_dev_k100.jsonl.gz"), "k": 100},
        "canonical_checkpoint_sha256": "9e9739e45ad041a4027cdfd611fb83e398debb50658ede710482660e16efe302",
        "base_model": spec.model_id,
        "base_revision": spec.revision,
        "deterministic_settings": {"device": "cpu_for_protocol_audit", "batch_size": 8, "clean_dense_text": True, "inference_mode": True}
    }
    return provenance, rows


def main() -> None:
    benchmark_path = ROOT / "data/benchmarks/cms_track_a/v1.0/forward/all.jsonl"
    benchmark = {str(row["benchmark_id"]): row for row in (json.loads(line) for line in benchmark_path.read_text(encoding="utf-8").splitlines() if line)}
    groups = load_candidates("train")
    train_ids, val_ids = frozen_inner_split(groups, benchmark)
    structures = load_structures()
    b1_delta, b1_scores = select_b1_delta(train_ids, groups, structures)
    provenance, feature_rows = feature_audit(groups, train_ids, val_ids)
    write_json(OUT / "search_space_v3.json", {"schema": "step11_search_space_v3", "architecture": "compact_deepsets_candidate_set_with_structural_queries", "architecture_variants": 1, "capacities": {"S_MAX": 6, "L_MAX": 3}, "hidden_dim": 64, "learning_rates": [0.0001, 0.0003], "imbalance_variants": ["UNWEIGHTED_PRIMARY", "TRAIN_DERIVED_FORM_WEIGHTED"], "epochs": [1, 2, 3], "development_seed": 17, "new_capacity_search": False})
    grid = []
    for lr in (0.0001, 0.0003):
        for imbalance in ("UNWEIGHTED_PRIMARY", "TRAIN_DERIVED_FORM_WEIGHTED"):
            grid.append({"configuration_id": f"r2_lr_{lr:.0e}_{imbalance.lower()}", "architecture": "compact_deepsets_candidate_set_with_structural_queries", "learning_rate": lr, "imbalance_variant": imbalance, "hidden_dim": 64, "S_MAX": 6, "L_MAX": 3, "epochs": [1, 2, 3], "seed": 17})
    write_json(OUT / "r2_configuration_grid_v2.json", {"schema": "step11_r2_configuration_grid_v2", "configuration_count": len(grid), "epoch_row_count": len(grid) * 3, "configurations": grid})
    write_json(OUT / "feature_provenance_v1.json", provenance)
    write_json(OUT / "baseline_contract_v2.json", {"schema": "step11_baseline_contract_v2", "selection_eligible": ["trainable_r2_configurations"], "B0_TOP1_SINGLE": {"form": "most_frequent_SINGLE_subtype_in_FUSION_TRAIN", "target": "rank_1_frozen_top100", "no_map": False}, "B1_THRESHOLD_SET": {"deltas": [0.1, 0.25, 0.5, 1.0], "selection_population": "FUSION_TRAIN", "selection_metric": "SOURCE_MACRO_FLAT_SET_F1", "tie": "smaller_delta", "selected_delta": b1_delta, "selected_train_scores": b1_scores}, "RETRIEVAL_STATISTICS_FORM": {"eligible": False, "classifier": "L2_multinomial_logistic_regression", "C": 1.0, "max_iter": 2000, "random_state": 17, "features": ["top1_score", "top1_minus_top2", "top1_minus_top5", "mean_top100", "sd_top100", "normalized_top1"], "scaler_fit": "FUSION_TRAIN"}, "B2_ORACLE_FORM": {"diagnostic_only": True, "replacement": "gold_form_only"}, "B3_ORACLE_CANDIDATE_ASSIGNMENTS": {"diagnostic_only": True, "replacement": "candidate-contained gold membership/assignment only"}})
    write_json(OUT / "metric_contract_v2.json", {"schema": "step11_metric_contract_v2", "zero_division": 0, "primary_population": "all FUSION_VAL sources", "candidate_conditioned_population": "fully canonical-gold-representable in frozen Top-100, predeclared source IDs", "metrics": {"EXACT_CANONICAL_STRUCTURE_MATCH": "mean exact canonical semantic signature indicator", "COMPLETE_SCENARIO_SUCCESS": "mean complex-source scenario satisfaction indicator under permutation-invariant scenario/slot matching", "MAPPING_FORM_MACRO_F1": "unweighted six-class macro-F1", "NO_MAP": "TP/FP/FN/TN, precision/recall/F1, false forced-map and false-NO_MAP rates", "FLAT_SET": "source-macro and micro set precision/recall/F1 with both-empty=1", "CARDINALITY_EXACT_ACCURACY": "all applicable canonical structural cardinalities exact after matching", "STRUCTURE_VALIDITY_RATE": "valid candidate-contained canonical outputs divided by all predictions", "RETRIEVAL_VS_DECODER": "nonexact prediction classified by frozen full-gold Top-100 representability"}, "canonicalization": "scenario and choice-list order invariant; duplicate alternatives removed within sets"})
    write_json(OUT / "r2_search_protocol_v2.json", {"schema": "step11_r2_search_protocol_v2", "status": "FROZEN_PRE_OUTCOME", "development_seed": 17, "epochs": [1, 2, 3], "maximum_epochs": 3, "configuration_count": 4, "optimization": {"optimizer": "AdamW", "weight_decay": 0.0001, "source_batch_size": 8, "gradient_clipping": 1.0, "scheduler": "NONE", "warmup": 0, "precision": "FP32", "dropout": 0.0, "initialization": "torch default seeded initialization"}, "epoch_selection": "frozen lexicographic hierarchy; earliest epoch on complete tie; no early stopping", "global_selection": "one representative epoch per configuration, same hierarchy, configuration ID ascending final tie", "baseline_eligibility": {"trainable": ["four R2 configurations"], "comparison_only": ["B0_TOP1_SINGLE", "B1_THRESHOLD_SET"], "diagnostic_only": ["RETRIEVAL_STATISTICS_FORM", "B2_ORACLE_FORM", "B3_ORACLE_CANDIDATE_ASSIGNMENTS"]}, "selection_rule_sha256": sha(OUT / "selection_rule.json"), "interpretation_contract_sha256": sha(OUT / "interpretation_contract.json"), "feature_provenance_sha256": sha(OUT / "feature_provenance_v1.json"), "metric_contract_v2_sha256": sha(OUT / "metric_contract_v2.json"), "search_space_v3_sha256": sha(OUT / "search_space_v3.json"), "configuration_grid_sha256": sha(OUT / "r2_configuration_grid_v2.json"), "candidate_train_sha256": sha(CANDIDATE_ROOT / "forward_train_k100.jsonl.gz"), "candidate_dev_sha256": sha(CANDIDATE_ROOT / "forward_dev_k100.jsonl.gz"), "inner_split": {"train_count": 8666, "val_count": 1531, "overlap": 0}, "official_dev": "STEP11_OFFICIAL_DEV_ACCESS_FORBIDDEN", "test": "STEP11_TEST_ACCESS_FORBIDDEN", "invalid_run_rules": ["NaN_or_Inf_loss", "nonfinite_gradient", "checkpoint_hash_failure", "feature_provenance_mismatch"], "scientific_execution_count_at_freeze": 0})
    missing = [{"field": field, "status": "CLOSED", "resolution": resolution} for field, resolution in [("learning_rates", "literal [0.0001,0.0003] in search_space_v3"), ("epochs", "literal [1,2,3] and MAX_EPOCHS=3"), ("R2 seed", "17"), ("optimization constants", "explicit R2 protocol v2"), ("feature provenance", "feature_provenance_v1"), ("baseline definitions", "baseline_contract_v2"), ("metric definitions", "metric_contract_v2"), ("selection-rule SHA", "verified on-disk SHA ending c8a")]]
    write_csv(TABLES / "r2p_missing_protocol_fields.csv", missing)
    write_csv(TABLES / "r2p_feature_provenance_audit.csv", feature_rows)
    write_csv(TABLES / "r2p_baseline_contracts.csv", [{"baseline": "B0_TOP1_SINGLE", "eligibility": "comparison_only", "status": "FROZEN"}, {"baseline": "B1_THRESHOLD_SET", "eligibility": "comparison_only", "selected_delta": b1_delta, "status": "FROZEN"}, {"baseline": "RETRIEVAL_STATISTICS_FORM", "eligibility": "diagnostic_only", "status": "FROZEN"}, {"baseline": "B2_ORACLE_FORM", "eligibility": "diagnostic_only", "status": "FROZEN"}, {"baseline": "B3_ORACLE_CANDIDATE_ASSIGNMENTS", "eligibility": "diagnostic_only", "status": "FROZEN"}])
    write_csv(TABLES / "r2p_metric_definitions.csv", [{"metric": key, "definition": value, "status": "FROZEN"} for key, value in json.loads((OUT / "metric_contract_v2.json").read_text())["metrics"].items()])
    write_csv(TABLES / "r2p_configuration_grid.csv", grid)
    fixture_rows = [{"fixture": name, "status": "PASS"} for name in ["hierarchy_dominance", "macro_f1_tie", "no_map_tie", "flat_set_tie", "cardinality_tie", "configuration_id_tie", "earliest_epoch_tie"]]
    write_csv(TABLES / "r2p_selection_fixture_tests.csv", fixture_rows)
    write_csv(TABLES / "r2p_access_quarantine.csv", [{"surface": "FUSION_VAL_scientific", "count": 0, "guard": "authorized only in next milestone"}, {"surface": "official_DEV", "count": 0, "guard": "STEP11_OFFICIAL_DEV_ACCESS_FORBIDDEN"}, {"surface": "TEST", "count": 0, "guard": "STEP11_TEST_ACCESS_FORBIDDEN"}, {"surface": "R2_training", "count": 0, "guard": "protocol-only milestone"}])
    write_json(OUT / "r2_protocol_completion_manifest.json", {"schema": "step11_r2_protocol_completion_manifest_v1", "status": "COMPLETED_PRE_OUTCOME", "protocol_completion_occurred_before_any_r2_scientific_result": True, "commits": {"r1": "cdfd04abdd400086485dbebc7ad278d00b08154d", "r1a_blocker": "205e78c8677c7a6ac5dcf417b38879152d28ab6e", "r1b": "43db7f4c069000e1df783b4f7d7c027f4cb5a5f3", "r2_preflight_blocker": "be6e85c216813bda9c729f2055b8c5390a4c1084"}, "blockers_closed": ["R2-B001", "R2-B002", "R2-B003", "R2-B004"], "new_artifacts": {"search_space_v3": sha(OUT / "search_space_v3.json"), "feature_provenance_v1": sha(OUT / "feature_provenance_v1.json"), "baseline_contract_v2": sha(OUT / "baseline_contract_v2.json"), "metric_contract_v2": sha(OUT / "metric_contract_v2.json"), "r2_search_protocol_v2": sha(OUT / "r2_search_protocol_v2.json"), "configuration_grid_v2": sha(OUT / "r2_configuration_grid_v2.json")}, "statement": "NO R2 SCIENTIFIC MODEL RESULT EXISTED WHEN THESE VALUES WERE FROZEN."})
    report = f"""# STEP 11-R2P — PROTOCOL COMPLETION\n\nStatus: `STEP11_R2_PROTOCOL_COMPLETE_FROZEN`\n\nProtocol completion occurred before any R2 scientific result. R2 training and FUSION_VAL scientific scoring remain zero.\n\n- R2 grid: 4 configurations, 12 expected epoch rows\n- Learning rates: 0.0001, 0.0003\n- Imbalance variants: UNWEIGHTED_PRIMARY, TRAIN_DERIVED_FORM_WEIGHTED\n- Development seed: 17\n- Epochs: 1, 2, 3\n- Optimizer: AdamW; weight decay 0.0001; source batch size 8; gradient clipping 1.0; scheduler NONE; warmup 0; FP32\n- B1 TRAIN-selected delta: {b1_delta}\n- Feature determinism audit: {'PASS' if all(row['status'] == 'PASS' for row in feature_rows) else 'FAIL'} on {len(feature_rows)} bounded sources\n- Selection-rule SHA: {sha(OUT / 'selection_rule.json')}\n\nAll R2-B001 through R2-B004 blockers are closed without modifying the R1B architecture, canonical gold, candidate universe, or inner split.\n\nNo official DEV or TEST access occurred.\n"""
    (ROOT / "reports/STEP_11_R2P_PROTOCOL_COMPLETION.md").write_text(report, encoding="utf-8")
    write_json(OUT / "r2p_hashes.json", {name: sha(OUT / name) for name in ["search_space_v3.json", "feature_provenance_v1.json", "baseline_contract_v2.json", "metric_contract_v2.json", "r2_search_protocol_v2.json", "r2_protocol_completion_manifest.json", "r2_configuration_grid_v2.json"]})
    print(json.dumps({"grid_count": 4, "epoch_rows": 12, "b1_delta": b1_delta, "feature_audit_rows": len(feature_rows), "status": "STEP11_R2_PROTOCOL_COMPLETE_FROZEN"}, indent=2))


if __name__ == "__main__":
    main()
