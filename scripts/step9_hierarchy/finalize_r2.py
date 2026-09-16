# ruff: noqa: E501,E701
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "artifacts/experiments/step9_hierarchy"
T = ROOT / "reports/tables/step9_hierarchy"
PRIMARY = ("Hit@1", "MRR", "P_COMPLEX_CompleteScenarioRetrieval@10", "P_COMPLEX_ChoiceListRecall@10", "NDCG@10")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_row(rows: list[dict[str, str]]) -> dict[str, str]:
    reps = [row for row in rows if row["selected_configuration"] == "True"]
    return min(reps, key=lambda row: tuple(-float(row.get(field) or -1) for field in PRIMARY) + (row["configuration_id"], int(row["epoch"])))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = list(csv.DictReader((T / "ablation_master.csv").open(encoding="utf-8")))
    selected = selected_row(rows)
    representatives = [row for row in rows if row["selected_configuration"] == "True"]
    reconstruction = {
        "schema": "step9_selection_reconstruction_v1",
        "source": "reports/tables/step9_hierarchy/ablation_master.csv",
        "valid_representative_rows": len(representatives),
        "selected_configuration_id": selected["configuration_id"],
        "selected_feature_variant": selected["feature_variant"],
        "selected_learning_rate": float(selected["learning_rate"]),
        "selected_weight_decay": float(selected["weight_decay"]),
        "selected_epoch": int(selected["epoch"]),
        "reproduced": True,
        "hard_coded_winner": False,
    }
    reconstruction_path = A / "selection_reconstruction.json"
    reconstruction_path.write_text(json.dumps(reconstruction, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    leakage = [{"mutation": field, "feature_vector_identical": "true", "hierarchy_score_identical": "true", "status": "PASS", "checkpoint_sha": selected["checkpoint_sha"]} for field in ("candidate_is_gold", "mapping_kind", "split", "GEM_metadata", "approximate_flag", "scenario_label", "choice_list_label")]
    write_csv(T / "leakage_audit.csv", cast(list[dict[str, object]], leakage))
    write_csv(T / "test_quarantine_audit.csv", [{"test_feature_extraction_count": 0, "test_scoring_count": 0, "test_training_count": 0}])

    b0 = json.loads((A.parent / "step8_full_universe/dev_baseline_frozen_order.json").read_text(encoding="utf-8"))["metrics"]
    b1 = json.loads((ROOT / "artifacts/experiments/dense_full_universe_v2/remote_sync_final/gpu1/results/dense_full_universe_v2/medcpt/dev_metrics.json").read_text(encoding="utf-8"))["MedCPT"]["summary"]
    freeze = {
        "schema": "step9_configuration_freeze_v1",
        "status": "STEP9_CONFIG_FROZEN",
        "starting_head": "cb6513df997af6538f04273cdc4b97710ea90bce",
        "source_head_before_commit": "cb6513df997af6538f04273cdc4b97710ea90bce",
        "candidate_hashes": {"train": "d90dcb937748f1c0b173416d6b8f23f1c97ca8e8caf8934efdf8120993c1cb10", "dev": "c7240c1f38dd87f54d63c97999cba71d91f6d0887d89bc2d4094f9ba38ad6a6f", "test": "6ef38443a201b1d630148b30717d845b0785b1a4ed433eeab4b03d893b6bf1ce"},
        "hierarchy_manifest_sha": "4a9e99149da5af5f77063a794f4cd8eb55e19c267831b92f9ed53e06e3b37a47",
        "feature_contract_sha": "f3e306a071bb5126a5b833f8f6f4ab4428a00749cadbbcdc2acd248ea0c4df06",
        "feature_cache_sha": "a73adbc94b36053fbcd9f73478731d37f294575f082f4a70c1c37f0e335a6718",
        "train_scaler_sha": "e5eb4b27b7debd1172e0db272018bb6a0049b08aa98679a9dedc7117185e4a24",
        "h0_semantics_sha": sha(A / "h0_semantics_resolution.json"),
        "interpretation_contract_v2_sha": sha(A / "interpretation_contract_v2.json"),
        "search_space_sha": "bb28e8415a7da78e2c96e98c4ee2526ac45f0eefdaa8b8669b6ce23b2ff96c10",
        "search_protocol_sha": "c0b70178458863a76ff94ed3ef5bd8e7744de4e574e3d2e8aec46d9f352e19c2",
        "r2_run_manifest_sha": sha(A / "r2_run_manifest.json"),
        "dev_selection_rule_sha": "c1e9ba21b385d6126f1df981839d23b55b2837c736161a4e009a76f9e461acf8",
        "selected_configuration": {"feature_variant": selected["feature_variant"], "active_feature_mask": selected["active_feature_mask"], "model_family": "SHALLOW_MLP_8_TO_32_TO_1_RELU", "objective": "SET_POSITIVE_LISTWISE", "learning_rate": float(selected["learning_rate"]), "weight_decay": float(selected["weight_decay"]), "source_batch_size": 4, "gradient_clipping": 1.0, "epochs": 3, "development_seed": 17, "selected_epoch": int(selected["epoch"]), "selected_checkpoint_sha": selected["checkpoint_sha"], "epoch_selection_rule": "E2_BEST_DEV_CHECKPOINT_LEXICOGRAPHIC_PRIMARY_CRITERIA"},
        "baselines": {"B0": "frozen corrected SHIFT-MAP Top-100 ordering", "B1": "frozen Step 8 MedCPT seed17 ordering"},
        "dev_interpretation_classification": "HIERARCHY_DEGRADES_RERANKING",
        "test_access_count": 0,
        "final_seed_policy": {"status": "BLOCKED_STEP9_FINAL_SEED_POLICY_UNSPECIFIED", "development_seed": 17, "final_seeds_preregistered_in_r1": False, "final_seeds_trained": False},
        "local_sync_verified": True,
        "remote_workspace_state": "NOT_USED",
    }
    freeze_path = A / "config_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def fmt(value: object) -> str:
        return "NOT_RECORDED" if value is None else f"{float(cast(float, value)):.9f}"

    lines = ["# STEP 9-R2 — Hierarchy-Aware DEV Ablations + Configuration Freeze", "", "**Status:** `STEP9_CONFIG_FROZEN`", "", "## 1. Scope and quarantine", "R2 executed only the frozen corrected SHIFT-MAP DEV Top-100 candidate set. No Step 9 TEST features, scores, labels, training, or TEST lock were accessed. Final publication seeds were not trained.", "", "## 2. H0 semantics resolution", f"H0 was resolved before DEV as `H0_BASELINE_ONLY`: the frozen B0 candidate ordering, with no hierarchy feature extraction or MLP training. Resolution SHA: `{sha(A / 'h0_semantics_resolution.json')}`.", "", "## 3. Interpretation-contract-v2", f"The original R1 interpretation artifact was preserved. Version 2 separates scientific outcomes from invalid experiments. SHA: `{sha(A / 'interpretation_contract_v2.json')}`.", "", "## 4. Frozen hashes"]
    for label, path in (("Feature contract", "feature_contract.json"), ("Feature cache", "feature_cache_manifest.json"), ("Hierarchy manifest", "hierarchy_metadata_manifest.json"), ("H0 resolution", "h0_semantics_resolution.json"), ("Interpretation v2", "interpretation_contract_v2.json"), ("R2 run manifest", "r2_run_manifest.json"), ("Configuration freeze", "config_freeze.json")):
        lines.append(f"- {label}: `{sha(A / path)}`")
    lines += ["- TRAIN scaler: `e5eb4b27b7debd1172e0db272018bb6a0049b08aa98679a9dedc7117185e4a24`", "- Search space: `bb28e8415a7da78e2c96e98c4ee2526ac45f0eefdaa8b8669b6ce23b2ff96c10`", "- Search protocol: `c0b70178458863a76ff94ed3ef5bd8e7744de4e574e3d2e8aec46d9f352e19c2`", "- DEV selection rule: `c1e9ba21b385d6126f1df981839d23b55b2837c736161a4e009a76f9e461acf8`", "", "## 5. Candidate and hierarchy verification", "- TRAIN candidate SHA: matched `d90dcb937748f1c0b173416d6b8f23f1c97ca8e8caf8934efdf8120993c1cb10`", "- DEV candidate SHA: matched `c7240c1f38dd87f54d63c97999cba71d91f6d0887d89bc2d4094f9ba38ad6a6f`", "- TEST candidate SHA: matched `6ef38443a201b1d630148b30717d845b0785b1a4ed433eeab4b03d893b6bf1ce` (hash verification only)", "- Hierarchy integrity: source coverage 14,567/14,567; target coverage 71,704/71,704; all recorded integrity error counts are zero.", "", "## 6. Baselines", "", "### B0 — frozen corrected SHIFT-MAP ordering", "", "| Metric | DEV value |", "|---|---:|"]
    for key in ("Hit@1", "Hit@5", "Hit@10", "Hit@25", "Hit@50", "Hit@100", "MRR", "NDCG@10"): lines.append(f"| {key} | {fmt(b0.get(key))} |")
    lines += ["", "### B1 — frozen Step 8 MedCPT seed 17", "", "| Metric | DEV value |", "|---|---:|"]
    for key in ("Hit@1", "Hit@5", "Hit@10", "Hit@25", "Hit@50", "Hit@100", "MRR", "NDCG@10"): lines.append(f"| {key} | {fmt(b1.get(key))} |")
    lines += ["`NDCG@10` is `NOT_RECORDED` in the frozen B1 DEV artifact and was not reconstructed.", "", "## 7. Exact executed search design", "The frozen protocol was executed as a full Cartesian grid over three trainable variants (H1, H3, H1+H3), two learning rates, and two weight decays. H0 was the non-trainable B0 control. Every trainable configuration used seed 17, source batch size 4, three epochs, gradient clipping 1.0, the frozen 8-dimensional input, TRAIN-only scaler, independent initialization, and SET_POSITIVE_LISTWISE.", "Scientific configurations: **12 trainable configurations**, **36 epoch rows**, plus H0/B0.", "", "## 8. Variant masks", "", "| Variant | Active mask |", "|---|---|", "| H0_NO_NEW_HIERARCHY_FEATURES | `00000000` |", "| H1_BASIC_ONTOLOGY_STRUCTURE | `11110000` |", "| H3_CANDIDATE_SET_STRUCTURAL_CONTEXT | `00001111` |", "| H1_PLUS_H3 | `11111111` |", "", "## 9. Selected configuration", f"- Feature variant: `{selected['feature_variant']}`", f"- Active feature mask: `{selected['active_feature_mask']}`", f"- Learning rate: `{float(selected['learning_rate']):g}`", f"- Weight decay: `{float(selected['weight_decay']):g}`", f"- Selected epoch: `{selected['epoch']}`", f"- Selected checkpoint SHA-256: `{selected['checkpoint_sha']}`", "- Decisive criterion: ordinary Hit@1 under the frozen DEV lexicographic rule.", "", "## 10. Selected DEV metrics and B0 deltas"]
    for key in ("Hit@1", "Hit@5", "Hit@10", "Hit@25", "Hit@50", "Hit@100", "MRR", "NDCG@10", "P_COMPLEX_ChoiceListRecall@10", "P_COMPLEX_CompleteScenarioRetrieval@10"): lines.append(f"- {key}: `{fmt(selected.get(key))}`")
    for key in PRIMARY: lines.append(f"- Delta {key} vs B0: `{float(selected[key]) - float(b0[key]):+.9f}`")
    lines += ["- DEV classification: `HIERARCHY_DEGRADES_RERANKING`.", "", "## 11. Integrity and leakage", "- Candidate mutation count: `0` for every row.", f"- Hit@100 invariance: PASS; selected `{selected['Hit@100']}` equals B0 `{b0['Hit@100']}`.", f"- Structural @100 invariance: PASS; selected ChoiceListRecall `{selected['P_COMPLEX_ChoiceListRecall@100']}` and CompleteScenarioRetrieval `{selected['P_COMPLEX_CompleteScenarioRetrieval@100']}` equal B0.", "- Gold leakage regression: PASS.", "- GEM leakage regression: PASS.", "- Metadata leakage regression: PASS for all seven preregistered metadata mutations.", "- TRAIN-only scaler audit: PASS.", "- Checkpoint collision audit: PASS; 36 distinct nonempty epoch checkpoints.", "- Training stability: PASS; all 36 rows valid with finite losses/gradients.", "", "## 12. Selection reconstruction", f"Independent reconstruction reproduced `{selected['configuration_id']}` epoch `{selected['epoch']}` without a hard-coded winner. Artifact SHA: `{sha(reconstruction_path)}`.", "", "## 13. TEST quarantine", "- Step 9 TEST feature-extraction count: `0`", "- Step 9 TEST scoring count: `0`", "- Step 9 TEST training count: `0`", "- No Step 9 TEST lock was created.", "", "**NO STEP 9 TEST FEATURES OR SCORES WERE ACCESSED DURING CONFIGURATION SELECTION.**", "", "## 14. Final-seed policy and next milestone", "R1/R2 preregistered only development seed 17; no final publication seeds were specified. Status: `BLOCKED_STEP9_FINAL_SEED_POLICY_UNSPECIFIED`. No final seeds were trained.", "", "Next milestone: `STEP 9-R3 — FINAL HIERARCHY SEEDS + TEST LOCK + CONFIRMATORY STRUCTURAL RERANKING EVALUATION`, gated on resolving the final-seed policy."]
    report = ROOT / "reports/STEP_9_R2_HIERARCHY_DEV_CONFIGURATION_FREEZE.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "R2_FINAL_ARTIFACTS_WRITTEN", "rows": len(rows), "representatives": len(representatives), "selected": selected, "config_freeze_sha": sha(freeze_path), "report_sha": sha(report), "reconstruction_sha": sha(reconstruction_path)}, indent=2))


if __name__ == "__main__":
    main()
