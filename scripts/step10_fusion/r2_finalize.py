# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/experiments/step10_fusion"
TABLES = ROOT / "reports/tables/step10_fusion"
PRIMARY = ("Hit@1", "MRR", "P_COMPLEX_CompleteScenarioRetrieval@10", "P_COMPLEX_ChoiceListRecall@10", "NDCG@10")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for row in rows:
        for key in list(row):
            if key in PRIMARY or key in {"lambda", "learning_rate", "weight_decay", "epoch", "selected_epoch", "runtime_seconds", "training_loss", "max_abs_residual", "max_abs_fusion_delta", "candidate_mutation_count"}:
                if row[key] in {"", "None", "null"}:
                    row[key] = None
                elif key in {"epoch", "selected_epoch", "candidate_mutation_count"}:
                    row[key] = int(float(row[key]))
                else:
                    row[key] = float(row[key])
    return rows


def key(row):
    return tuple(-float(row[k]) for k in PRIMARY) + (float(row.get("lambda") or 0.0), str(row["configuration_id"]), int(row.get("epoch") or 0))


def main():
    rows = read_csv(TABLES / "ablation_master.csv")
    reps = read_csv(TABLES / "configuration_representatives.csv")
    assert len(rows) == 37 and len(reps) == 13
    assert sum(r["configuration_id"] != "LAMBDA_0_SEMANTIC_ONLY" for r in rows) == 36
    assert all(str(r["valid"]).lower() == "true" for r in rows)
    assert all(int(r["candidate_mutation_count"]) == 0 for r in rows)
    assert all(str(r["Hit@100_invariant"]).lower() == "true" for r in rows)
    assert all(str(r["structural_at_100_invariant"]).lower() == "true" for r in rows)
    selected = min(reps, key=key)
    ordered = sorted(reps, key=key)
    runner_up = ordered[1]
    b0 = next(r for r in reps if r["configuration_id"] == "LAMBDA_0_SEMANTIC_ONLY")
    classification = "SEMANTIC_BASELINE_RETAINS_PROMOTION" if selected["configuration_id"] == "LAMBDA_0_SEMANTIC_ONLY" else ("FUSION_IMPROVES_RANKING" if all(selected[k] > b0[k] for k in PRIMARY) else "FUSION_MIXED_RESULT" if any(selected[k] > b0[k] for k in PRIMARY) else "FUSION_DEGRADES_RANKING")
    grid_sha = sha(OUT / "r2_configuration_grid.json")
    run_sha = sha(OUT / "r2_run_manifest.json")
    r1_sha = sha(OUT / "r1_artifact_hashes.json")
    split_sha = sha(OUT / "inner_split_manifest.json")
    scaler_sha = sha(OUT / "inner_train_scaler_manifest.json")
    anchor_sha = "d90dcb937748f1c0b173416d6b8f23f1c97ca8e8caf8934efdf8120993c1cb10"
    selection_sha = sha(OUT / "dev_selection_rule.json")
    interpretation_sha = sha(OUT / "interpretation_contract.json")
    freeze = {
        "schema": "step10_configuration_freeze_v1",
        "selected_model": "B0_SEMANTIC_ONLY" if selected["configuration_id"] == "LAMBDA_0_SEMANTIC_ONLY" else "TRAINABLE_FUSION",
        "selected_configuration_id": selected["configuration_id"],
        "selected_lambda": selected["lambda"],
        "selected_learning_rate": selected["learning_rate"],
        "selected_weight_decay": selected["weight_decay"],
        "selected_epoch": selected["epoch"],
        "selected_checkpoint_sha256": selected["checkpoint_sha"] if selected["configuration_id"] != "LAMBDA_0_SEMANTIC_ONLY" else None,
        "r1_hash_manifest_sha256": r1_sha,
        "r2_run_manifest_sha256": run_sha,
        "configuration_grid_sha256": grid_sha,
        "selection_rule_sha256": selection_sha,
        "interpretation_contract_sha256": interpretation_sha,
        "inner_split_manifest_sha256": split_sha,
        "fusion_train_source_sha256": "150a57de2e9101ee9bb5c1699bbb7b3a7f9b1fc1d0863018a8766808c151f24c",
        "fusion_val_source_sha256": "225df0f8343bc5358e1980688018916df4c75675595a77f37ca65ceea17bbcb9",
        "inner_scaler_sha256": scaler_sha,
        "semantic_anchor_candidate_sha256": anchor_sha,
        "h3_feature_order": ["candidate_same_parent_fraction", "candidate_same_family_fraction", "candidate_shared_ancestor_fraction", "candidate_same_root_fraction"],
        "selected_metric_vector": {k: selected[k] for k in PRIMARY},
        "b0_metric_vector": {k: b0[k] for k in PRIMARY},
        "runner_up_configuration_id": runner_up["configuration_id"],
        "runner_up_metric_vector": {k: runner_up[k] for k in PRIMARY},
        "decisive_criterion": "Hit@1;MRR;P_COMPLEX_CompleteScenarioRetrieval@10;P_COMPLEX_ChoiceListRecall@10;NDCG@10;smaller_lambda;configuration_id;earlier_epoch",
        "development_seed": 17,
        "objective": "FULL_TOP100_SET_POSITIVE_LISTWISE",
        "batch_size": 32,
        "epochs": 3,
        "residual_bound": [-1, 1],
        "official_dev_access_count": 0,
        "test_access_count": 0,
        "interpretation": classification,
        "local_sync_verified": True,
        "starting_head": "cd57fe297bdbf2623832b52c0a55fc16967d4039",
    }
    (OUT / "config_freeze.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    recon = {"schema": "step10_deterministic_selection_reconstruction_v1", "representative_count": len(reps), "epoch_row_count": len(rows), "selected_configuration_id": selected["configuration_id"], "selected_epoch": selected["epoch"], "runner_up_configuration_id": runner_up["configuration_id"], "classification": classification, "reconstructed_from": "reports/tables/step10_fusion/ablation_master.csv + configuration_representatives.csv", "pass": True}
    (OUT / "selection_reconstruction.json").write_text(json.dumps(recon, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "runner_up": runner_up, "classification": classification, "config_freeze_sha256": sha(OUT / "config_freeze.json"), "reconstruction": recon}, indent=2, default=str))


if __name__ == "__main__":
    main()
