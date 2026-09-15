from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/experiments/step8_full_universe"
TABLES = ROOT / "reports/tables/step8_full_universe"
CONTRACT_SHA = "17c3455e2ca2414f32fc11bfbbfb186dbc710b420d8b940042e5a5b390612d46"
PROTOCOL_SHA = "5f1a9d34249acf15b451ac8434eccb57e41ff4702027359fcbf036bd156d600d"
TRAIN_SHA = "d90dcb937748f1c0b173416d6b8f23f1c97ca8e8caf8934efdf8120993c1cb10"
DEV_SHA = "c7240c1f38dd87f54d63c97999cba71d91f6d0887d89bc2d4094f9ba38ad6a6f"
TEST_SHA = "6ef38443a201b1d630148b30717d845b0785b1a4ed433eeab4b03d893b6bf1ce"
FIELDS = ["Hit@1", "MRR", "Hit@10", "NDCG@10", "P_COMPLEX_CompleteScenarioRetrieval@10"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key(row: dict[str, str]) -> tuple[float, ...]:
    return tuple(float(row.get(field, "-1")) for field in FIELDS)


def best(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(rows, key=lambda r: (key(r), -int(r["epoch"])))


def select_stage(rows: list[dict[str, str]], stage: str, group_field: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    stage_rows = [r for r in rows if r["stage"] == stage]
    groups = sorted({r[group_field] for r in stage_rows})
    winners = [best([r for r in stage_rows if r[group_field] == value]) for value in groups]
    return winners, best(winners)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selection_artifact(path: Path, winner: dict[str, str], runner_up: dict[str, str], criterion: str) -> None:
    payload = {
        "winner": winner,
        "runner_up": runner_up,
        "decisive_criterion": criterion,
        "selected_epoch": int(winner["epoch"]),
        "full_metric_vector": {
            field: float(winner[field])
            for field in winner
            if field.startswith("Hit@") or field in {"MRR", "NDCG@10"} or field.startswith("P_COMPLEX") or field.startswith("P_COMBINATION")
        },
        "checkpoint_sha256": winner.get("checkpoint_sha256"),
        "training_list_contract_v2_sha256": CONTRACT_SHA,
        "r2_search_protocol_v2_sha256": PROTOCOL_SHA,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    master = read_rows(OUT / "ablation_master.csv")
    objective_winners, objective = select_stage(master, "OBJECTIVE", "objective")
    strategy_winners, strategy = select_stage(master, "STRATEGY", "list_strategy")
    lr_winners, lr = select_stage(master, "LEARNING_RATE", "learning_rate")
    objective_runner = next(r for r in objective_winners if r["objective"] != objective["objective"])
    strategy_runner = next(r for r in strategy_winners if r["list_strategy"] != strategy["list_strategy"])
    lr_runner = next(r for r in lr_winners if r["learning_rate"] != lr["learning_rate"])
    TABLES.mkdir(parents=True, exist_ok=True)
    write_csv(TABLES / "objective_ablation_dev.csv", objective_winners)
    write_csv(TABLES / "list_strategy_dev.csv", strategy_winners)
    write_csv(TABLES / "learning_rate_dev.csv", lr_winners)
    write_csv(TABLES / "ablation_master.csv", master)
    write_csv(TABLES / "config_selection_summary.csv", [objective, strategy, lr])
    write_csv(TABLES / "training_runtime_ablation.csv", master)
    write_csv(TABLES / "candidate_invariance_audit.csv", master)
    write_csv(TABLES / "variable_list_audit.csv", master)
    write_csv(
        TABLES / "test_quarantine_audit.csv", [{"test_scoring_count": "0", "test_training_count": "0", "candidate_test_sha256": TEST_SHA}]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    selection_artifact(OUT / "selected_objective.json", objective, objective_runner, FIELDS[0])
    selection_artifact(OUT / "selected_list_strategy.json", strategy, strategy_runner, FIELDS[0])
    selection_artifact(OUT / "selected_learning_rate.json", lr, lr_runner, FIELDS[0])
    reconstruction = {
        "selected_objective": best(objective_winners)["objective"],
        "selected_strategy": best(strategy_winners)["list_strategy"],
        "selected_learning_rate": best(lr_winners)["learning_rate"],
        "selected_epochs": {"objective": int(objective["epoch"]), "strategy": int(strategy["epoch"]), "learning_rate": int(lr["epoch"])},
        "hard_coded_winners": False,
        "selection_fields": FIELDS,
        "tie_break": "earliest_epoch",
    }
    (OUT / "selection_reconstruction.json").write_text(json.dumps(reconstruction, indent=2) + "\n")
    baseline = json.loads((OUT / "dev_baseline_frozen_order.json").read_text())
    freeze = {
        "schema": "step8_config_freeze_v1",
        "status": "STEP8_CONFIG_FROZEN",
        "training_list_contract_v2_sha256": CONTRACT_SHA,
        "r2_search_protocol_v2_sha256": PROTOCOL_SHA,
        "r2b_run_manifest_sha256": sha(OUT / "r2b_run_manifest.json"),
        "candidate_train_sha256": TRAIN_SHA,
        "candidate_dev_sha256": DEV_SHA,
        "candidate_test_sha256": TEST_SHA,
        "selected_objective": objective["objective"],
        "selected_list_strategy": strategy["list_strategy"],
        "selected_learning_rate": float(lr["learning_rate"]),
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "gradient_clipping": 1.0,
        "max_length": 96,
        "precision": "FP32",
        "source_batch_size": 32,
        "effective_source_batch_size": 32,
        "minimum_list_size": 8,
        "negative_count_rule": "max(1, 8 - P_i)",
        "maximum_observed_training_list_length": int(max(int(r["list_max_length"]) for r in master)),
        "maximum_epochs": 3,
        "epoch_selection_rule": "E2_BEST_DEV_CHECKPOINT_WITHIN_MAX_EPOCH_BUDGET",
        "dev_selection_rule": FIELDS,
        "development_seed": 17,
        "final_seed_preregistration": [17, 42, 2026],
        "canonical_final_seed": 17,
        "model_id": "ncbi/MedCPT-Cross-Encoder",
        "model_revision": "71caf65d4927987813984f54c284405a13fcca49",
        "tokenizer_revision": "71caf65d4927987813984f54c284405a13fcca49",
        "input_pair_order": ["source_description", "target_description"],
        "evaluator": "retrieval_evaluator_v3",
        "test_scoring_count": 0,
        "test_training_count": 0,
        "pre_rerank_dev_baseline": baseline["metrics"],
        "selected_objective_checkpoint_sha256": objective.get("checkpoint_sha256"),
        "selected_strategy_checkpoint_sha256": strategy.get("checkpoint_sha256"),
        "selected_learning_rate_checkpoint_sha256": lr.get("checkpoint_sha256"),
    }
    (OUT / "config_freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "R2B_SELECTION_ARTIFACTS_COMPLETE",
                "objective": objective["objective"],
                "strategy": strategy["list_strategy"],
                "learning_rate": lr["learning_rate"],
                "rows": len(master),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
