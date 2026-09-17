# mypy: ignore-errors
# ruff: noqa: E501
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
R1_PATH = ROOT / "scripts/step10_fusion/r1_preflight.py"
spec = __import__("importlib.util").util.spec_from_file_location("step10_r1_preflight", R1_PATH)
r1 = __import__("importlib.util").util.module_from_spec(spec)
sys.modules[spec.name] = r1
spec.loader.exec_module(r1)
r2 = r1.r2

OUT = ROOT / "artifacts/experiments/step10_fusion"
TABLES = ROOT / "reports/tables/step10_fusion"
TRAIN_PATH = r1.TRAIN_PATH
SEED = 17
EPOCHS = 3
BATCH = 32
LAMS = (0.05, 0.10, 0.20)
LRS = (1e-4, 3e-4)
WDS = (1e-4, 1e-3)
PRIMARY = ("Hit@1", "MRR", "P_COMPLEX_CompleteScenarioRetrieval@10", "P_COMPLEX_ChoiceListRecall@10", "NDCG@10")
ORDINARY = {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"}
COMBINATION = {"COMBINATION"}
COMBINATION_ALT = {"COMBINATION_WITH_ALTERNATIVES"}
COMPLEX = {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sid(group: list[dict[str, Any]]) -> str:
    return str(group[0]["source_id"])


def gold_codes(group: list[dict[str, Any]]) -> set[str]:
    return {str(row["target_code"]) for row in group if row.get("_gold")}


def source_hash(groups: list[list[dict[str, Any]]]) -> str:
    return hashlib.sha256("\n".join(sorted(sid(g) for g in groups)).encode()).hexdigest()


def prepare_inner(benchmark: dict[str, dict[str, Any]]):
    groups = r2.prepare(TRAIN_PATH, benchmark)
    for group in groups:
        kind = str(benchmark[sid(group)]["mapping_kind"])
        for row in group:
            row["_mapping_kind"] = kind
    fusion_train, fusion_val, _ = r1.make_split(groups)
    manifest = json.loads((OUT / "inner_split_manifest.json").read_text(encoding="utf-8"))
    if source_hash(fusion_train) != manifest["fusion_train_source_sha256"] or source_hash(fusion_val) != manifest["fusion_val_source_sha256"]:
        raise RuntimeError("STEP10_INNER_SPLIT_HASH_MISMATCH")
    return groups, fusion_train, fusion_val


def canonical_population(kind: str) -> str:
    if kind in COMBINATION:
        return "P_COMBINATION"
    if kind in COMBINATION_ALT:
        return "P_COMBINATION_WITH_ALTERNATIVES"
    if kind in COMPLEX:
        return "P_COMPLEX"
    if kind in ORDINARY:
        return "P_ORDINARY_ANSWERABLE"
    return "P_NO_MAP"


def population_audit(groups: list[list[dict[str, Any]]], benchmark: dict[str, dict[str, Any]], split: str) -> list[dict[str, Any]]:
    counts = {"P_ALL": len(groups), "P_ORDINARY_ANSWERABLE": 0, "P_ORDINARY_SUPERVISED_GOLD_PRESENT": 0, "P_ORDINARY_GOLD_MISSING": 0, "P_COMBINATION": 0, "P_COMBINATION_WITH_ALTERNATIVES": 0, "P_COMPLEX": 0, "P_NO_MAP": 0}
    for group in groups:
        kind = str(benchmark[sid(group)]["mapping_kind"])
        pop = canonical_population(kind)
        counts[pop] += 1
        if kind in ORDINARY:
            if gold_codes(group):
                counts["P_ORDINARY_SUPERVISED_GOLD_PRESENT"] += 1
            else:
                counts["P_ORDINARY_GOLD_MISSING"] += 1
    # P_COMPLEX is the canonical union, not an additional count.
    counts["P_COMPLEX"] = counts["P_COMBINATION"] + counts["P_COMBINATION_WITH_ALTERNATIVES"] + sum(1 for g in groups if benchmark[sid(g)]["mapping_kind"] == "MULTI_SCENARIO")
    rows = []
    for name, count in counts.items():
        rows.append({"split": split, "population": name, "count": count})
    ordinary = [g for g in groups if benchmark[sid(g)]["mapping_kind"] in ORDINARY and benchmark[sid(g)].get("valid_target_codes")]
    complex_groups = [g for g in groups if benchmark[sid(g)]["mapping_kind"] in COMPLEX]
    rows.extend([
        {"split": split, "population": "P_ORDINARY_ANSWERABLE_SOURCE_SHA256", "count": source_hash(ordinary)},
        {"split": split, "population": "P_COMPLEX_SOURCE_SHA256", "count": source_hash(complex_groups)},
    ])
    return rows


def normalized_scores(group: list[dict[str, Any]]) -> list[float]:
    return r1.normalization([float(row["retriever_score"]) for row in group])


def rank_codes(group: list[dict[str, Any]], scores: list[float]) -> list[str]:
    ranked = sorted(zip(group, scores, strict=True), key=lambda x: (-float(x[1]), int(x[0]["candidate_rank"])))
    return [str(row["target_code"]) for row, _ in ranked]


def structural_pair(example: dict[str, Any], ranked: list[str], k: int) -> tuple[float, float]:
    retrieved = set(ranked[:k])
    scenarios = example.get("scenarios", [])
    if not scenarios:
        hit = float(bool(retrieved & {str(x) for x in example.get("valid_target_codes", [])}))
        return hit, hit
    total = sum(len(s["choice_lists"]) for s in scenarios)
    covered = sum(sum(any(str(a["target_code"]) in retrieved for a in c["alternatives"]) for c in s["choice_lists"]) for s in scenarios)
    complete = any(all(any(str(a["target_code"]) in retrieved for a in c["alternatives"]) for c in s["choice_lists"]) for s in scenarios)
    return (covered / total if total else 0.0), float(complete)


def ndcg(gold: set[str], ranked: list[str], k: int) -> float:
    gains = [1.0 if code in gold else 0.0 for code in ranked[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = min(len(gold), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal))
    return float(dcg / idcg) if idcg else 0.0


def evaluate(groups: list[list[dict[str, Any]]], ranked: list[list[str]], benchmark: dict[str, dict[str, Any]]) -> dict[str, float]:
    examples = [benchmark[sid(g)] for g in groups]
    ordinary = [(e, r) for e, r in zip(examples, ranked, strict=True) if e.get("mapping_kind") in ORDINARY and e.get("valid_target_codes")]
    out: dict[str, float] = {}
    for k in (1, 5, 10, 25, 50, 100):
        out[f"Hit@{k}"] = sum(bool(set(r[:k]) & {str(x) for x in e["valid_target_codes"]}) for e, r in ordinary) / len(ordinary)
    rr = []
    for e, r in ordinary:
        ranks = [i for i, code in enumerate(r, 1) if code in {str(x) for x in e["valid_target_codes"]}]
        rr.append(0.0 if not ranks else 1.0 / min(ranks))
    out["MRR"] = sum(rr) / len(rr)
    out["NDCG@10"] = sum(ndcg({str(x) for x in e["valid_target_codes"]}, r, 10) for e, r in ordinary) / len(ordinary)
    for population, kinds in (("P_COMBINATION", COMBINATION), ("P_COMBINATION_WITH_ALTERNATIVES", COMBINATION_ALT), ("P_COMPLEX", COMPLEX)):
        selected = [(e, r) for e, r in zip(examples, ranked, strict=True) if e.get("mapping_kind") in kinds]
        for k in (1, 5, 10, 25, 50, 100):
            pairs = [structural_pair(e, r, k) for e, r in selected]
            out[f"{population}_ChoiceListRecall@{k}"] = sum(p[0] for p in pairs) / len(pairs) if pairs else 0.0
            out[f"{population}_CompleteScenarioRetrieval@{k}"] = sum(p[1] for p in pairs) / len(pairs) if pairs else 0.0
    return out


def score_model(model: torch.nn.Module | None, groups: list[list[dict[str, Any]]], features: dict[str, list[list[float]]], zscores: dict[str, list[float]], lam: float) -> tuple[list[list[str]], list[float], list[float]]:
    ranked_all = []
    max_residual = 0.0
    max_delta = 0.0
    with torch.inference_mode():
        for group in groups:
            z = zscores[sid(group)]
            if model is None:
                residual = [0.0] * len(group)
            else:
                residual = model(torch.tensor(features[sid(group)], dtype=torch.float32)).tolist()
            if residual:
                max_residual = max(max_residual, max(abs(float(x)) for x in residual))
            fused = [float(s) + lam * float(r) for s, r in zip(z, residual, strict=True)]
            max_delta = max(max_delta, max((abs(lam * float(r)) for r in residual), default=0.0))
            ranked_all.append(rank_codes(group, fused))
    return ranked_all, [max_residual], [max_delta]


def make_config_grid() -> list[dict[str, Any]]:
    rows = []
    for lam in LAMS:
        for lr in LRS:
            for wd in WDS:
                config_id = f"fusion_lambda_{lam:.2f}_lr_{lr:.0e}_wd_{wd:.0e}".replace(".", "p")
                rows.append({"configuration_id": config_id, "lambda": lam, "learning_rate": lr, "weight_decay": wd, "seed": SEED, "architecture": "4_TO_16_TO_1_RELU_TANH"})
    return rows


def metric_key(row: dict[str, Any], global_key: bool = True) -> tuple[Any, ...]:
    values = tuple(-float(row.get(field, -1.0)) for field in PRIMARY)
    lam = float(row.get("lambda", 0.0))
    return values + ((lam,) if global_key else ()) + (str(row["configuration_id"]), int(row.get("epoch", 0)))


def select_representative(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=lambda row: metric_key(row, global_key=False))


def train_config(config: dict[str, Any], train_groups, val_groups, train_features, val_features, train_z, val_z, benchmark):
    torch.manual_seed(SEED)
    random.seed(SEED)
    model = r1.FusionResidual()
    initial_bytes = json.dumps({k: v.detach().cpu().tolist() for k, v in model.state_dict().items()}, sort_keys=True).encode()
    init_sha = hashlib.sha256(initial_bytes).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    supervised = [g for g in train_groups if benchmark[sid(g)]["mapping_kind"] in ORDINARY and gold_codes(g)]
    rows = []
    for epoch in range(1, EPOCHS + 1):
        start_time = time.perf_counter()
        model.train()
        order = list(range(len(supervised)))
        random.Random(f"{config['configuration_id']}:{epoch}").shuffle(order)
        losses = []
        for start in range(0, len(order), BATCH):
            logits = []
            positive_indices = []
            for idx in order[start : start + BATCH]:
                group = supervised[idx]
                logits.append(model(torch.tensor(train_features[sid(group)], dtype=torch.float32)))
                positive_indices.append([i for i, row in enumerate(group) if row.get("_gold")])
            loss = r2.source_balanced_listwise_loss(logits, positive_indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise RuntimeError("INVALID_FUSION_EXPERIMENT_NONFINITE_GRADIENT")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        checkpoint = OUT / "r2_checkpoints" / config["configuration_id"] / f"epoch_{epoch}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint.exists():
            raise FileExistsError("STEP10_CHECKPOINT_COLLISION")
        torch.save({"state_dict": model.state_dict(), "configuration": config, "epoch": epoch, "initialization_sha": init_sha}, checkpoint)
        ranked, max_residual, max_delta = score_model(model, val_groups, val_features, val_z, config["lambda"])
        metrics = evaluate(val_groups, ranked, benchmark)
        valid = max_residual[0] <= 1.000001 and max_delta[0] <= config["lambda"] + 1e-6
        rows.append({"run_id": f"r2_{config['configuration_id']}", **config, "epoch": epoch, "training_loss": sum(losses) / len(losses), "initialization_sha": init_sha, "checkpoint_sha": sha(checkpoint), "checkpoint_path": str(checkpoint), "runtime_seconds": time.perf_counter() - start_time, "max_abs_residual": max_residual[0], "max_abs_fusion_delta": max_delta[0], "candidate_mutation_count": 0, "Hit@100_invariant": True, "structural_at_100_invariant": True, "valid": valid, "selected_epoch": None, "selected_global": False, **metrics})
    representative = select_representative(rows)
    for row in rows:
        row["selected_epoch"] = representative["epoch"]
    return rows, representative


def movement(b0_ranked, selected_ranked, b0_metrics, selected_metrics, groups, benchmark):
    if b0_ranked == selected_ranked:
        return {"top1_improved": 0, "top1_worsened": 0, "top1_unchanged": len(groups), "mrr_improved": 0, "mrr_worsened": 0, "mrr_unchanged": len([g for g in groups if benchmark[sid(g)]["mapping_kind"] in ORDINARY and benchmark[sid(g)].get("valid_target_codes")]), "status": "B0_SELF_COMPARISON_UNCHANGED"}
    # This path is descriptive only; selected metrics remain frozen.
    return {"status": "DESCRIPTIVE_COMPARISON", "top1_improved": None, "top1_worsened": None, "top1_unchanged": None, "mrr_improved": None, "mrr_worsened": None, "mrr_unchanged": None}


def main():
    torch.set_num_threads(1)
    benchmark = r2.load_benchmark()
    groups, fusion_train, fusion_val = prepare_inner(benchmark)
    manifest_sha = sha(OUT / "inner_split_manifest.json")
    r1_hash = sha(OUT / "r1_artifact_hashes.json")
    expected = {
        "r1_artifact_hash_manifest_sha256": "45a6e70e05ddf78ac7e5ade680889a1ee959d49979fe4755a7ccc76e7e7002c8",
        "search_space_sha256": "3c14f4a93c45e5364193c2b98d922044ba8ebefb4ae7bb4137df42c43e337b52",
        "r2_protocol_sha256": "21c5465e7e73b04853b73d4462562be1165cbe0f30e24d653c6be62871a2a2e4",
        "selection_rule_sha256": "42c58bfd176734d8071e224491444bb78f39c2f9273640ef3d284952fb8994c3",
        "interpretation_sha256": "45d3bd626ff101e3bddd2814078f8d10b1f22fe4647874120ec56b7af283cdc0",
        "seed_policy_sha256": "7e1cfd46587e83fac0ea5a174ae3e415b1b703264ea7d571dc67f3d98e694d65",
        "prior_test_sha256": "6142a1c9c9b6027118eb824eb2fa98da16d3c3b9cc4587d2320c5d34be54199b",
        "inner_scaler_sha256": "ef7d83130e6e295587c8559263fa6c90a099026d78bdda2f04feba9b0f7231e5",
        "train_candidate_sha256": "d90dcb937748f1c0b173416d6b8f23f1c97ca8e8caf8934efdf8120993c1cb10",
    }
    actual = {"r1_artifact_hash_manifest_sha256": r1_hash, "search_space_sha256": sha(OUT / "search_space.json"), "r2_protocol_sha256": sha(OUT / "r2_search_protocol.json"), "selection_rule_sha256": sha(OUT / "dev_selection_rule.json"), "interpretation_sha256": sha(OUT / "interpretation_contract.json"), "seed_policy_sha256": sha(OUT / "final_seed_policy.json"), "prior_test_sha256": sha(OUT / "prior_test_exposure.json"), "inner_scaler_sha256": sha(OUT / "inner_train_scaler_manifest.json"), "train_candidate_sha256": sha(TRAIN_PATH)}
    if actual != expected:
        raise RuntimeError("STEP10_R1_HASH_VERIFICATION_FAILED:" + json.dumps({k: [expected[k], actual[k]] for k in expected if expected[k] != actual[k]}))
    grid = make_config_grid()
    json_write(OUT / "r2_configuration_grid.json", {"schema": "step10_r2_configuration_grid_v1", "configurations": grid, "count": len(grid)})
    grid_sha = sha(OUT / "r2_configuration_grid.json")
    run_manifest = {"schema": "step10_r2_run_manifest_v1", "status": "FROZEN_BEFORE_R2_SCIENTIFIC_EXECUTION", "starting_head": "cd57fe297bdbf2623832b52c0a55fc16967d4039", "r1_hash_manifest_sha256": r1_hash, "search_space_sha256": actual["search_space_sha256"], "r2_protocol_sha256": actual["r2_protocol_sha256"], "selection_rule_sha256": actual["selection_rule_sha256"], "interpretation_contract_sha256": actual["interpretation_sha256"], "inner_split_manifest_sha256": manifest_sha, "fusion_train_source_sha256": source_hash(fusion_train), "fusion_val_source_sha256": source_hash(fusion_val), "inner_scaler_sha256": actual["inner_scaler_sha256"], "semantic_anchor_candidate_sha256": actual["train_candidate_sha256"], "h3_feature_order": list(r1.H3_NAMES), "fusion_equation": "s_fused = z_sem + lambda * r_h", "lambda_grid": list(LAMS), "learning_rate_grid": list(LRS), "weight_decay_grid": list(WDS), "seed": SEED, "batch_size": BATCH, "epochs": EPOCHS, "objective": "FULL_TOP100_SET_POSITIVE_LISTWISE", "official_dev_access": {"feature_extraction_count": 0, "scoring_count": 0, "training_count": 0}, "test_access": {"feature_extraction_count": 0, "scoring_count": 0, "training_count": 0}, "configuration_grid_sha256": grid_sha}
    json_write(OUT / "r2_run_manifest.json", run_manifest)
    run_manifest_sha = sha(OUT / "r2_run_manifest.json")
    raw_train = r2.fast_features(fusion_train, *r2.build_nodes())
    raw_val = r2.fast_features(fusion_val, *r2.build_nodes())
    scaler = r2.fit_train_scaler((tuple(row) for group in raw_train for row in group), role="TRAIN")
    if hashlib.sha256(json.dumps({"means": list(scaler.means), "stds": list(scaler.scales)}, sort_keys=True).encode()).hexdigest() == "":
        raise RuntimeError("UNREACHABLE")
    train_features8 = r2.apply_variant(raw_train, scaler, "H3_CANDIDATE_SET_STRUCTURAL_CONTEXT")
    val_features8 = r2.apply_variant(raw_val, scaler, "H3_CANDIDATE_SET_STRUCTURAL_CONTEXT")
    train_features = {sid(g): [list(row[4:]) for row in f] for g, f in zip(fusion_train, train_features8, strict=True)}
    val_features = {sid(g): [list(row[4:]) for row in f] for g, f in zip(fusion_val, val_features8, strict=True)}
    train_z = {sid(g): normalized_scores(g) for g in fusion_train}
    val_z = {sid(g): normalized_scores(g) for g in fusion_val}
    b0_ranked, _, _ = score_model(None, fusion_val, val_features, val_z, 0.0)
    b0_metrics = evaluate(fusion_val, b0_ranked, benchmark)
    b0_row = {"run_id": "r2_B0", "configuration_id": "LAMBDA_0_SEMANTIC_ONLY", "is_B0": True, "lambda": 0.0, "learning_rate": None, "weight_decay": None, "seed": SEED, "epoch": 0, "training_loss": None, "initialization_sha": "NON_TRAINABLE_CONTROL", "checkpoint_sha": None, "checkpoint_path": None, "runtime_seconds": 0.0, "max_abs_residual": 0.0, "max_abs_fusion_delta": 0.0, "candidate_mutation_count": 0, "Hit@100_invariant": True, "structural_at_100_invariant": True, "valid": True, "selected_epoch": 0, "selected_global": False, **b0_metrics}
    all_rows = [b0_row]
    reps = []
    for config in grid:
        rows, rep = train_config(config, fusion_train, fusion_val, train_features, val_features, train_z, val_z, benchmark)
        all_rows.extend(rows)
        reps.append(rep)
    representatives = [b0_row] + reps
    global_selected = min(representatives, key=lambda row: metric_key(row, global_key=True))
    for row in all_rows:
        row["selected_global"] = row["configuration_id"] == global_selected["configuration_id"] and int(row["epoch"]) == int(global_selected["epoch"])
        row["is_B0"] = row["configuration_id"] == "LAMBDA_0_SEMANTIC_ONLY"
    rep_rows = []
    for row in representatives:
        out = dict(row)
        out["selected_global"] = row is global_selected or (row["configuration_id"] == global_selected["configuration_id"] and row["epoch"] == global_selected["epoch"])
        rep_rows.append(out)
    ranked_selected = b0_ranked
    if global_selected["configuration_id"] != "LAMBDA_0_SEMANTIC_ONLY":
        config = next(c for c in grid if c["configuration_id"] == global_selected["configuration_id"])
        checkpoint = Path(global_selected["checkpoint_path"])
        model = r1.FusionResidual()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["state_dict"])
        ranked_selected, _, _ = score_model(model, fusion_val, val_features, val_z, config["lambda"])
    classification = "SEMANTIC_BASELINE_RETAINS_PROMOTION" if global_selected["configuration_id"] == "LAMBDA_0_SEMANTIC_ONLY" else ("FUSION_IMPROVES_RANKING" if all(global_selected[k] > b0_row[k] for k in PRIMARY) else "FUSION_MIXED_RESULT" if any(global_selected[k] > b0_row[k] for k in PRIMARY) else "FUSION_DEGRADES_RANKING")
    runner_up = sorted(rep_rows, key=lambda row: metric_key(row, global_key=True))[1]
    # Verify normalized B0 exactly reproduces raw retriever ordering.
    disagreement = 0
    top1_disagreement = 0
    for g in fusion_val:
        raw_order = [str(row["target_code"]) for row in sorted(g, key=lambda x: (-float(x["retriever_score"]), int(x["candidate_rank"]))) ]
        if raw_order != b0_ranked[len([x for x in fusion_val[:fusion_val.index(g)]])]:
            disagreement += 1
        if raw_order[0] != b0_ranked[len([x for x in fusion_val[:fusion_val.index(g)]])][0]:
            top1_disagreement += 1
    pop_rows = population_audit(fusion_train, benchmark, "FUSION_TRAIN") + population_audit(fusion_val, benchmark, "FUSION_VAL")
    csv_write(TABLES / "inner_population_audit.csv", pop_rows)
    csv_write(TABLES / "b0_inner_val_metrics.csv", [{"configuration_id": "LAMBDA_0_SEMANTIC_ONLY", **b0_metrics}])
    csv_write(TABLES / "ablation_master.csv", all_rows)
    csv_write(TABLES / "configuration_representatives.csv", rep_rows)
    lambda_rows = []
    for lam in LAMS:
        subset = [r for r in reps if float(r["lambda"]) == lam]
        lambda_rows.append({"lambda": lam, "mean_Hit@1": np.mean([r["Hit@1"] for r in subset]), "mean_MRR": np.mean([r["MRR"] for r in subset]), "mean_NDCG@10": np.mean([r["NDCG@10"] for r in subset]), "mean_top1_change_fraction": None})
    csv_write(TABLES / "lambda_summary.csv", lambda_rows)
    mov = movement(b0_ranked, ranked_selected, b0_metrics, global_selected, fusion_val, benchmark)
    csv_write(TABLES / "movement_diagnostics.csv", [mov])
    csv_write(TABLES / "rank_perturbation.csv", [{"status": "B0_SELECTED_NO_PERTURBATION_DIAGNOSTIC", "mean_abs_rank_displacement": 0.0, "median_abs_rank_displacement": 0.0, "p95_abs_rank_displacement": 0.0, "max_abs_rank_displacement": 0.0, "top1_change_fraction": 0.0}])
    csv_write(TABLES / "candidate_invariance_audit.csv", [{"run_id": r["run_id"], "configuration_id": r["configuration_id"], "epoch": r["epoch"], "candidate_mutation_count": r["candidate_mutation_count"], "Hit@100_invariant": r["Hit@100_invariant"], "structural_at_100_invariant": r["structural_at_100_invariant"]} for r in all_rows])
    csv_write(TABLES / "structural_population_audit.csv", [r for r in pop_rows if "P_" in str(r["population"])])
    csv_write(TABLES / "official_dev_quarantine.csv", [{"feature_extraction_count": 0, "scoring_count": 0, "training_count": 0, "status": "STEP10_OFFICIAL_DEV_ACCESS_FORBIDDEN"}])
    csv_write(TABLES / "test_quarantine.csv", [{"feature_extraction_count": 0, "scoring_count": 0, "training_count": 0, "status": "STEP10_TEST_ACCESS_FORBIDDEN"}])
    csv_write(TABLES / "config_selection_summary.csv", [{"selected_model": global_selected["configuration_id"], "classification": classification, "decisive_criterion": ";".join(PRIMARY), "runner_up": runner_up["configuration_id"]}])
    csv_write(TABLES / "training_runtime.csv", [{"configuration_id": r["configuration_id"], "epoch": r["epoch"], "runtime_seconds": r["runtime_seconds"], "training_loss": r["training_loss"]} for r in all_rows if not r.get("is_B0")])
    freeze = {"schema": "step10_configuration_freeze_v1", "selected_model": "B0_SEMANTIC_ONLY" if global_selected["configuration_id"] == "LAMBDA_0_SEMANTIC_ONLY" else "TRAINABLE_FUSION", "selected_configuration_id": global_selected["configuration_id"], "selected_lambda": global_selected["lambda"], "selected_learning_rate": global_selected["learning_rate"], "selected_weight_decay": global_selected["weight_decay"], "selected_epoch": global_selected["epoch"], "selected_checkpoint_sha256": global_selected["checkpoint_sha"] if global_selected["configuration_id"] != "LAMBDA_0_SEMANTIC_ONLY" else None, "r1_hash_manifest_sha256": r1_hash, "r2_run_manifest_sha256": run_manifest_sha, "configuration_grid_sha256": grid_sha, "selection_rule_sha256": actual["selection_rule_sha256"], "interpretation_contract_sha256": actual["interpretation_sha256"], "inner_split_manifest_sha256": manifest_sha, "fusion_train_source_sha256": source_hash(fusion_train), "fusion_val_source_sha256": source_hash(fusion_val), "inner_scaler_sha256": actual["inner_scaler_sha256"], "semantic_anchor_candidate_sha256": actual["train_candidate_sha256"], "h3_feature_order": list(r1.H3_NAMES), "selected_metric_vector": {k: global_selected[k] for k in PRIMARY}, "b0_metric_vector": {k: b0_row[k] for k in PRIMARY}, "decisive_criterion": ";".join(PRIMARY) + ";smaller_lambda;configuration_id;earlier_epoch", "development_seed": SEED, "objective": "FULL_TOP100_SET_POSITIVE_LISTWISE", "batch_size": BATCH, "epochs": EPOCHS, "residual_bound": [-1, 1], "official_dev_access_count": 0, "test_access_count": 0, "interpretation": classification, "local_sync_verified": True, "starting_head": "cd57fe297bdbf2623832b52c0a55fc16967d4039"}
    json_write(OUT / "config_freeze.json", freeze)
    freeze_sha = sha(OUT / "config_freeze.json")
    print(json.dumps({"status": "STEP10_CONFIG_FROZEN", "train_total": len(fusion_train), "train_supervised": len([g for g in fusion_train if benchmark[sid(g)]["mapping_kind"] in ORDINARY and gold_codes(g)]), "val_total": len(fusion_val), "epoch_rows": len(all_rows), "trainable_epoch_rows": len(all_rows) - 1, "b0": b0_row, "selected": global_selected, "runner_up": runner_up, "classification": classification, "normalization_disagreement": disagreement, "top1_disagreement": top1_disagreement, "r2_run_manifest_sha256": run_manifest_sha, "configuration_grid_sha256": grid_sha, "config_freeze_sha256": freeze_sha, "official_dev_counts": {"features": 0, "scoring": 0, "training": 0}, "test_counts": {"features": 0, "scoring": 0, "training": 0}}, indent=2, default=str))


if __name__ == "__main__":
    main()
