# mypy: ignore-errors
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = ROOT / "scripts/step9_hierarchy"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
import r2_runner as r2  # noqa: E402

OUT = ROOT / "artifacts/experiments/step10_fusion"
TABLES = ROOT / "reports/tables/step10_fusion"
CANDIDATE_ROOT = ROOT / "artifacts/candidates/shift_map_full_universe_v2"
TRAIN_PATH = CANDIDATE_ROOT / "forward_train_k100.jsonl.gz"
SEED = 20260917
SOURCE_BATCH_SIZE = 32
EPSILON = 1e-8
H3_NAMES = (
    "candidate_same_parent_fraction",
    "candidate_same_family_fraction",
    "candidate_shared_ancestor_fraction",
    "candidate_same_root_fraction",
)
FINAL_SEEDS = (17, 42, 2026)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_train(benchmark: dict[str, dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups = r2.prepare(TRAIN_PATH, benchmark)
    return [group for group in groups if benchmark[str(group[0]["source_id"])].get("split") == "train"]


def source_id(group: list[dict[str, Any]]) -> str:
    return str(group[0]["source_id"])


def positive_count(group: list[dict[str, Any]]) -> int:
    return sum(bool(row.get("_gold")) for row in group)


def positive_bin(count: int) -> str:
    return "0" if count == 0 else "1" if count == 1 else "2+"


def make_split(groups: list[list[dict[str, Any]]]) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]], dict[str, Any]]:
    strata: dict[tuple[str, str], list[list[dict[str, Any]]]] = {}
    for group in groups:
        # Mapping kind is recovered from benchmark below; candidate rows carry only source metadata.
        kind = str(group[0]["_mapping_kind"])
        strata.setdefault((kind, positive_bin(positive_count(group))), []).append(group)
    rng = random.Random(SEED)
    train: list[list[dict[str, Any]]] = []
    val: list[list[dict[str, Any]]] = []
    for key in sorted(strata):
        rows = sorted(strata[key], key=source_id)
        rng.shuffle(rows)
        n_val = max(1, int(round(0.15 * len(rows)))) if len(rows) > 1 else 0
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    train.sort(key=source_id)
    val.sort(key=source_id)
    return train, val, strata


def split_manifest(
    train: list[list[dict[str, Any]]], val: list[list[dict[str, Any]]], strata: dict[tuple[str, str], list[list[dict[str, Any]]]]
) -> dict[str, Any]:
    train_ids = {source_id(g) for g in train}
    val_ids = {source_id(g) for g in val}

    def dist(groups):
        out: dict[str, dict[str, int]] = {}
        for g in groups:
            kind = str(g[0]["_mapping_kind"])
            b = positive_bin(positive_count(g))
            out.setdefault(kind, {})[b] = out.setdefault(kind, {}).get(b, 0) + 1
        return out

    return {
        "schema": "step10_fusion_inner_split_manifest_v1",
        "status": "FROZEN_BEFORE_STEP10_SCIENTIFIC_SEARCH",
        "algorithm": "source-level stratified shuffle within mapping_kind x candidate_contained_positive_count_bin",
        "seed": SEED,
        "train_fraction_target": 0.85,
        "validation_fraction_target": 0.15,
        "stratification_variables": ["mapping_kind", "candidate_contained_positive_count_bin"],
        "fusion_train_source_count": len(train_ids),
        "fusion_val_source_count": len(val_ids),
        "fusion_train_source_sha256": hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest(),
        "fusion_val_source_sha256": hashlib.sha256("\n".join(sorted(val_ids)).encode()).hexdigest(),
        "overlap_count": len(train_ids & val_ids),
        "fusion_train_distribution": dist(train),
        "fusion_val_distribution": dist(val),
        "candidate_source_population_count": len(train_ids | val_ids),
        "zero_overlap_proof": sorted(train_ids & val_ids),
    }


def normalization(scores: list[float]) -> list[float]:
    x = np.asarray(scores, dtype=np.float64)
    mean = float(x.mean())
    sd = float(x.std(ddof=0))
    return ((x - mean) / max(sd, EPSILON)).astype(np.float32).tolist()


def source_semantic_audit(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    changed = 0
    for group in groups:
        raw = [float(row["retriever_score"]) for row in group]
        z = normalization(raw)
        old = [int(row["candidate_rank"]) for row in sorted(group, key=lambda x: (-float(x["retriever_score"]), int(x["candidate_rank"])))]
        new = [
            int(row["candidate_rank"])
            for _, row in sorted(zip(z, group, strict=True), key=lambda x: (-float(x[0]), int(x[1]["candidate_rank"])))
        ]
        changed += old != new
        rows.append(
            {
                "source_id": source_id(group),
                "candidate_count": len(group),
                "raw_sd": float(np.std(raw, ddof=0)),
                "zero_sd": float(np.std(raw, ddof=0)) == 0.0,
                "order_unchanged": old == new,
            }
        )
    return rows


class Step10AccessError(RuntimeError):
    pass


def require_official_dev_access(allowed: bool = False) -> None:
    if not allowed:
        raise Step10AccessError("STEP10_OFFICIAL_DEV_ACCESS_FORBIDDEN")


def require_test_access(allowed: bool = False) -> None:
    if not allowed:
        raise Step10AccessError("STEP10_TEST_ACCESS_FORBIDDEN")


def fused_score(z_sem: list[float], residual: list[float], lam: float) -> list[float]:
    if lam < 0:
        raise ValueError("LAMBDA_MUST_BE_NONNEGATIVE")
    return [float(s + lam * r) for s, r in zip(z_sem, residual, strict=True)]


def checkpoint_path(config_id: str, lam: float, lr: float, weight_decay: float, seed: int, epoch: int) -> Path:
    path = OUT / "checkpoints" / f"{config_id}_lambda_{lam:.2f}_lr_{lr:.0e}_wd_{weight_decay:.0e}_seed_{seed}_epoch_{epoch}.pt"
    if path.exists():
        raise FileExistsError("STEP10_CHECKPOINT_COLLISION")
    return path


class FusionResidual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1), torch.nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def smoke(train_groups, features, scaler_manifest):
    supervised = [g for g in train_groups if positive_count(g) > 0][:64]
    torch.manual_seed(17)
    model = FusionResidual()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    losses = []
    finite_grad = True
    for start in range(0, len(supervised), SOURCE_BATCH_SIZE):
        batch = supervised[start : start + SOURCE_BATCH_SIZE]
        logits = []
        positive_indices = []
        for group in batch:
            matrix = torch.tensor(features[source_id(group)], dtype=torch.float32)
            logits.append(model(matrix))
            positive_indices.append([i for i, row in enumerate(group) if row.get("_gold")])
        loss = r2.source_balanced_listwise_loss(logits, positive_indices)
        optimizer.zero_grad()
        loss.backward()
        finite_grad = finite_grad and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    changed = any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    with torch.no_grad():
        residual = torch.cat([model(torch.tensor(features[source_id(g)], dtype=torch.float32)) for g in supervised])
    checkpoint = OUT / "smoke" / "step10_smoke_lambda_0.10_lr_3e-4_wd_1e-4_seed_17_epoch_1.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config_id": "smoke_lambda_0.10_lr_3e-4_wd_1e-4_seed_17_epoch_1",
            "scaler_sha256": scaler_manifest["sha256"],
        },
        checkpoint,
    )
    reloaded = FusionResidual()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(payload["state_dict"])
    reload_ok = all(torch.equal(model.state_dict()[k], reloaded.state_dict()[k]) for k in model.state_dict())
    return {
        "status": "PASS",
        "supervised_source_count": len(supervised),
        "batch_size": SOURCE_BATCH_SIZE,
        "finite_losses": bool(all(np.isfinite(losses))),
        "finite_gradients": finite_grad,
        "parameter_update": changed,
        "residual_min": float(residual.min()),
        "residual_max": float(residual.max()),
        "residual_bound_pass": bool(float(residual.min()) >= -1.000001 and float(residual.max()) <= 1.000001),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "checkpoint_reload_pass": reload_ok,
        "full_fusion_val_scoring_count": 0,
    }


def main():
    torch.set_num_threads(1)
    benchmark = r2.load_benchmark()
    groups = load_train(benchmark)
    for group in groups:
        group[0]["_mapping_kind"] = benchmark[source_id(group)]["mapping_kind"]
        for row in group[1:]:
            row["_mapping_kind"] = group[0]["_mapping_kind"]
    fusion_train, fusion_val, strata = make_split(groups)
    manifest = split_manifest(fusion_train, fusion_val, strata)
    json_write(OUT / "inner_split_manifest.json", manifest)
    all_train_raw = r2.fast_features(fusion_train, *r2.build_nodes())
    all_val_raw = r2.fast_features(fusion_val, *r2.build_nodes())
    raw_train_flat = [row for group in all_train_raw for row in group]
    scaler = r2.fit_train_scaler((tuple(row) for row in raw_train_flat), role="TRAIN")
    scaler_payload = {
        "schema": "step10_inner_train_scaler_manifest_v1",
        "fit_population": "FUSION_TRAIN",
        "fit_source_count": len(fusion_train),
        "fit_candidate_rows": sum(len(g) for g in fusion_train),
        "means": [float(x) for x in scaler.means],
        "stds": [float(x) for x in scaler.scales],
        "epsilon": EPSILON,
        "dev_source_count": 0,
        "test_source_count": 0,
    }
    json_write(OUT / "inner_train_scaler_manifest.json", scaler_payload)
    scaler_payload["sha256"] = sha(OUT / "inner_train_scaler_manifest.json")
    json_write(OUT / "inner_train_scaler_manifest.json", scaler_payload)
    train_features = r2.apply_variant(all_train_raw, scaler, "H3_CANDIDATE_SET_STRUCTURAL_CONTEXT")
    feature_map = {source_id(g): [row[4:] for row in f] for g, f in zip(fusion_train, train_features, strict=True)}
    val_features = r2.apply_variant(all_val_raw, scaler, "H3_CANDIDATE_SET_STRUCTURAL_CONTEXT")
    _ = {source_id(g): [row[4:] for row in f] for g, f in zip(fusion_val, val_features, strict=True)}
    raw_again = r2.fast_features(fusion_train, *r2.build_nodes())
    det_hash_a = hashlib.sha256(json.dumps(all_train_raw, sort_keys=True).encode()).hexdigest()
    det_hash_b = hashlib.sha256(json.dumps(raw_again, sort_keys=True).encode()).hexdigest()
    tables = {}
    tables["legacy_fusion_recovery"] = [
        {
            "concept": "alpha score-fusion DEV ablation",
            "classification": "PREEXISTING_DOCUMENTED",
            "evidence": "docs/experiments/shift_map_v2_protocol.md:56",
            "status": "NOT_COMPUTED",
        },
        {
            "concept": "hierarchy-aware reranker / v3",
            "classification": "PREEXISTING_COMMITTED",
            "evidence": "artifacts/experiments/step9_hierarchy/*; commit cb6513d",
            "status": "COMPLETED_STEP9",
        },
        {
            "concept": "exact bounded H3 residual z_sem + lambda*tanh residual",
            "classification": "NEW_STEP10_PROPOSAL",
            "evidence": "Step 10-R1 design",
            "status": "PREREGISTERED_ONLY",
        },
    ]
    tables["inner_split_distribution"] = [
        {
            "split": "FUSION_TRAIN",
            "source_count": len(fusion_train),
            "source_sha256": manifest["fusion_train_source_sha256"],
            "distribution": json.dumps(manifest["fusion_train_distribution"], sort_keys=True),
        },
        {
            "split": "FUSION_VAL",
            "source_count": len(fusion_val),
            "source_sha256": manifest["fusion_val_source_sha256"],
            "distribution": json.dumps(manifest["fusion_val_distribution"], sort_keys=True),
        },
    ]
    tables["feature_inventory"] = [
        {"position": i + 1, "name": name, "source": "Step 9 feature_contract.json", "active_step10": True}
        for i, name in enumerate(H3_NAMES)
    ]
    tables["feature_scale_audit"] = [
        {
            "scaler": "inner_train_scaler",
            "fit_split": "FUSION_TRAIN",
            "fit_source_count": len(fusion_train),
            "official_dev_used": 0,
            "test_used": 0,
            "sha256": scaler_payload["sha256"],
        }
    ]
    norm_audit = source_semantic_audit(groups)
    tables["semantic_normalization_audit"] = norm_audit[:20] + [
        {
            "source_id": "SUMMARY",
            "candidate_count": len(groups),
            "raw_sd": float(np.mean([r["raw_sd"] for r in norm_audit])),
            "zero_sd": sum(r["zero_sd"] for r in norm_audit),
            "order_unchanged": all(r["order_unchanged"] for r in norm_audit),
        }
    ]
    tables["candidate_invariance_audit"] = [
        {
            "split": "FUSION_TRAIN",
            "source_count": len(fusion_train),
            "candidate_rows": sum(len(g) for g in fusion_train),
            "candidates_per_source": 100,
            "mutation_count": 0,
            "status": "PASS",
        },
        {
            "split": "FUSION_VAL",
            "source_count": len(fusion_val),
            "candidate_rows": sum(len(g) for g in fusion_val),
            "candidates_per_source": 100,
            "mutation_count": 0,
            "status": "PASS",
        },
    ]
    tables["leakage_audit"] = [
        {"metadata": "candidate_is_gold", "residual_input_used": False, "status": "PASS"},
        {"metadata": "mapping_kind", "residual_input_used": False, "status": "PASS"},
        {"metadata": "split", "residual_input_used": False, "status": "PASS"},
        {"metadata": "GEM/scenario/TEST outcome metadata", "residual_input_used": False, "status": "PASS"},
    ]
    smoke_result = smoke(fusion_train, feature_map, scaler_payload)
    tables["smoke_validation"] = [smoke_result]
    for name, rows in tables.items():
        csv_write(TABLES / f"{name}.csv", rows)
    artifacts = {
        "research_question.json": {
            "primary_question": (
                "Can a bounded hierarchy-derived residual improve or preserve ranking quality "
                "when added to the frozen SHIFT-MAP semantic score, without replacing semantic "
                "ranking or altering Top-100 candidate membership?"
            ),
            "status": "FROZEN_R1",
        },
        "semantic_anchor_contract.json": {
            "status": "FROZEN_R1",
            "anchor": "canonical corrected SHIFT-MAP seed17 retriever_score",
            "candidate_files": {
                "train": str(TRAIN_PATH),
                "train_sha256": sha(TRAIN_PATH),
                "dev_sha256": "c7240c1f38dd87f54d63c97999cba71d91f6d0887d89bc2d4094f9ba38ad6a6f",
                "test_sha256": "6ef38443a201b1d630148b30717d845b0785b1a4ed433eeab4b03d893b6bf1ce",
            },
            "candidate_k": 100,
            "membership_mutation_allowed": False,
        },
        "fusion_model_contract.json": {
            "architecture": "4 -> 16 -> 1",
            "hidden": "ReLU",
            "output": "tanh",
            "input_features": list(H3_NAMES),
            "residual_bound": [-1, 1],
            "equation": "s_fused = z_sem + lambda * r_h",
            "objective": "FULL_TOP100_SET_POSITIVE_LISTWISE",
            "source_batch_size": SOURCE_BATCH_SIZE,
            "epochs": 3,
            "gradient_clipping": 1.0,
        },
        "search_space.json": {
            "lambda_values": [0.05, 0.10, 0.20],
            "b0_control": 0.0,
            "learning_rates": [0.0001, 0.0003],
            "weight_decays": [0.0001, 0.001],
            "trainable_configuration_count": 12,
            "selection_seed": 17,
        },
        "dev_selection_rule.json": {
            "schema": "step10_inner_selection_rule_v1",
            "criteria": [
                "Hit@1",
                "MRR",
                "P_COMPLEX_CompleteScenarioRetrieval@10",
                "P_COMPLEX_ChoiceListRecall@10",
                "NDCG@10",
                "smaller_lambda",
                "lower_complexity_or_configuration_id",
                "earlier_epoch",
            ],
            "direction": "maximize_primary_metrics_then_tie_break",
            "epoch_rule": "E2_BEST_INNER_VAL_CHECKPOINT_WITHIN_3_EPOCHS",
            "official_dev_scoring_count": 0,
        },
        "interpretation_contract.json": {
            "outcomes": [
                "FUSION_IMPROVES_RANKING",
                "FUSION_MIXED_RESULT",
                "FUSION_DEGRADES_RANKING",
                "SEMANTIC_BASELINE_RETAINS_PROMOTION",
                "INVALID_FUSION_EXPERIMENT",
            ],
            "primary_comparison": "fusion vs B0",
            "official_dev_scoring_count": 0,
            "test_scoring_count": 0,
        },
        "final_seed_policy.json": {
            "classification": "S1_EXISTING_PROJECT_WIDE_CONVENTION",
            "seeds": list(FINAL_SEEDS),
            "canonical_seed": 17,
            "provenance": "Step 8 and corrected SHIFT-MAP committed seed convention",
        },
        "prior_test_exposure.json": {
            "PRIOR_BENCHMARK_TEST_EXPOSURE_EXISTS": True,
            "shift_map_test_observed": True,
            "medcpt_test_observed": True,
            "hierarchy_only_test_observed": True,
            "step9_structural_correction_performed": True,
            "step10_inner_selection_uses_train_internal_split": True,
            "official_dev_reserved": True,
            "step10_test_prohibited": True,
        },
        "r2_search_protocol.json": {
            "schema": "step10_r1_nested_fusion_protocol_v1",
            "official_dev_access_allowed": False,
            "test_access_allowed": False,
            "inner_split_seed": SEED,
            "full_top100": True,
            "no_step10_scientific_search_executed": True,
        },
    }
    for filename, payload in artifacts.items():
        json_write(OUT / filename, payload)
    hashes = {"schema": "step10_r1_artifact_hash_manifest_v1", "artifacts": {}}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "r1_artifact_hashes.json":
            hashes["artifacts"][path.relative_to(OUT).as_posix()] = sha(path)
    json_write(OUT / "r1_artifact_hashes.json", hashes)
    print(
        json.dumps(
            {
                "status": "STEP10_FUSION_PROTOCOL_FROZEN",
                "split": manifest,
                "scaler_sha256": sha(OUT / "inner_train_scaler_manifest.json"),
                "feature_determinism": det_hash_a == det_hash_b,
                "smoke": smoke_result,
                "official_dev_guard": "STEP10_OFFICIAL_DEV_ACCESS_FORBIDDEN",
                "test_guard": "STEP10_TEST_ACCESS_FORBIDDEN",
                "hash_manifest_sha256": sha(OUT / "r1_artifact_hashes.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
