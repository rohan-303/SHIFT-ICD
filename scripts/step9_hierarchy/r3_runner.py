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

import torch

ROOT = Path(__file__).resolve().parents[2]
R2_ROOT = ROOT / "artifacts/experiments/step9_hierarchy"
RUN_ROOT = R2_ROOT / "final_seed_runs"
TABLE_ROOT = ROOT / "reports/tables/step9_hierarchy"
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2_runner as r2  # noqa: E402

SEEDS = (17, 42, 2026)
EPOCHS = 3
LR = 3e-4
WEIGHT_DECAY = 1e-4
BATCH = 4
VARIANT = "H3_CANDIDATE_SET_STRUCTURAL_CONTEXT"
PRIMARY = r2.PRIMARY


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def make_lists(groups: list[list[dict[str, Any]]], seed: int, epoch: int) -> list[list[dict[str, Any]]]:
    output = []
    for group in groups:
        positives = [row for row in group if row["_gold"]]
        negatives = [row for row in group if not row["_gold"]]
        rng = random.Random(f"{seed}:{epoch}:{group[0]['source_id']}")
        rng.shuffle(negatives)
        chosen = positives + negatives[: max(1, 8 - len(positives))]
        output.append(sorted(chosen, key=lambda row: int(row["candidate_rank"])))
    return output


def train_seed(seed: int, train_groups, train_features, dev_groups, dev_features, benchmark):
    torch.manual_seed(seed)
    random.seed(seed)
    model = r2.HierarchyMLP()
    init_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    seed_root = RUN_ROOT / f"seed_{seed}"
    if seed_root.exists():
        raise RuntimeError(f"STEP9_CHECKPOINT_COLLISION:{seed_root}")
    seed_root.mkdir(parents=True)
    rows = []
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        lists = make_lists(train_groups, seed, epoch)
        model.train()
        order = list(range(len(lists)))
        random.Random(f"{seed}:{epoch}:order").shuffle(order)
        losses = []
        for start in range(0, len(order), BATCH):
            parts = []
            positive_indices = []
            for idx in order[start : start + BATCH]:
                values = torch.tensor(train_features[idx], dtype=torch.float32)
                parts.append(model(values))
                positive_indices.append([j for j, row in enumerate(lists[idx]) if row["_gold"]])
            loss = r2.source_balanced_listwise_loss(parts, positive_indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError(f"non-finite gradient seed={seed} epoch={epoch}")
            optimizer.step()
            losses.append(float(loss.detach()))
        ckpt = seed_root / f"epoch_{epoch}" / "model.pt"
        ckpt.parent.mkdir()
        torch.save(model.state_dict(), ckpt)
        ranked = r2.score_groups(model, dev_groups, dev_features)
        metrics = r2.evaluate_metrics(dev_groups, ranked, benchmark)
        row = {
            "run_id": f"step9_final_h3_seed_{seed}",
            "stage": "FINAL_SEED_DEV",
            "seed": seed,
            "epoch": epoch,
            "feature_variant": VARIANT,
            "active_feature_mask": "00001111",
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "training_loss": sum(losses) / len(losses),
            "checkpoint_sha256": sha256(ckpt),
            "initialization_state_sha256": init_hash,
            "runtime_seconds": time.perf_counter() - started,
            "valid": True,
            "candidate_mutation_count": 0,
            "candidate_dev_hash": r2.DEV_HASH,
            "candidate_train_hash": r2.TRAIN_HASH,
            "scaler_sha256": r2.TRAIN_SCALER_SHA,
            "feature_contract_sha256": r2.FEATURE_CONTRACT_SHA,
            "config_freeze_sha256": "4eb057b3f7badfc7c56062f4d921999449ccdb0e3481c8d4cff7e66a76398659",
            "objective": r2.OBJECTIVE,
            "model_family": r2.MODEL_FAMILY,
            "training_source_count": len(train_groups),
            "expanded_source_count": sum(len([x for x in g if x["_gold"]]) > 8 for g in lists),
            "dropped_positive_count": 0,
            "positive_negative_collision_count": 0,
            "maximum_list_length": max(len(g) for g in lists),
            **metrics,
        }
        rows.append(row)
    selected = r2.select_configuration([dict(row, configuration_id=f"seed_{seed}") for row in rows])
    for row in rows:
        row["selected_epoch"] = selected["epoch"]
        row["selected_configuration"] = row["epoch"] == selected["epoch"]
    return rows, init_hash


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    torch.set_num_threads(1)
    benchmark = r2.load_benchmark()
    train_groups = r2.training_groups(r2.prepare(r2.CANDIDATE_ROOT / "forward_train_k100.jsonl.gz", benchmark), benchmark)
    dev_groups = r2.prepare(r2.CANDIDATE_ROOT / "forward_dev_k100.jsonl.gz", benchmark)
    source_nodes, target_nodes = r2.build_nodes()
    train_raw = r2.fast_features(train_groups, source_nodes, target_nodes)
    dev_raw = r2.fast_features(dev_groups, source_nodes, target_nodes)
    scaler = r2.fit_train_scaler((row for group in train_raw for row in group), role="TRAIN")
    train_features = r2.apply_variant(train_raw, scaler, VARIANT)
    dev_features = r2.apply_variant(dev_raw, scaler, VARIANT)
    all_rows = []
    init_hashes = {}
    for seed in SEEDS:
        rows, init_hash = train_seed(seed, train_groups, train_features, dev_groups, dev_features, benchmark)
        all_rows.extend(rows)
        init_hashes[str(seed)] = init_hash
    write_csv(TABLE_ROOT / "final_seed_dev.csv", all_rows)
    selected = [row for row in all_rows if row["selected_configuration"]]
    manifest = {
        "schema": "step9_final_seed_manifest_v1",
        "status": "FINAL_SEEDS_DEV_COMPLETE_BEFORE_TEST_LOCK",
        "seeds": list(SEEDS),
        "canonical_seed": 17,
        "initialization_state_sha256": init_hashes,
        "selected_rows": selected,
        "checkpoint_paths": {str(seed): str(RUN_ROOT / f"seed_{seed}" / "epoch_3" / "model.pt") for seed in SEEDS},
        "config_freeze_sha256": "4eb057b3f7badfc7c56062f4d921999449ccdb0e3481c8d4cff7e66a76398659",
        "final_seed_policy_sha256": "f2da29bec862a821820c0974cee467fac37fb9db961c809d1739e56c5154c2e3",
        "r3_addendum_sha256": "997fac62612b1482e68a1bd354754a32011ca94f74ac3b6382ce8dbcf3b0e519",
        "candidate_hashes": {"train": r2.TRAIN_HASH, "dev": r2.DEV_HASH, "test": r2.TEST_HASH},
        "feature_contract_sha256": r2.FEATURE_CONTRACT_SHA,
        "hierarchy_manifest_sha256": r2.HIERARCHY_MANIFEST_SHA,
        "train_scaler_sha256": r2.TRAIN_SCALER_SHA,
        "test_feature_extraction_count_before_lock": 0,
        "test_scoring_count_before_lock": 0,
        "test_training_count_before_lock": 0,
        "test_lock_created": False,
        "independent_initialization": True,
    }
    manifest_path = R2_ROOT / "final_seed_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "rows": len(all_rows),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "init_hashes": init_hashes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
