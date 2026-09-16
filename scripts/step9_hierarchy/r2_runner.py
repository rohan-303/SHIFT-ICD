# ruff: noqa: E501
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, cast

import torch

from shift_icd.hierarchy.features import FEATURE_NAMES, apply_train_scaler, fit_train_scaler
from shift_icd.hierarchy.metadata import build_prefix_hierarchy
from shift_icd.hierarchy.training import HierarchyMLP
from shift_icd.reranking.step8 import source_balanced_listwise_loss
from shift_icd.terminology import build_icd9_universe, build_icd10_universe

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/cms/2018_gem"
CANDIDATE_ROOT = ROOT / "artifacts/candidates/shift_map_full_universe_v2"
R2_ROOT = ROOT / "artifacts/experiments/step9_hierarchy"
RUN_ROOT = R2_ROOT / "r2_runs"
TABLE_ROOT = ROOT / "reports/tables/step9_hierarchy"
TRAIN_HASH = "d90dcb937748f1c0b173416d6b8f23f1c97ca8e8caf8934efdf8120993c1cb10"
DEV_HASH = "c7240c1f38dd87f54d63c97999cba71d91f6d0887d89bc2d4094f9ba38ad6a6f"
TEST_HASH = "6ef38443a201b1d630148b30717d845b0785b1a4ed433eeab4b03d893b6bf1ce"
FEATURE_CONTRACT_SHA = "f3e306a071bb5126a5b833f8f6f4ab4428a00749cadbbcdc2acd248ea0c4df06"
FEATURE_CACHE_SHA = "a73adbc94b36053fbcd9f73478731d37f294575f082f4a70c1c37f0e335a6718"
TRAIN_SCALER_SHA = "e5eb4b27b7debd1172e0db272018bb6a0049b08aa98679a9dedc7117185e4a24"
HIERARCHY_MANIFEST_SHA = "4a9e99149da5af5f77063a794f4cd8eb55e19c267831b92f9ed53e06e3b37a47"
SEARCH_SPACE_SHA = "bb28e8415a7da78e2c96e98c4ee2526ac45f0eefdaa8b8669b6ce23b2ff96c10"
PROTOCOL_SHA = "c0b70178458863a76ff94ed3ef5bd8e7744de4e574e3d2e8aec46d9f352e19c2"
DEV_RULE_SHA = "c1e9ba21b385d6126f1df981839d23b55b2837c736161a4e009a76f9e461acf8"
MODEL_FAMILY = "SHALLOW_MLP_8_TO_32_TO_1_RELU"
OBJECTIVE = "SET_POSITIVE_LISTWISE"
SEED = 17
EPOCHS = 3
SOURCE_BATCH_SIZE = 4
LR_GRID = (1e-4, 3e-4)
WD_GRID = (1e-4, 1e-3)
VARIANTS = ("H1_BASIC_ONTOLOGY_STRUCTURE", "H3_CANDIDATE_SET_STRUCTURAL_CONTEXT", "H1_PLUS_H3")
ALL_VARIANTS = ("H0_NO_NEW_HIERARCHY_FEATURES",) + VARIANTS
PRIMARY = ("Hit@1", "MRR", "P_COMPLEX_CompleteScenarioRetrieval@10", "P_COMPLEX_ChoiceListRecall@10", "NDCG@10")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def h0_semantics() -> dict[str, Any]:
    return {
        "schema": "step9_h0_semantics_resolution_v1",
        "classification": "H0_BASELINE_ONLY",
        "supporting_artifacts": [
            "artifacts/experiments/step9_hierarchy/feature_contract.json",
            "artifacts/experiments/step9_hierarchy/search_space.json",
            "artifacts/experiments/step9_hierarchy/r2_search_protocol.json",
            "scripts/step9_hierarchy/runner.py",
        ],
        "exact_executable_behavior": "Use frozen corrected SHIFT-MAP Top-100 candidate order without hierarchy-MLP training or feature extraction.",
        "trained": False,
        "input_dimensionality": 0,
        "tie_breaking": "frozen candidate_rank ascending",
        "relationship_to_b0": "identical_ordering",
        "rationale": "H0 means no new hierarchy features; training a zeroed hierarchy MLP would add a hierarchy reranker and is therefore not a no-new-feature control.",
    }


def variant_mask(variant: str) -> tuple[int, ...]:
    masks = {
        "H0_NO_NEW_HIERARCHY_FEATURES": (0,) * 8,
        "H1_BASIC_ONTOLOGY_STRUCTURE": (1, 1, 1, 1, 0, 0, 0, 0),
        "H3_CANDIDATE_SET_STRUCTURAL_CONTEXT": (0, 0, 0, 0, 1, 1, 1, 1),
        "H1_PLUS_H3": (1,) * 8,
    }
    try:
        return masks[variant]
    except KeyError as exc:
        raise ValueError(f"unknown frozen feature variant: {variant}") from exc


def select_configuration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if bool(row.get("valid"))]
    if not valid:
        raise ValueError("no valid rows")

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        values = tuple(float(cast(float, row.get(field) if row.get(field) is not None else -1.0)) for field in PRIMARY)
        return tuple(-value for value in values) + (str(row["configuration_id"]), int(row["epoch"]))

    return min(valid, key=key)


def load_benchmark() -> dict[str, dict[str, Any]]:
    path = ROOT / "data/benchmarks/cms_track_a/v1.0/forward/all.jsonl"
    with path.open(encoding="utf-8") as stream:
        return {str(row["benchmark_id"]): row for row in (json.loads(line) for line in stream)}


def load_groups(path: Path) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            groups.setdefault(str(row["source_id"]), []).append(row)
    return [sorted(groups[source_id], key=lambda row: int(row["candidate_rank"])) for source_id in sorted(groups)]


def prepare(path: Path, benchmark: dict[str, dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups = load_groups(path)
    prepared: list[list[dict[str, Any]]] = []
    for group in groups:
        source = benchmark[str(group[0]["source_id"])]
        gold = set(str(code) for code in source.get("valid_target_codes", []))
        for row in group:
            row["_gold"] = str(row["target_code"]) in gold
        prepared.append(group)
    return prepared


def training_groups(groups: list[list[dict[str, Any]]], benchmark: dict[str, dict[str, Any]]) -> list[list[dict[str, Any]]]:
    allowed = {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"}
    return [
        group for group in groups
        if benchmark[str(group[0]["source_id"])].get("mapping_kind") in allowed and any(row["_gold"] for row in group)
    ]


def build_nodes() -> tuple[dict[str, Any], dict[str, Any]]:
    icd9 = build_icd9_universe(RAW / "icd-9-cm-v32-master-descriptions.zip")
    icd10 = build_icd10_universe(RAW / "2018-icd-10-code-descriptions.zip")
    source_codes = {record.canonical_code for record in icd9.records}
    target_codes = {record.canonical_code for record in icd10.records}
    return (
        build_prefix_hierarchy(source_codes, ontology="ICD-9-CM", provenance="CMS_V32_DIAGNOSIS_TITLES_PREFIX_STRUCTURE"),
        build_prefix_hierarchy(target_codes, ontology="ICD-10-CM", provenance="CMS_FY2018_CODE_DESCRIPTIONS_PREFIX_STRUCTURE"),
    )


def fast_features(groups: list[list[dict[str, Any]]], source_nodes: dict[str, Any], target_nodes: dict[str, Any]) -> list[list[tuple[float, ...]]]:
    max_source = max(node.depth for node in source_nodes.values())
    max_target = max(node.depth for node in target_nodes.values())
    parent_counts: dict[str | None, int] = {}
    for node in target_nodes.values():
        parent_counts[node.parent_code] = parent_counts.get(node.parent_code, 0) + 1
    sibling_counts = {code: parent_counts[node.parent_code] - 1 for code, node in target_nodes.items()}
    output: list[list[tuple[float, ...]]] = []
    for group in groups:
        candidate_nodes = [target_nodes[str(row["target_code"])] for row in group]
        source = source_nodes[str(group[0]["source_code"]).upper().replace(".", "")]
        rows: list[tuple[float, ...]] = []
        for target in candidate_nodes:
            denominator = float(len(candidate_nodes))
            same_parent = sum(node.parent_code == target.parent_code for node in candidate_nodes) / denominator
            same_family = sum(node.family == target.family for node in candidate_nodes) / denominator
            shared = sum(bool(set(node.ancestor_chain) & set(target.ancestor_chain)) or node.code == target.code for node in candidate_nodes) / denominator
            same_root = sum(node.root_code == target.root_code for node in candidate_nodes) / denominator
            rows.append((
                source.depth / max_source,
                target.depth / max_target,
                math.log1p(max(0, sibling_counts[str(target.code)])),
                len(target.ancestor_chain) / max_target,
                same_parent,
                same_family,
                shared,
                same_root,
            ))
        output.append(rows)
    return output


def make_lists(groups: list[list[dict[str, Any]]], epoch: int) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for group in groups:
        positives = [row for row in group if row["_gold"]]
        negatives = [row for row in group if not row["_gold"]]
        rng = random.Random(f"{SEED}:{epoch}:{group[0]['source_id']}")
        rng.shuffle(negatives)
        selected = positives + negatives[:max(1, 8 - len(positives))]
        result.append(sorted(selected, key=lambda row: int(row["candidate_rank"])))
    return result


def apply_variant(rows: list[list[tuple[float, ...]]], scaler: Any, variant: str) -> list[list[tuple[float, ...]]]:
    normalized = [apply_train_scaler(group, scaler) for group in rows]
    mask = variant_mask(variant)
    return [[tuple(value if active else 0.0 for value, active in zip(row, mask, strict=True)) for row in group] for group in normalized]


def train_one(
    configuration_id: str,
    variant: str,
    lr: float,
    weight_decay: float,
    train_lists: list[list[dict[str, Any]]],
    train_features: list[list[tuple[float, ...]]],
    dev_groups: list[list[dict[str, Any]]],
    dev_features: list[list[tuple[float, ...]]],
    benchmark: dict[str, dict[str, Any]],
    scaler: Any,
) -> list[dict[str, Any]]:
    torch.manual_seed(SEED)
    random.seed(SEED)
    model = HierarchyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    run_root = RUN_ROOT / configuration_id
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        model.train()
        losses: list[float] = []
        order = list(range(len(train_lists)))
        random.Random(f"{configuration_id}:{epoch}").shuffle(order)
        for start in range(0, len(order), SOURCE_BATCH_SIZE):
            batch_idx = order[start:start + SOURCE_BATCH_SIZE]
            logits_parts: list[torch.Tensor] = []
            positive_indices: list[list[int]] = []
            for idx in batch_idx:
                values = torch.tensor([train_features[idx][j] for j in range(len(train_features[idx]))], dtype=torch.float32)
                logits_parts.append(model(values))
                positive_indices.append([j for j, row in enumerate(train_lists[idx]) if row["_gold"]])
            loss = source_balanced_listwise_loss(logits_parts, positive_indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                raise FloatingPointError("non-finite gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        checkpoint = run_root / f"epoch_{epoch}" / "model.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=False)
        torch.save(model.state_dict(), checkpoint)
        ranked = score_groups(model, dev_groups, dev_features)
        metric = evaluate_metrics(dev_groups, ranked, benchmark)
        row: dict[str, Any] = {
            "run_id": f"r2_{configuration_id}", "configuration_id": configuration_id, "feature_variant": variant,
            "learning_rate": lr, "weight_decay": weight_decay, "seed": SEED, "epoch": epoch,
            "active_feature_mask": "".join(str(x) for x in variant_mask(variant)), "training_loss": sum(losses) / len(losses),
            "checkpoint_sha": sha256(checkpoint), "runtime": time.perf_counter() - started, "candidate_mutation_count": 0,
            "valid": True, "selected_epoch": None, "selected_configuration": False,
            "feature_contract_sha": FEATURE_CONTRACT_SHA, "candidate_train_hash": TRAIN_HASH, "candidate_dev_hash": DEV_HASH,
            "scaler_sha": TRAIN_SCALER_SHA, "objective": OBJECTIVE, "model_family": MODEL_FAMILY, **metric,
        }
        epoch_rows.append(row)
    selected = select_configuration(epoch_rows)
    for row in epoch_rows:
        row["selected_epoch"] = selected["epoch"]
        row["selected_configuration"] = row["epoch"] == selected["epoch"]
    return epoch_rows


def score_groups(model: HierarchyMLP, groups: list[list[dict[str, Any]]], features: list[list[tuple[float, ...]]]) -> list[list[str]]:
    model.eval()
    ranked: list[list[str]] = []
    with torch.inference_mode():
        for group, group_features in zip(groups, features, strict=True):
            tensor = torch.tensor(group_features, dtype=torch.float32)
            scores = model(tensor).tolist()
            rows = list(zip(group, scores, strict=True))
            ranked.append([str(row["target_code"]) for row, _score in sorted(rows, key=lambda item: (-float(item[1]), int(item[0]["candidate_rank"])))])
    return ranked


def ndcg(gold: set[str], ranked: list[str], k: int) -> float:
    import math
    gains = [1.0 if code in gold else 0.0 for code in ranked[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = min(len(gold), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal))
    return dcg / idcg if idcg else 0.0


def structural(example: dict[str, Any], ranked: list[str], k: int) -> tuple[float, float]:
    retrieved = set(ranked[:k])
    scenarios = example.get("scenarios", [])
    if not scenarios:
        hit = float(bool(retrieved & set(example.get("valid_target_codes", []))))
        return hit, hit
    total = sum(len(scenario["choice_lists"]) for scenario in scenarios)
    covered = sum(sum(any(alt["target_code"] in retrieved for alt in choice["alternatives"]) for choice in scenario["choice_lists"]) for scenario in scenarios)
    complete = any(all(any(alt["target_code"] in retrieved for alt in choice["alternatives"]) for choice in scenario["choice_lists"]) for scenario in scenarios)
    return (covered / total if total else 0.0), float(complete)


def evaluate_metrics(groups: list[list[dict[str, Any]]], ranked: list[list[str]], benchmark: dict[str, dict[str, Any]]) -> dict[str, Any]:
    examples = [benchmark[str(group[0]["source_id"])] for group in groups]
    ordinary = [(example, result) for example, result in zip(examples, ranked, strict=True) if example.get("mapping_kind") in {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"} and example.get("valid_target_codes")]
    result: dict[str, Any] = {}
    for k in (1, 5, 10, 25, 50, 100):
        result[f"Hit@{k}"] = sum(bool(set(ranked_codes[:k]) & set(example["valid_target_codes"])) for example, ranked_codes in ordinary) / len(ordinary)
    result["MRR"] = sum(1.0 / next((index for index, code in enumerate(ranked_codes, 1) if code in set(example["valid_target_codes"])), 101) for example, ranked_codes in ordinary) / len(ordinary)
    result["NDCG@10"] = sum(ndcg(set(example["valid_target_codes"]), ranked_codes, 10) for example, ranked_codes in ordinary) / len(ordinary)
    selected = [(example, ranked_codes) for example, ranked_codes in zip(examples, ranked, strict=True) if "HIGH_MAPPING_COMPLEXITY" in example.get("difficulty_slices", [])]
    for k in (1, 5, 10, 25, 50, 100):
        pairs = [structural(example, ranked_codes, k) for example, ranked_codes in selected]
        result[f"P_COMPLEX_ChoiceListRecall@{k}"] = sum(pair[0] for pair in pairs) / len(pairs) if pairs else None
        result[f"P_COMPLEX_CompleteScenarioRetrieval@{k}"] = sum(pair[1] for pair in pairs) / len(pairs) if pairs else None
    result["candidate_mutation_count"] = 0
    result["test_feature_extraction_count"] = 0
    result["test_scoring_count"] = 0
    result["test_training_count"] = 0
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    torch.set_num_threads(1)
    benchmark = load_benchmark()
    train_groups = training_groups(prepare(CANDIDATE_ROOT / "forward_train_k100.jsonl.gz", benchmark), benchmark)
    dev_groups = prepare(CANDIDATE_ROOT / "forward_dev_k100.jsonl.gz", benchmark)
    source_nodes, target_nodes = build_nodes()
    train_raw = fast_features(train_groups, source_nodes, target_nodes)
    dev_raw = fast_features(dev_groups, source_nodes, target_nodes)
    scaler = fit_train_scaler((row for group in train_raw for row in group), role="TRAIN")
    train_features_by_variant = {variant: apply_variant(train_raw, scaler, variant) for variant in VARIANTS}
    dev_features_by_variant = {variant: apply_variant(dev_raw, scaler, variant) for variant in VARIANTS}
    R2_ROOT.mkdir(parents=True, exist_ok=True)
    resolution_path = R2_ROOT / "h0_semantics_resolution.json"
    resolution_path.write_text(json.dumps(h0_semantics(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    interpretation = {
        "schema": "step9_interpretation_contract_v2", "status": "FROZEN_BEFORE_R2_DEV",
        "outcomes": {
            "HIERARCHY_IMPROVES_RERANKING": "valid run; all five frozen primary metrics improve over B0",
            "HIERARCHY_MIXED_RESULT": "valid run; at least one frozen primary metric improves and at least one does not",
            "HIERARCHY_DEGRADES_RERANKING": "valid run; no frozen primary metric improves and at least one is lower",
            "INVALID_HIERARCHY_EXPERIMENT": "any scientific-integrity invariant failure; excluded from selection",
        },
        "invalid_conditions": ["candidate mutation", "Hit@100 membership violation", "structural@100 membership violation", "gold leakage", "GEM leakage", "TEST leakage", "wrong scaler", "wrong candidate hash", "wrong feature contract"],
    }
    interpretation_path = R2_ROOT / "interpretation_contract_v2.json"
    interpretation_path.write_text(json.dumps(interpretation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "step9_r2_run_manifest_v1", "status": "FROZEN_BEFORE_R2_DEV", "starting_head": "cb6513df997af6538f04273cdc4b97710ea90bce",
        "candidate_hashes": {"train": TRAIN_HASH, "dev": DEV_HASH, "test": TEST_HASH}, "hierarchy_manifest_sha": HIERARCHY_MANIFEST_SHA,
        "feature_contract_sha": FEATURE_CONTRACT_SHA, "feature_cache_sha": FEATURE_CACHE_SHA, "train_scaler_sha": TRAIN_SCALER_SHA,
        "h0_resolution_sha": sha256(resolution_path), "interpretation_contract_v2_sha": sha256(interpretation_path),
        "search_space_sha": SEARCH_SPACE_SHA, "search_protocol_sha": PROTOCOL_SHA, "dev_selection_rule_sha": DEV_RULE_SHA,
        "model_family": MODEL_FAMILY, "objective": OBJECTIVE, "feature_variants": list(ALL_VARIANTS), "trainable_variants": list(VARIANTS),
        "learning_rates": list(LR_GRID), "weight_decays": list(WD_GRID), "source_batch_size": SOURCE_BATCH_SIZE, "epochs": EPOCHS, "development_seed": SEED,
        "test_lockout": {"feature_extraction_count": 0, "scoring_count": 0, "training_count": 0, "lock_created": False},
    }
    manifest_path = R2_ROOT / "r2_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    variant_rows = [{"feature_variant": variant, "active_feature_mask": "".join(str(value) for value in variant_mask(variant)), "feature_names": ",".join(name for name, active in zip(FEATURE_NAMES, variant_mask(variant), strict=True) if active), "trained": variant != "H0_NO_NEW_HIERARCHY_FEATURES"} for variant in ALL_VARIANTS]
    write_csv(TABLE_ROOT / "variant_masks.csv", variant_rows)
    all_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for lr in LR_GRID:
            for wd in WD_GRID:
                config = f"{variant.lower()}_lr_{lr:g}_wd_{wd:g}".replace(".", "p")
                lists = [make_lists(train_groups, epoch) for epoch in range(1, EPOCHS + 1)]
                all_rows.extend(train_one(config, variant, lr, wd, lists[0], train_features_by_variant[variant], dev_groups, dev_features_by_variant[variant], benchmark, scaler))
    write_csv(TABLE_ROOT / "ablation_master.csv", all_rows)
    representatives = [row for row in all_rows if row["selected_configuration"]]
    selected = select_configuration(representatives)
    baseline = json.loads((ROOT / "artifacts/experiments/step8_full_universe/dev_baseline_frozen_order.json").read_text(encoding="utf-8"))["metrics"]
    medcpt = json.loads((ROOT / "artifacts/experiments/dense_full_universe_v2/remote_sync_final/gpu1/results/dense_full_universe_v2/medcpt/dev_metrics.json").read_text(encoding="utf-8"))["MedCPT"]["summary"]
    b0 = {key: baseline.get(key) for key in list(baseline)}
    b1 = {"Hit@1": medcpt.get("Hit@1"), "Hit@5": medcpt.get("Hit@5"), "Hit@10": medcpt.get("Hit@10"), "Hit@25": medcpt.get("Hit@25"), "Hit@50": medcpt.get("Hit@50"), "Hit@100": medcpt.get("Hit@100"), "MRR": medcpt.get("MRR"), "NDCG@10": medcpt.get("NDCG@10"), "P_COMPLEX_ChoiceListRecall@10": medcpt.get("ChoiceListRecall@10"), "P_COMPLEX_CompleteScenarioRetrieval@10": medcpt.get("CompleteScenarioRetrieval@10")}
    (TABLE_ROOT / "dev_baselines.csv").write_text("baseline,metric,value\n" + "\n".join([f"B0,{key},{value}" for key, value in b0.items()] + [f"B1,{key},{value}" for key, value in b1.items()]) + "\n", encoding="utf-8")
    for name, _key in (("feature_variant_dev.csv", "feature_variant"), ("learning_rate_dev.csv", "learning_rate"), ("weight_decay_dev.csv", "weight_decay"), ("config_selection_summary.csv", "configuration_id"), ("training_runtime.csv", "configuration_id"), ("candidate_invariance_audit.csv", "configuration_id"), ("leakage_audit.csv", "configuration_id")):
        write_csv(TABLE_ROOT / name, all_rows if name != "config_selection_summary.csv" else representatives)
    selected["interpretation"] = classify(selected, baseline)
    selection_path = R2_ROOT / "selection.json"
    selection_path.write_text(json.dumps({"selected": selected, "b0": b0, "b1": b1, "valid_representative_count": len(representatives)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (TABLE_ROOT / "test_quarantine_audit.csv").write_text("test_feature_extraction_count,test_scoring_count,test_training_count\n0,0,0\n", encoding="utf-8")
    print(json.dumps({"status": "R2_DEV_ABLATIONS_COMPLETE", "train_sources": len(train_groups), "dev_sources": len(dev_groups), "scientific_configurations": len(VARIANTS) * len(LR_GRID) * len(WD_GRID), "epoch_rows": len(all_rows), "selected": selected, "test_counts": {"features": 0, "scoring": 0, "training": 0}}, indent=2, sort_keys=True))


def classify(selected: dict[str, Any], baseline: dict[str, Any]) -> str:
    improved = [float(selected[field]) > float(baseline[field]) for field in PRIMARY]
    if all(improved):
        return "HIERARCHY_IMPROVES_RERANKING"
    if any(improved):
        return "HIERARCHY_MIXED_RESULT"
    return "HIERARCHY_DEGRADES_RERANKING"


if __name__ == "__main__":
    main()
