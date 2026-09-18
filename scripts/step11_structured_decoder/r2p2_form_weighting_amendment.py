# ruff: noqa: E501
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/experiments/step11_structured_decoder"
TABLES = ROOT / "reports/tables/step11_structured_decoder"
EXPECTED_HEAD = "eea9ef1b4d4b28d4a242c391146ad96fbc18cacd"
CLASS_ORDER = ("NO_MAP", "SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE", "COMBINATION", "COMBINATION_WITH_ALTERNATIVES")
EXPECTED_TRAIN_SHA = "150a57de2e9101ee9bb5c1699bbb7b3a7f9b1fc1d0863018a8766808c151f24c"
BLOCKER_SHA = "67112cf32f1245a70d3febfc5aebe65900a7a877a3bbdf7270f0577778eca319"
BLOCKED_REPORT_SHA = "902870284d5f9d8ae1bfb21e94497f7fb6417c2cf7c6a3310fd8d6addf66e6c9"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    blocker = json.loads((OUT / "r2_execution_preflight_blocker.json").read_text(encoding="utf-8"))
    if sha(OUT / "r2_execution_preflight_blocker.json") != BLOCKER_SHA or sha(ROOT / "reports/STEP_11_R2_EXECUTION_PREFLIGHT_BLOCKED.md") != BLOCKED_REPORT_SHA:
        raise RuntimeError("STEP11_R2_X001_BLOCKER_HASH_MISMATCH")
    if blocker["counts_at_stop"]["full_r2_configurations_trained"] != 0 or blocker["counts_at_stop"]["scientific_epoch_rows"] != 0:
        raise RuntimeError("STEP11_R2_PRE_OUTCOME_PROOF_FAILED")

    step10 = load_module("step10_r1_for_r2p2", ROOT / "scripts/step10_fusion/r1_preflight.py")
    benchmark_runner = load_module("step9_r2_for_r2p2", ROOT / "scripts/step9_hierarchy/r2_runner.py")
    benchmark = benchmark_runner.load_benchmark()
    groups = benchmark_runner.prepare(step10.TRAIN_PATH, benchmark)
    for group in groups:
        group[0]["_mapping_kind"] = str(benchmark[str(group[0]["source_id"])]["mapping_kind"])
        for row in group[1:]:
            row["_mapping_kind"] = group[0]["_mapping_kind"]
    fusion_train, fusion_val, _ = step10.make_split(groups)
    train_ids = {str(g[0]["source_id"]) for g in fusion_train}
    val_ids = {str(g[0]["source_id"]) for g in fusion_val}
    train_sha = hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest()
    val_sha = hashlib.sha256("\n".join(sorted(val_ids)).encode()).hexdigest()
    if train_sha != EXPECTED_TRAIN_SHA or len(fusion_train) != 8666 or len(fusion_val) != 1531 or train_ids & val_ids:
        raise RuntimeError("STEP11_R2_INNER_SPLIT_MISMATCH")

    counts = {name: 0 for name in CLASS_ORDER}
    for group in fusion_train:
        form = str(benchmark[str(group[0]["source_id"])]["mapping_kind"])
        if form not in counts:
            raise RuntimeError(f"STEP11_UNKNOWN_MAPPING_FORM:{form}")
        counts[form] += 1
    if counts != {"NO_MAP": 251, "SINGLE_EXACT": 2095, "SINGLE_APPROXIMATE": 4310, "ALTERNATIVE": 1613, "COMBINATION": 190, "COMBINATION_WITH_ALTERNATIVES": 207}:
        raise RuntimeError("STEP11_R2_TRAIN_FORM_COUNTS_MISMATCH")

    total = float(sum(counts.values()))
    classes = float(len(CLASS_ORDER))
    weights = {name: total / (classes * float(counts[name])) for name in CLASS_ORDER}
    normalization = sum(float(counts[name]) * weights[name] for name in CLASS_ORDER) / total
    if abs(normalization - 1.0) > 1e-15:
        raise RuntimeError("STEP11_FORM_WEIGHT_NORMALIZATION_FAILED")

    weighting = {
        "schema": "step11_form_weighting_contract_v1",
        "status": "FROZEN_PRE_OUTCOME",
        "class_order": list(CLASS_ORDER),
        "fusion_train_source_count": len(fusion_train),
        "fusion_train_source_sha256": train_sha,
        "class_counts": counts,
        "N": int(total),
        "C": len(CLASS_ORDER),
        "formula": "w_c = N / (C * n_c)",
        "weights_float64": weights,
        "weighted_average_normalization": normalization,
        "normalization_property": "sum_c(n_c * w_c) / N = 1; no additional renormalization",
        "scope": ["L_form"],
        "unweighted_primary": {name: 1.0 for name in CLASS_ORDER},
        "zero_count_behavior": "raise STEP11_FORM_WEIGHT_UNDEFINED_ZERO_CLASS",
        "dtype_policy": "compute counts/weights in float64; cast weights to form_logits dtype/device at training",
        "fusion_val_rows_used": 0,
        "official_dev_rows_used": 0,
        "test_rows_used": 0,
    }
    write_json(OUT / "form_weighting_contract_v1.json", weighting)
    weighting_sha = sha(OUT / "form_weighting_contract_v1.json")

    loss_v2 = json.loads((OUT / "loss_contract_v2.json").read_text(encoding="utf-8"))
    loss_v3 = {
        **loss_v2,
        "schema": "step11_loss_contract_v3",
        "supersedes": "loss_contract_v2",
        "imbalance": {
            "UNWEIGHTED_PRIMARY": "six unit form weights",
            "TRAIN_DERIVED_FORM_WEIGHTED": "balanced inverse-frequency weights from form_weighting_contract_v1",
        },
        "form_weighting_contract_sha256": weighting_sha,
        "scope": "only L_form; all structural components remain unchanged",
        "zero_count_behavior": "STEP11_FORM_WEIGHT_UNDEFINED_ZERO_CLASS",
    }
    write_json(OUT / "loss_contract_v3.json", loss_v3)
    loss_sha = sha(OUT / "loss_contract_v3.json")

    space_v3 = json.loads((OUT / "search_space_v3.json").read_text(encoding="utf-8"))
    space_v4 = {
        **space_v3,
        "schema": "step11_search_space_v4",
        "loss_contract_v3_sha256": loss_sha,
        "form_weighting_contract_v1_sha256": weighting_sha,
    }
    write_json(OUT / "search_space_v4.json", space_v4)
    space_sha = sha(OUT / "search_space_v4.json")

    grid_v2 = json.loads((OUT / "r2_configuration_grid_v2.json").read_text(encoding="utf-8"))
    grid_v3 = {
        **grid_v2,
        "schema": "step11_r2_configuration_grid_v3",
        "loss_contract_v3_sha256": loss_sha,
        "form_weighting_contract_v1_sha256": weighting_sha,
        "search_space_v4_sha256": space_sha,
    }
    for config in grid_v3["configurations"]:
        config["loss_contract_sha256"] = loss_sha
        config["form_weighting_contract_sha256"] = "UNIT_WEIGHTS" if config["imbalance_variant"] == "UNWEIGHTED_PRIMARY" else weighting_sha
    write_json(OUT / "r2_configuration_grid_v3.json", grid_v3)
    grid_sha = sha(OUT / "r2_configuration_grid_v3.json")

    protocol_v2 = json.loads((OUT / "r2_search_protocol_v2.json").read_text(encoding="utf-8"))
    protocol_v3 = {
        **protocol_v2,
        "schema": "step11_r2_search_protocol_v3",
        "supersedes": "r2_search_protocol_v2",
        "loss_contract_v3_sha256": loss_sha,
        "form_weighting_contract_v1_sha256": weighting_sha,
        "search_space_v4_sha256": space_sha,
        "configuration_grid_v3_sha256": grid_sha,
        "status": "FROZEN_PRE_OUTCOME",
        "scientific_execution_count_at_freeze": 0,
    }
    write_json(OUT / "r2_search_protocol_v3.json", protocol_v3)
    protocol_sha = sha(OUT / "r2_search_protocol_v3.json")

    r1b_expected = {
        "model_contract_v2_sha256": "7c5d920934305985195827ff0396554c54e13cb126db115e5cf17fd384dd109d",
        "assembler_contract_v2_sha256": "62b22e20798208726ef6dbd01444ee4968a1abf03a2035d0c0c7175327c9bc05",
        "loss_contract_v2_sha256": "baf5097c321f71df2063d0403457d058a7a38e72052c5cfebeae3ed952bfc33b",
        "cardinality_target_contract_v2_sha256": "2f2fb0e6be37bc2517d22833e6aa95d138e0f4ee147e474a68b0cfa5b7942799",
        "r1b_repair_manifest_sha256": "0b0dc365f549ef34b1bad04ed0c173dedbbdedcc936de3f095be92f912c4c4a4",
        "feature_provenance_v1_sha256": "adf7030faea7ff7f3dd3d9cd6af3fb5b03dcc7cba8e576e76545d240c10f6165",
        "baseline_contract_v2_sha256": "16587bc3c10f6e3b641bc65fa85d570022eda6ce33220dc0fbb5856fb5158f3b",
    }
    r1b_actual = {key: sha(OUT / filename) for key, filename in {
        "model_contract_v2_sha256": "model_contract_v2.json",
        "assembler_contract_v2_sha256": "assembler_contract_v2.json",
        "loss_contract_v2_sha256": "loss_contract_v2.json",
        "cardinality_target_contract_v2_sha256": "cardinality_target_contract_v2.json",
        "r1b_repair_manifest_sha256": "r1b_protocol_repair_manifest.json",
        "feature_provenance_v1_sha256": "feature_provenance_v1.json",
        "baseline_contract_v2_sha256": "baseline_contract_v2.json",
    }.items()}
    if r1b_actual != r1b_expected:
        raise RuntimeError("BLOCKED_STEP11_R2_R1B_OR_R2P_CONTRACT_MISMATCH")

    protected = {
        "metric_contract_v2_sha256": sha(OUT / "metric_contract_v2.json"),
        "selection_rule_sha256": sha(OUT / "selection_rule.json"),
        "interpretation_contract_sha256": sha(OUT / "interpretation_contract.json"),
    }
    expected_protected = {
        "metric_contract_v2_sha256": "19fb3122266a3c6f11cec7e58d1bb0f5860faf2dbba113d7c3d099037de740a6",
        "selection_rule_sha256": "f17bd93e0a864070ee13b015c3dc72770ab5c3444f42ebbcecf2ca7eb98cdc8a",
        "interpretation_contract_sha256": "ecad6ee74d4bd48e3efca7f45597de729dae0fb8079c0dd8a62087c9f9efb5ef",
    }
    if protected != expected_protected:
        raise RuntimeError("BLOCKED_STEP11_R2_PROTECTED_CONTRACT_CHANGED")

    write_csv(TABLES / "r2p2_form_weights.csv", [{"class_index": i, "mapping_form": name, "train_count": counts[name], "weight": format(weights[name], ".17g"), "formula_numerator": int(total), "formula_denominator": len(CLASS_ORDER) * counts[name]} for i, name in enumerate(CLASS_ORDER)])
    write_csv(TABLES / "r2p2_configuration_grid.csv", [{"configuration_id": c["configuration_id"], "learning_rate": c["learning_rate"], "imbalance_variant": c["imbalance_variant"], "loss_contract_sha256": c["loss_contract_sha256"], "form_weighting_contract_sha256": c["form_weighting_contract_sha256"], "epochs": "1,2,3"} for c in grid_v3["configurations"]])

    amendment = {
        "schema": "step11_r2p2_protocol_amendment_manifest_v1",
        "status": "FROZEN_PRE_OUTCOME",
        "r1b_commit": "43db7f4c069000e1df783b4f7d7c027f4cb5a5f3",
        "r2_protocol_completion_commit": "4bacfbfdcbda3903d16bbd1d841c18f901941a2f",
        "r2_execution_blocker_commit": "eea9ef1b4d4b28d4a242c391146ad96fbc18cacd",
        "r2_x001_blocker_sha256": BLOCKER_SHA,
        "r2_x001_blocked_report_sha256": BLOCKED_REPORT_SHA,
        "reason": "TRAIN_DERIVED_FORM_WEIGHTED lacked an executable formula, scaling, scope, and zero-count behavior",
        "scientific_counts_at_amendment": {"feature_caches": 0, "configurations_trained": 0, "epoch_rows": 0, "checkpoints": 0, "fusion_val_evaluations": 0, "official_dev_evaluations": 0, "test_scoring": 0},
        "form_weighting_contract_v1_sha256": weighting_sha,
        "loss_contract_v3_sha256": loss_sha,
        "search_space_v4_sha256": space_sha,
        "configuration_grid_v3_sha256": grid_sha,
        "r2_search_protocol_v3_sha256": protocol_sha,
        **protected,
        "fusion_train_source_sha256": train_sha,
        "fusion_train_class_counts": counts,
        "statement": "FORM WEIGHTING WAS EXPLICITLY FROZEN BEFORE FEATURE-CACHE MATERIALIZATION, TRAINING, CHECKPOINT CREATION, OR FUSION_VAL SCIENTIFIC SCORING.",
    }
    write_json(OUT / "r2p2_protocol_amendment_manifest.json", amendment)
    amendment_sha = sha(OUT / "r2p2_protocol_amendment_manifest.json")

    preflight = {
        "schema": "step11_r2p2_complete_preflight_v1",
        "status": "STEP11_R2_EXECUTION_PROTOCOL_COMPLETE_FROZEN",
        "scientific_execution": False,
        "form_weighting_rule_frozen_before_any_r2_scientific_result": True,
        "starting_head": EXPECTED_HEAD,
        "class_order": list(CLASS_ORDER),
        "fusion_train_count": len(fusion_train),
        "fusion_train_source_sha256": train_sha,
        "fusion_val_count": len(fusion_val),
        "fusion_val_source_sha256": val_sha,
        "overlap_count": 0,
        "future_configuration_count": 4,
        "future_epoch_row_count": 12,
        "contracts": {"form_weighting_contract_v1": weighting_sha, "loss_contract_v3": loss_sha, "search_space_v4": space_sha, "configuration_grid_v3": grid_sha, "r2_search_protocol_v3": protocol_sha, **r1b_actual, **protected},
        "counts_at_freeze": amendment["scientific_counts_at_amendment"],
        "official_dev_access": "STEP11_OFFICIAL_DEV_ACCESS_FORBIDDEN",
        "test_access": "STEP11_TEST_ACCESS_FORBIDDEN",
    }
    write_json(OUT / "r2p2_complete_preflight.json", preflight)
    preflight_sha = sha(OUT / "r2p2_complete_preflight.json")

    report = f"""# STEP 11-R2P2 — FORM-WEIGHTING PROTOCOL AMENDMENT

## Status

`STEP11_R2_EXECUTION_PROTOCOL_COMPLETE_FROZEN`

The amendment occurred before feature-cache materialization, training, checkpoint creation, FUSION_VAL scientific scoring, official DEV access, and TEST access.

## R2-X001 closure

The previous blocker identified that `TRAIN_DERIVED_FORM_WEIGHTED` lacked an executable weighting formula, scaling rule, loss scope, and zero-count behavior. Those fields are now frozen in `form_weighting_contract_v1.json`.

## Frozen formula

For class `c`, `w_c = N / (C * n_c)`, using only FUSION_TRAIN source counts. `N=8666`, `C=6`, and the class order is the authoritative six-form order. Computation and manifest storage use float64. The weighted-average property `sum_c(n_c*w_c)/N = 1` was verified; no additional normalization is applied.

## Counts and weights

{json.dumps({"counts": counts, "weights_float64": weights}, indent=2, sort_keys=True)}

The weighting applies only to `L_form`. Structural count, activity, slot, assignment, and flat-set losses are unchanged. The unweighted variant uses six unit weights. A zero-count required class raises `STEP11_FORM_WEIGHT_UNDEFINED_ZERO_CLASS`.

## Versioned contracts

- `form_weighting_contract_v1.json`: `{weighting_sha}`
- `loss_contract_v3.json`: `{loss_sha}`
- `search_space_v4.json`: `{space_sha}`
- `r2_configuration_grid_v3.json`: `{grid_sha}`
- `r2_search_protocol_v3.json`: `{protocol_sha}`
- amendment manifest: `{amendment_sha}`
- complete preflight: `{preflight_sha}`

Metric, selection, and interpretation contracts remained unchanged: `{protected}`.

## Scientific grid preserved

The grid remains four configurations: two learning rates (`0.0001`, `0.0003`) × two imbalance variants. The future execution remains three epochs per configuration, for 12 expected scientific epoch rows. No R2 scientific result exists.

## Validation

TRAIN source count: 8,666; source SHA: `{train_sha}`. FUSION_VAL remains 1,531 sources with source SHA `{val_sha}` and zero overlap. DEV and TEST remain quarantined. Full R2 feature caches: 0; configurations trained: 0; epoch rows: 0; checkpoints: 0; FUSION_VAL scientific evaluations: 0; official DEV evaluations: 0; TEST scoring: 0.

The amended protocol authorizes a future R2 execution preflight, not R2 training in this milestone.
"""
    (ROOT / "reports/STEP_11_R2P2_FORM_WEIGHTING_PROTOCOL_AMENDMENT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": preflight["status"], "weights": weights, "weighting_sha256": weighting_sha, "loss_sha256": loss_sha, "search_space_sha256": space_sha, "grid_sha256": grid_sha, "protocol_sha256": protocol_sha, "amendment_sha256": amendment_sha, "preflight_sha256": preflight_sha, "scientific_counts": amendment["scientific_counts_at_amendment"]}, sort_keys=True))


if __name__ == "__main__":
    main()
