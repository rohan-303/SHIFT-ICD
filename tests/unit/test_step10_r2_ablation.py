from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ruff: noqa: E501

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/step10_fusion/r2_ablation.py"
spec = importlib.util.spec_from_file_location("step10_r2_ablation", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_corrected_mrr_missing_gold_is_zero() -> None:
    benchmark = {"s": {"mapping_kind": "SINGLE_EXACT", "valid_target_codes": ["gold"]}}
    groups = [[{"source_id": "s", "target_code": "not_gold", "_gold": False}]]
    assert module.evaluate(groups, [["not_gold"]], benchmark)["MRR"] == 0.0


def test_canonical_population_contract_does_not_require_gold() -> None:
    assert module.canonical_population("COMBINATION") == "P_COMBINATION"
    assert module.canonical_population("COMBINATION_WITH_ALTERNATIVES") == "P_COMBINATION_WITH_ALTERNATIVES"
    assert module.canonical_population("MULTI_SCENARIO") == "P_COMPLEX"


def test_b0_wins_complete_primary_tie_due_to_lambda_zero() -> None:
    b0 = {"configuration_id": "LAMBDA_0_SEMANTIC_ONLY", "lambda": 0.0, "epoch": 0, **{m: 0.5 for m in module.PRIMARY}}
    fusion = {"configuration_id": "fusion_lambda_0p05_lr_1e-04_wd_1e-04", "lambda": 0.05, "epoch": 1, **{m: 0.5 for m in module.PRIMARY}}
    assert min([fusion, b0], key=lambda row: module.metric_key(row, global_key=True))["configuration_id"] == "LAMBDA_0_SEMANTIC_ONLY"


def test_configuration_grid_is_exactly_twelve() -> None:
    grid = module.make_config_grid()
    assert len(grid) == 12
    assert {row["lambda"] for row in grid} == {0.05, 0.10, 0.20}
    assert {row["learning_rate"] for row in grid} == {1e-4, 3e-4}
    assert {row["weight_decay"] for row in grid} == {1e-4, 1e-3}


def test_selection_reconstruction_is_deterministic() -> None:
    rows = [
        {"configuration_id": "b", "epoch": 2, "lambda": 0.1, **{m: 0.5 for m in module.PRIMARY}},
        {"configuration_id": "a", "epoch": 1, "lambda": 0.1, **{m: 0.5 for m in module.PRIMARY}},
    ]
    assert module.select_representative(rows)["configuration_id"] == "a"


def test_step10_access_guards_remain_fail_closed() -> None:
    with pytest.raises(module.r1.Step10AccessError, match="STEP10_OFFICIAL_DEV_ACCESS_FORBIDDEN"):
        module.r1.require_official_dev_access()
    with pytest.raises(module.r1.Step10AccessError, match="STEP10_TEST_ACCESS_FORBIDDEN"):
        module.r1.require_test_access()
