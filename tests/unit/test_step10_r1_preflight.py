from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/step10_fusion/r1_preflight.py"
spec = importlib.util.spec_from_file_location("step10_r1_preflight", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_semantic_normalization_preserves_order_and_zero_sd() -> None:
    raw = [0.1, 0.4, 0.2, -0.1]
    z = module.normalization(raw)
    assert sorted(range(len(raw)), key=lambda i: raw[i], reverse=True) == sorted(range(len(z)), key=lambda i: z[i], reverse=True)
    assert module.normalization([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]


def test_lambda_zero_reproduces_semantic_scores() -> None:
    z = [-1.0, 0.0, 1.0]
    residual = [1.0, -1.0, 1.0]
    assert module.fused_score(z, residual, 0.0) == z


@pytest.mark.parametrize("lam", [0.05, 0.10, 0.20])
def test_bounded_residual_controls_perturbation(lam: float) -> None:
    z = [-0.75, 0.25, 1.5]
    residual = [-1.0, 0.0, 1.0]
    fused = module.fused_score(z, residual, lam)
    assert all(-1.0 <= r <= 1.0 for r in residual)
    assert all(abs(a - b) <= lam + 1e-12 for a, b in zip(fused, z, strict=True))


def test_fail_closed_official_dev_and_test_guards() -> None:
    with pytest.raises(module.Step10AccessError, match="STEP10_OFFICIAL_DEV_ACCESS_FORBIDDEN"):
        module.require_official_dev_access()
    with pytest.raises(module.Step10AccessError, match="STEP10_TEST_ACCESS_FORBIDDEN"):
        module.require_test_access()


def test_candidate_and_leakage_contracts_are_frozen() -> None:
    manifest = json.loads((ROOT / "artifacts/experiments/step10_fusion/inner_split_manifest.json").read_text())
    assert manifest["overlap_count"] == 0
    assert manifest["zero_overlap_proof"] == []
    search = json.loads((ROOT / "artifacts/experiments/step10_fusion/search_space.json").read_text())
    assert search["lambda_values"] == [0.05, 0.10, 0.20]
    assert search["trainable_configuration_count"] == 12
    assert search["b0_control"] == 0.0


def test_h3_input_contract_has_four_features() -> None:
    contract = json.loads((ROOT / "artifacts/experiments/step10_fusion/fusion_model_contract.json").read_text())
    assert len(contract["input_features"]) == 4
    assert contract["output"] == "tanh"
    assert contract["residual_bound"] == [-1, 1]
