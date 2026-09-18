# ruff: noqa: E501, I001
from __future__ import annotations

import json
from pathlib import Path

import torch

from shift_icd.structured_decoder import StructuredAssignmentDecoder, assemble_from_outputs, build_oracle_outputs, structured_assignment_loss

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/experiments/step11_structured_decoder"


def scenario(sid: int, groups: list[list[str]]) -> dict:
    return {"scenario_id": sid, "choice_lists": [{"choice_list_id": i + 1, "alternatives": group} for i, group in enumerate(groups)]}


def main() -> None:
    torch.manual_seed(17)
    model = StructuredAssignmentDecoder(3, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    structures = [
        {"mapping_form": "NO_MAP", "scenarios": [], "flat_alternatives": [], "no_map": True},
        {"mapping_form": "SINGLE_EXACT", "scenarios": [], "flat_alternatives": ["A"], "no_map": False},
        {"mapping_form": "ALTERNATIVE", "scenarios": [], "flat_alternatives": ["A", "B"], "no_map": False},
        {"mapping_form": "COMBINATION", "scenarios": [scenario(1, [["A"], ["B"]])], "flat_alternatives": [], "no_map": False},
        {"mapping_form": "COMBINATION_WITH_ALTERNATIVES", "scenarios": [scenario(1, [["A", "B"], ["C", "D"]]), scenario(2, [["A", "C"], ["B", "D"]])], "flat_alternatives": [], "no_map": False},
    ]
    candidate_ids = ["A", "B", "C", "D"]
    features = torch.randn((len(structures), len(candidate_ids), 3))
    before = {"assignment": model.assignment_head.weight.detach().clone(), "scenario": model.scenario_queries.detach().clone(), "slot": model.slot_queries.detach().clone()}
    outputs = model(features)
    loss = structured_assignment_loss(outputs, structures, [candidate_ids] * len(structures))
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_finite = all(param.grad is not None and torch.isfinite(param.grad).all() for param in [model.assignment_head.weight, model.scenario_queries, model.slot_queries])
    optimizer.step()
    updates = {"assignment_head": not torch.equal(before["assignment"], model.assignment_head.weight), "scenario_queries": not torch.equal(before["scenario"], model.scenario_queries), "slot_queries": not torch.equal(before["slot"], model.slot_queries)}
    checkpoint = OUT / "smoke" / "step11_r1b_smoke_only_config_repaired_seed_17_epoch_1.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    reloaded = StructuredAssignmentDecoder(3, 8)
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    reloaded.eval()
    model.eval()
    with torch.no_grad():
        first = model(features)
        second = reloaded(features)
    reload_pass = all(torch.equal(first[key], second[key]) for key in first)
    oracle_cases = []
    for structure in structures:
        oracle = build_oracle_outputs(structure, candidate_ids)
        oracle_cases.append(assemble_from_outputs(oracle, candidate_ids))
    payload = {"status": "STEP11_R1B_SMOKE_ONLY", "train_subset_sources": len(structures), "loss_finite": bool(torch.isfinite(loss)), "loss": float(loss.detach()), "gradients_finite": gradient_finite, "parameter_updates": updates, "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"), "reload_pass": reload_pass, "oracle_families_covered": [s["mapping_form"] for s in structures], "full_fusion_val_scientific_evaluation_count": 0, "official_dev_scientific_evaluation_count": 0, "test_scoring_count": 0, "full_r2_configurations_trained": 0}
    (OUT / "r1b_smoke_validation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
