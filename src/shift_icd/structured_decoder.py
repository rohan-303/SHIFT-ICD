from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch.nn as nn

from shift_icd.data.schemas import GemSourceMapping

MappingForm = Literal[
    "NO_MAP", "SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE",
    "COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO",
]

_ALLOWED_CONTEXTS = {"fusion_val_smoke", "train_subset_smoke"}


def _alternative_codes(mapping: GemSourceMapping) -> list[str]:
    return sorted({a.target_code for s in mapping.scenarios for c in s.choice_lists for a in c.alternatives})


def canonicalize_gold(mapping: GemSourceMapping) -> dict[str, Any]:
    """Return a deterministic, lossless symbolic structure; never flatten scenarios."""
    scenarios = []
    for scenario in sorted(mapping.scenarios, key=lambda item: item.scenario_id):
        choices = []
        for choice in sorted(scenario.choice_lists, key=lambda item: item.choice_list_id):
            alternatives = sorted({
                alternative.target_code for alternative in choice.alternatives
            })
            choices.append({"choice_list_id": choice.choice_list_id, "alternatives": alternatives})
        scenarios.append({"scenario_id": scenario.scenario_id, "choice_lists": choices})
    return {
        "schema_version": "step11-structured-gold-v1",
        "source_code": mapping.source_code,
        "direction": mapping.direction,
        "mapping_form": mapping.mapping_kind,
        "approximation": {
            "any": mapping.approximate_any,
            "all": mapping.approximate_all,
        },
        "flat_alternatives": [] if mapping.no_map or mapping.scenarios else _alternative_codes(mapping),
        "scenarios": scenarios,
        "no_map": mapping.no_map,
        "cardinality": {
            "scenario_count": mapping.scenario_count,
            "required_slot_count_max": mapping.required_component_count,
            "choice_list_count": mapping.choice_list_count,
            "flat_unique_target_count": mapping.unique_target_count,
            "valid_mapping_set_count": mapping.valid_mapping_set_count,
        },
    }


def serialize_structure(structure: Mapping[str, Any]) -> str:
    return json.dumps(structure, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def deserialize_structure(payload: str) -> dict[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, dict) or value.get("schema_version") != "step11-structured-gold-v1":
        raise ValueError("invalid Step 11 structured-gold payload")
    return value


def assemble_structure(structure: Mapping[str, Any], candidate_scores: Mapping[str, float]) -> dict[str, Any]:
    """Deterministically assemble valid output from permitted candidate identities."""
    form = structure["mapping_form"]
    if form == "NO_MAP":
        return {"schema_version": "step11-structured-output-v1", "mapping_form": form, "scenarios": [], "targets": []}
    if structure["scenarios"]:
        scenario_outputs = []
        for scenario in structure["scenarios"]:
            selected = []
            valid = True
            for choice in scenario["choice_lists"]:
                choice_options: list[str] = [str(code) for code in choice["alternatives"] if str(code) in candidate_scores]
                if not choice_options:
                    valid = False
                    break
                selected.append(max(choice_options, key=lambda code: (candidate_scores[code], code)))
            if valid:
                scenario_outputs.append((sum(candidate_scores[c] for c in selected), scenario["scenario_id"], selected))
        if not scenario_outputs:
            return {"schema_version": "step11-structured-output-v1", "mapping_form": "NO_MAP", "scenarios": [], "targets": []}
        _, scenario_id, selected = max(scenario_outputs, key=lambda item: (item[0], -item[1]))
        return {
            "schema_version": "step11-structured-output-v1",
            "mapping_form": form,
            "scenarios": [{"scenario_id": scenario_id, "targets": selected}],
            "targets": sorted(set(selected)),
        }
    raw_options: list[Any] = structure["flat_alternatives"]
    options: list[str] = [str(code) for code in raw_options if str(code) in candidate_scores]
    if not options:
        return {"schema_version": "step11-structured-output-v1", "mapping_form": "NO_MAP", "scenarios": [], "targets": []}
    flat_selected = max(options, key=lambda code: (candidate_scores[code], code))
    return {"schema_version": "step11-structured-output-v1", "mapping_form": form, "scenarios": [], "targets": [flat_selected]}


def source_balanced_loss(source_component_losses: Sequence[Sequence[float]]) -> float:
    if not source_component_losses:
        raise ValueError("loss requires at least one source")
    source_losses = [sum(values) / len(values) for values in source_component_losses if values]
    if len(source_losses) != len(source_component_losses):
        raise ValueError("each source requires at least one component loss")
    return sum(source_losses) / len(source_losses)


class StructuredSetDecoder(nn.Module):
    """Compact candidate-set decoder; inputs contain frozen retrieval evidence only."""

    def __init__(self, feature_dim: int, hidden_dim: int, form_count: int = 6) -> None:
        super().__init__()
        self.candidate = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.form = nn.Linear(hidden_dim, form_count)
        self.cardinality = nn.Linear(hidden_dim, 4)
        self.membership = nn.Linear(hidden_dim, 1)

    def forward(self, candidate_features: Any) -> dict[str, Any]:
        projected = self.candidate(candidate_features)
        pooled_projected = projected.mean(dim=1)
        return {
            "form_logits": self.form(pooled_projected),
            "cardinality_logits": self.cardinality(pooled_projected),
            "membership_logits": self.membership(projected).squeeze(-1),
        }


def checkpoint_path(config_id: str, seed: int, epoch: int) -> str:
    if not config_id or seed not in {17, 42, 2026} or epoch < 1:
        raise ValueError("STEP11_CHECKPOINT_COLLISION")
    return f"step11_{config_id}_seed_{seed}_epoch_{epoch}.pt"


class StructuredDecoderLock:
    def assert_allowed(self, context: str) -> None:
        if context == "official_dev":
            raise RuntimeError("STEP11_OFFICIAL_DEV_ACCESS_FORBIDDEN")
        if context == "test":
            raise RuntimeError("STEP11_TEST_ACCESS_FORBIDDEN")
        if context not in _ALLOWED_CONTEXTS:
            raise RuntimeError(f"STEP11_CONTEXT_FORBIDDEN:{context}")
