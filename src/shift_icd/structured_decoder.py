# ruff: noqa: E501, B905
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from itertools import permutations
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

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


# R1B repaired structural capacities, recovered from the frozen R1 gold audit.
MAX_SCENARIOS = 6
MAX_SLOTS = 3
_FORM_INDEX = {name: index for index, name in enumerate(("NO_MAP", "SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE", "COMBINATION", "COMBINATION_WITH_ALTERNATIVES"))}
_INDEX_FORM = {value: key for key, value in _FORM_INDEX.items()}


class StructuredAssignmentDecoder(nn.Module):
    """R1B candidate-set decoder with explicit scenario/slot assignment outputs."""

    def __init__(self, feature_dim: int, hidden_dim: int, form_count: int = 6) -> None:
        super().__init__()
        self.candidate = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.scenario_queries = nn.Parameter(torch.randn(MAX_SCENARIOS, hidden_dim) * 0.02)
        self.slot_queries = nn.Parameter(torch.randn(MAX_SCENARIOS, MAX_SLOTS, hidden_dim) * 0.02)
        self.form_head = nn.Linear(hidden_dim, form_count)
        self.scenario_count_head = nn.Linear(hidden_dim, MAX_SCENARIOS + 1)
        self.scenario_activity_head = nn.Linear(hidden_dim, 1)
        self.slot_count_head = nn.Linear(hidden_dim, MAX_SLOTS + 1)
        self.slot_activity_head = nn.Linear(hidden_dim, 1)
        self.assignment_head = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.slot_assignment_head = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.membership_head = nn.Linear(hidden_dim, 1)

    def forward(self, candidate_features: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = self.candidate(candidate_features)
        pooled = projected.mean(dim=1)
        scenario_state = pooled.unsqueeze(1) + self.scenario_queries.unsqueeze(0)
        slot_state = scenario_state.unsqueeze(2) + self.slot_queries.unsqueeze(0)
        candidate_assignment = self.assignment_head(projected)
        slot_assignment = self.slot_assignment_head(slot_state)
        assignment = torch.einsum("bkh,bslh->bksl", candidate_assignment, slot_assignment) / projected.shape[-1] ** 0.5
        return {
            "form_logits": self.form_head(pooled),
            "scenario_count_logits": self.scenario_count_head(pooled),
            "scenario_activity_logits": self.scenario_activity_head(scenario_state).squeeze(-1),
            "slot_count_logits": self.slot_count_head(scenario_state),
            "slot_activity_logits": self.slot_activity_head(slot_state).squeeze(-1),
            "assignment_logits": assignment,
            "membership_logits": self.membership_head(projected).squeeze(-1),
        }


def _normal_form(structure: Mapping[str, Any]) -> str:
    return str(structure.get("mapping_form", "NO_MAP"))


def _sorted_scenarios(structure: Mapping[str, Any]) -> list[dict[str, Any]]:
    scenarios = structure.get("scenarios", [])
    return sorted(scenarios, key=lambda item: tuple(tuple(sorted(map(str, choice.get("alternatives", [])))) for choice in item.get("choice_lists", [])))


def _sorted_choice_lists(scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(scenario.get("choice_lists", []), key=lambda item: tuple(sorted(map(str, item.get("alternatives", [])))))


def _canonical_decoded(form: str, scenarios: list[list[list[str]]], flat_targets: list[str]) -> dict[str, Any]:
    canonical_scenarios = []
    for scenario_index, choices in enumerate(sorted(scenarios, key=lambda group: tuple(tuple(sorted(choice)) for choice in group)), start=1):
        canonical_scenarios.append({"scenario_id": scenario_index, "choice_lists": [{"choice_list_id": choice_index, "alternatives": sorted(set(choice))} for choice_index, choice in enumerate(sorted(choices, key=lambda choice: tuple(sorted(choice))), start=1)]})
    return {"mapping_form": form, "scenarios": canonical_scenarios, "flat_alternatives": sorted(set(flat_targets)) if not canonical_scenarios else [], "no_map": form == "NO_MAP"}


def _top_indices(scores: torch.Tensor, count: int) -> list[int]:
    return sorted(range(scores.numel()), key=lambda index: (-float(scores[index]), index))[:count]


def assemble_from_outputs(outputs: Mapping[str, torch.Tensor], candidate_ids: Sequence[str]) -> dict[str, Any]:
    """Decode only model tensors and candidate identities; no gold structure is accepted."""
    form = _INDEX_FORM[int(torch.argmax(outputs["form_logits"]).item())]
    if form == "NO_MAP":
        return _canonical_decoded(form, [], [])
    membership = outputs["membership_logits"].reshape(-1)
    if form in {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"}:
        selected = [index for index, value in enumerate(torch.sigmoid(membership)) if float(value) >= 0.5]
        if form in {"SINGLE_EXACT", "SINGLE_APPROXIMATE"} or not selected:
            selected = _top_indices(membership, 1)
        return _canonical_decoded(form, [], [str(candidate_ids[index]) for index in selected])
    scenario_count = int(torch.argmax(outputs["scenario_count_logits"]).item())
    scenario_count = min(MAX_SCENARIOS, max(1, scenario_count))
    scenario_scores = outputs["scenario_activity_logits"].reshape(-1)
    scenario_indices = _top_indices(scenario_scores, scenario_count)
    scenarios: list[list[list[str]]] = []
    for scenario_index in scenario_indices:
        slot_count = int(torch.argmax(outputs["slot_count_logits"][scenario_index]).item())
        slot_count = min(MAX_SLOTS, max(1, slot_count))
        slot_indices = _top_indices(outputs["slot_activity_logits"][scenario_index], slot_count)
        choices: list[list[str]] = []
        for slot_index in slot_indices:
            assignment = outputs["assignment_logits"][:, scenario_index, slot_index].reshape(-1)
            selected = [idx for idx, value in enumerate(torch.sigmoid(assignment)) if float(value) >= 0.5]
            if not selected:
                selected = _top_indices(assignment, 1)
            if form == "COMBINATION":
                selected = _top_indices(assignment, 1)
            choices.append([str(candidate_ids[index]) for index in selected])
        scenarios.append(choices)
    return _canonical_decoded(form, scenarios, [])


def build_oracle_outputs(structure: Mapping[str, Any], candidate_ids: Sequence[str]) -> dict[str, torch.Tensor]:
    """Construct perfect tensors using only the repaired model-output schema."""
    form = _normal_form(structure)
    outputs: dict[str, torch.Tensor] = {
        "form_logits": torch.full((6,), -20.0),
        "scenario_count_logits": torch.full((MAX_SCENARIOS + 1,), -20.0),
        "scenario_activity_logits": torch.full((MAX_SCENARIOS,), -20.0),
        "slot_count_logits": torch.full((MAX_SCENARIOS, MAX_SLOTS + 1), -20.0),
        "slot_activity_logits": torch.full((MAX_SCENARIOS, MAX_SLOTS), -20.0),
        "assignment_logits": torch.full((len(candidate_ids), MAX_SCENARIOS, MAX_SLOTS), -20.0),
        "membership_logits": torch.full((len(candidate_ids),), -20.0),
    }
    outputs["form_logits"][_FORM_INDEX[form]] = 20.0
    scenarios = _sorted_scenarios(structure)
    if form in {"NO_MAP", "SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"}:
        outputs["scenario_count_logits"][0] = 20.0
        targets = structure.get("flat_alternatives", [])
        for index, candidate in enumerate(candidate_ids):
            if candidate in targets:
                outputs["membership_logits"][index] = 20.0
        return outputs
    outputs["scenario_count_logits"][len(scenarios)] = 20.0
    candidate_index = {str(code): index for index, code in enumerate(candidate_ids)}
    for scenario_index, scenario in enumerate(scenarios):
        outputs["scenario_activity_logits"][scenario_index] = 20.0
        choices = _sorted_choice_lists(scenario)
        outputs["slot_count_logits"][scenario_index, len(choices)] = 20.0
        for slot_index, choice in enumerate(choices):
            outputs["slot_activity_logits"][scenario_index, slot_index] = 20.0
            for target in choice.get("alternatives", []):
                if str(target) in candidate_index:
                    outputs["assignment_logits"][candidate_index[str(target)], scenario_index, slot_index] = 20.0
    return outputs


def match_structures(structure: Mapping[str, Any], candidate_ids: Sequence[str]) -> tuple[tuple[tuple[str, ...], ...], ...]:
    """Return a deterministic exchangeability-invariant structural signature."""
    del candidate_ids
    scenarios = []
    for scenario in structure.get("scenarios", []):
        choices = tuple(sorted(tuple(sorted(map(str, choice.get("alternatives", [])))) for choice in scenario.get("choice_lists", [])))
        scenarios.append(choices)
    return tuple(sorted(scenarios))


def hierarchical_match(structure: Mapping[str, Any], scenario_costs: torch.Tensor | None = None, slot_costs: torch.Tensor | None = None) -> tuple[tuple[int, int, tuple[tuple[int, int], ...]], ...]:
    """Deterministic exhaustive scenario/slot matching with query-index tie breaks."""
    scenarios = _sorted_scenarios(structure)
    scenario_count = len(scenarios)
    if scenario_count == 0:
        return ()
    scenario_costs = torch.zeros((MAX_SCENARIOS, scenario_count)) if scenario_costs is None else scenario_costs.detach().cpu()
    slot_costs = torch.zeros((MAX_SCENARIOS, MAX_SLOTS, scenario_count, MAX_SLOTS)) if slot_costs is None else slot_costs.detach().cpu()
    best: tuple[float, tuple[int, ...], tuple[tuple[tuple[int, int], ...], ...]] | None = None
    for query_order in permutations(range(MAX_SCENARIOS), scenario_count):
        slot_matches: list[tuple[tuple[int, int], ...]] = []
        total = sum(float(scenario_costs[query_index, gold_index]) for gold_index, query_index in enumerate(query_order))
        valid = True
        for gold_index, query_index in enumerate(query_order):
            gold_slots = _sorted_choice_lists(scenarios[gold_index])
            slot_count = len(gold_slots)
            if slot_count > MAX_SLOTS:
                valid = False
                break
            slot_best = min(
                ((tuple(slot_order), sum(float(slot_costs[query_index, query_slot, gold_index, gold_slot]) for gold_slot, query_slot in enumerate(slot_order))) for slot_order in permutations(range(MAX_SLOTS), slot_count)),
                key=lambda item: (item[1], item[0]),
            )
            total += slot_best[1]
            slot_matches.append(tuple((query_slot, gold_slot) for gold_slot, query_slot in enumerate(slot_best[0])))
        candidate = (total, query_order, tuple(slot_matches))
        if valid and (best is None or candidate[:2] < best[:2]):
            best = candidate
    if best is None:
        raise ValueError("structured gold exceeds frozen query capacities")
    return tuple((query_index, gold_index, best[2][gold_index]) for gold_index, query_index in enumerate(best[1]))


def structured_assignment_loss(outputs: Mapping[str, torch.Tensor], structures: Sequence[Mapping[str, Any]], candidate_batches: Sequence[Sequence[str]]) -> torch.Tensor:
    """Source-balanced repaired loss with masked missing-slot assignment targets."""
    losses: list[torch.Tensor] = []
    for batch_index, (structure, candidates) in enumerate(zip(structures, candidate_batches)):
        form = _FORM_INDEX[_normal_form(structure)]
        source_terms = [F.cross_entropy(outputs["form_logits"][batch_index:batch_index + 1], torch.tensor([form], device=outputs["form_logits"].device))]
        scenarios = _sorted_scenarios(structure)
        scenario_count = 0 if form < _FORM_INDEX["COMBINATION"] else len(scenarios)
        source_terms.append(F.cross_entropy(outputs["scenario_count_logits"][batch_index:batch_index + 1], torch.tensor([scenario_count], device=outputs["form_logits"].device)))
        if form < _FORM_INDEX["COMBINATION"]:
            target = torch.tensor([1.0 if str(candidate) in structure.get("flat_alternatives", []) else 0.0 for candidate in candidates], device=outputs["form_logits"].device)
            source_terms.append(F.binary_cross_entropy_with_logits(outputs["membership_logits"][batch_index], target))
        else:
            scenario_activity = torch.zeros(MAX_SCENARIOS, device=outputs["form_logits"].device)
            for scenario_index, scenario in enumerate(scenarios):
                scenario_activity[scenario_index] = 1.0
                choices = _sorted_choice_lists(scenario)
                source_terms.append(F.cross_entropy(outputs["slot_count_logits"][batch_index, scenario_index:scenario_index + 1], torch.tensor([len(choices)], device=outputs["form_logits"].device)))
                slot_activity = torch.zeros(MAX_SLOTS, device=outputs["form_logits"].device)
                slot_activity[:len(choices)] = 1.0
                source_terms.append(F.binary_cross_entropy_with_logits(outputs["slot_activity_logits"][batch_index, scenario_index], slot_activity))
                for slot_index, choice in enumerate(choices):
                    valid = [str(target) for target in choice.get("alternatives", []) if str(target) in candidates]
                    if not valid:
                        continue
                    assignment_target = torch.tensor([1.0 if str(candidate) in valid else 0.0 for candidate in candidates], device=outputs["form_logits"].device)
                    source_terms.append(F.binary_cross_entropy_with_logits(outputs["assignment_logits"][batch_index, :, scenario_index, slot_index], assignment_target))
            source_terms.append(F.binary_cross_entropy_with_logits(outputs["scenario_activity_logits"][batch_index], scenario_activity))
        losses.append(torch.stack(source_terms).mean())
    return torch.stack(losses).mean()
