from __future__ import annotations

import random
from pathlib import Path
from typing import Any, cast

import torch

from shift_icd.reranking.step8 import source_balanced_listwise_loss


def make_contract_v2_list(group: list[dict[str, Any]], *, seed: int, epoch: int, minimum_list_size: int = 8) -> list[dict[str, Any]]:
    positives = [row for row in group if bool(row["_gold"])]
    negatives = [row for row in group if not bool(row["_gold"])]
    if not positives or not negatives:
        raise ValueError("SOURCE_MUST_HAVE_POSITIVE_AND_TRUE_NEGATIVE")
    rng = random.Random(f"{seed}:{epoch}:{group[0]['source_id']}")
    rng.shuffle(negatives)
    selected = positives + negatives[: max(1, minimum_list_size - len(positives))]
    return sorted(selected, key=lambda row: int(row["candidate_rank"]))


def validate_variable_lists(groups: list[list[dict[str, Any]]]) -> dict[str, int]:
    result = {"source_count": len(groups), "positive_count": 0, "dropped_positive_count": 0, "collision_count": 0}
    for group in groups:
        positives = [row for row in group if bool(row["_gold"])]
        result["positive_count"] += len(positives)
        result["collision_count"] += int(any(bool(row["_gold"]) and not bool(row["_gold"]) for row in group))
        selected = make_contract_v2_list(group, seed=17, epoch=1)
        result["dropped_positive_count"] += len(set(id(row) for row in positives) - set(id(row) for row in selected))
    return result


class HierarchyMLP(torch.nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.net(features)
        return torch.as_tensor(output).squeeze(-1)


def listwise_loss(logits: torch.Tensor, lists: list[list[int]], positive_indices: list[list[int]]) -> torch.Tensor:
    chunks = [logits[start : start + len(indices)] for start, indices in _offsets(lists)]
    return cast(torch.Tensor, source_balanced_listwise_loss(chunks, positive_indices))


def _offsets(lists: list[list[int]]) -> list[tuple[int, list[int]]]:
    output: list[tuple[int, list[int]]] = []
    cursor = 0
    for group in lists:
        output.append((cursor, group))
        cursor += len(group)
    return output


def checkpoint_save_reload(model: HierarchyMLP, path: Path) -> HierarchyMLP:
    path.parent.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), path)
    restored = HierarchyMLP()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    restored.eval()
    return restored
