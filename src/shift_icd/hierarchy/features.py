from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .metadata import HierarchyRecord

FEATURE_NAMES: tuple[str, ...] = (
    "source_depth_norm",
    "target_depth_norm",
    "target_sibling_count_log1p",
    "target_ancestor_count_norm",
    "candidate_same_parent_fraction",
    "candidate_same_family_fraction",
    "candidate_shared_ancestor_fraction",
    "candidate_same_root_fraction",
)


def _norm(value: int, maximum: int) -> float:
    return float(value) / float(maximum) if maximum else 0.0


def _shared_ancestor(a: HierarchyRecord, b: HierarchyRecord) -> bool:
    return bool(set(a.ancestor_chain) & set(b.ancestor_chain)) or a.code == b.code


def extract_feature_vector(
    source_code: str,
    target_code: str,
    candidate_codes: Sequence[str],
    source_nodes: dict[str, HierarchyRecord],
    target_nodes: dict[str, HierarchyRecord],
) -> tuple[float, ...]:
    """Extract ontology/candidate-set features without labels or mapping metadata."""
    source = source_nodes[source_code]
    target = target_nodes[target_code]
    candidates = [target_nodes[code] for code in candidate_codes]
    max_source_depth = max((node.depth for node in source_nodes.values()), default=0)
    max_target_depth = max((node.depth for node in target_nodes.values()), default=0)
    sibling_count = sum(int(node.parent_code == target.parent_code) for node in target_nodes.values()) - 1
    denominator = max(1, len(candidates))
    same_parent = sum(int(node.parent_code == target.parent_code) for node in candidates) / denominator
    same_family = sum(int(node.family == target.family) for node in candidates) / denominator
    shared_ancestor = sum(int(_shared_ancestor(node, target)) for node in candidates) / denominator
    same_root = sum(int(node.root_code == target.root_code) for node in candidates) / denominator
    return (
        _norm(source.depth, max_source_depth),
        _norm(target.depth, max_target_depth),
        math.log1p(max(0, sibling_count)),
        _norm(len(target.ancestor_chain), max_target_depth),
        same_parent,
        same_family,
        shared_ancestor,
        same_root,
    )


@dataclass(frozen=True, slots=True)
class TrainScaler:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES


def fit_train_scaler(rows: Iterable[Sequence[float]], *, role: str) -> TrainScaler:
    if role != "TRAIN":
        raise ValueError("TRAIN_ONLY_NORMALIZATION_REQUIRED")
    values = [tuple(float(x) for x in row) for row in rows]
    if not values or any(len(row) != len(FEATURE_NAMES) for row in values):
        raise ValueError("INVALID_FEATURE_MATRIX")
    means = tuple(sum(row[i] for row in values) / len(values) for i in range(len(FEATURE_NAMES)))
    scales = tuple(max((sum((row[i] - means[i]) ** 2 for row in values) / len(values)) ** 0.5, 1.0) for i in range(len(FEATURE_NAMES)))
    return TrainScaler(means, scales)


def apply_train_scaler(rows: Iterable[Sequence[float]], scaler: TrainScaler) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple((float(x) - mean) / scale for x, mean, scale in zip(row, scaler.means, scaler.scales, strict=True)) for row in rows)
