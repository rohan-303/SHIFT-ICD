from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def build_cross_encoder_text(source_description: str, target_description: str) -> tuple[str, str]:
    """Return only the two authoritative descriptions used by the CE tokenizer."""
    return source_description, target_description


def preserve_candidate_membership(original: Sequence[Sequence[str]], reranked: Sequence[Sequence[str]]) -> bool:
    if len(original) != len(reranked):
        raise ValueError("reranking changed source count")
    for before, after in zip(original, reranked, strict=True):
        if set(before) != set(after) or len(before) != len(after):
            raise ValueError("reranking changed candidate membership")
    return True


def candidate_coverage(
    rows: Sequence[Any],
    gold: set[str] | None = None,
    k: int = 100,
    *,
    row_format: bool = False,
) -> float:
    if row_format:
        ordinary = [r for r in rows if r.get("kind") in {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"}]
        return float(np.mean([bool(set(r["gold"]) & set(r["candidates"][:k])) for r in ordinary])) if ordinary else 0.0
    if gold is None or not rows:
        raise ValueError("gold and candidate rows are required")
    return float(bool(set(rows[0][:k]) & gold))


def listwise_set_positive_loss(scores: np.ndarray, positive_indices: Iterable[int]) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    positives = np.asarray(list(positive_indices), dtype=np.int64)
    if scores.ndim != 1 or len(positives) == 0 or np.any(positives < 0) or np.any(positives >= len(scores)):
        raise ValueError("invalid score or positive-index shape")
    denominator = np.logaddexp.reduce(scores)
    numerator = np.logaddexp.reduce(scores[positives])
    return float(denominator - numerator)


def source_balanced_bce(logits: Sequence[Sequence[float]], labels: Sequence[Sequence[float]]) -> float:
    if len(logits) != len(labels) or not logits:
        raise ValueError("source/logit population mismatch")
    source_losses: list[float] = []
    for source_logits, source_labels in zip(logits, labels, strict=True):
        x = np.asarray(source_logits, dtype=np.float64)
        y = np.asarray(source_labels, dtype=np.float64)
        if x.shape != y.shape or x.size == 0:
            raise ValueError("source candidate shape mismatch")
        source_losses.append(float(np.mean(np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x))))) )
    return float(np.mean(source_losses))


def sample_negative_ranks(
    ranks: Sequence[int], positive_ranks: set[int], *, strategy: str, seed: int, count: int
) -> list[int]:
    available = [r for r in ranks if r not in positive_ranks]
    if count > len(available):
        raise ValueError("not enough non-positive candidates")
    rng = np.random.default_rng(seed)
    if strategy == "hard":
        return available[:count]
    if strategy == "random":
        return sorted(rng.choice(available, size=count, replace=False).tolist())
    if strategy != "mixed":
        raise ValueError(f"unknown negative strategy: {strategy}")
    bands = [
        [r for r in available if 1 <= r <= 10],
        [r for r in available if 11 <= r <= 25],
        [r for r in available if 26 <= r <= 100],
    ]
    quotas = [min(3, len(bands[0])), min(2, len(bands[1])), min(2, len(bands[2]))]
    chosen: list[int] = []
    for band, quota in zip(bands, quotas, strict=True):
        if quota:
            chosen.extend(rng.choice(band, size=quota, replace=False).tolist())
    remainder = [r for r in available if r not in chosen]
    if len(chosen) < count:
        chosen.extend(rng.choice(remainder, size=count - len(chosen), replace=False).tolist())
    return sorted(chosen[:count])


def ndcg_at_k(relevant: set[str], ranked_codes: Sequence[str], k: int) -> float:
    gains = [1.0 if code in relevant else 0.0 for code in ranked_codes[:k]]
    dcg = sum(gain / np.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sum(1.0 / np.log2(index + 2) for index in range(min(len(relevant), k)))
    return float(dcg / ideal) if ideal else 0.0
