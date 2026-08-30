from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return cast(np.ndarray, np.divide(values, np.maximum(norms, 1e-12), dtype=np.float32))


def exact_rank(query: np.ndarray, matrix: np.ndarray, codes: Sequence[str], top_k: int) -> list[tuple[str, float]]:
    query = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    scores = matrix @ query
    ranked = sorted(zip(codes, scores.tolist(), strict=True), key=lambda item: (-item[1], item[0]))
    return [(code, float(score)) for code, score in ranked[:top_k]]


def rrf_fuse(bm25_codes: Sequence[str], dense_codes: Sequence[str], rrf_k: int, top_k: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, code in enumerate(bm25_codes, 1):
        scores[code] = scores.get(code, 0.0) + 1.0 / (rrf_k + rank)
    for rank, code in enumerate(dense_codes, 1):
        scores[code] = scores.get(code, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
