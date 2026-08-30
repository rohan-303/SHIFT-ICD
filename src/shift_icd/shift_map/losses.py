from __future__ import annotations

import torch
from torch import Tensor


def _masked_logits(query: Tensor, candidates: Tensor, temperature: float) -> Tensor:
    if candidates.ndim == 2:
        candidates = candidates.unsqueeze(0)
    if query.ndim != 2 or candidates.ndim != 3 or query.shape[0] != candidates.shape[0] or query.shape[1] != candidates.shape[2]:
        raise ValueError("query must be [batch, dim] and candidates [batch, candidates, dim]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.einsum("bd,bcd->bc", query, candidates) / temperature


def l1_masked_single_infonce(
    query: Tensor, candidates: Tensor, positive_indices: list[int], valid_positive_indices: list[list[int]], temperature: float
) -> Tensor:
    logits = _masked_logits(query, candidates, temperature)
    if len(positive_indices) != query.shape[0] or len(valid_positive_indices) != query.shape[0]:
        raise ValueError("one positive specification is required per source")
    protected = torch.zeros_like(logits, dtype=torch.bool)
    for i, indices in enumerate(valid_positive_indices):
        protected[i, indices] = True
    for i, index in enumerate(positive_indices):
        if not protected[i, index]:
            raise ValueError("selected positive is outside full valid positive set")
    return -torch.log_softmax(logits, dim=1)[
        torch.arange(query.shape[0], device=query.device), torch.tensor(positive_indices, device=query.device)
    ].mean()


def l2_set_positive_infonce(query: Tensor, candidates: Tensor, valid_positive_indices: list[list[int]], temperature: float) -> Tensor:
    logits = _masked_logits(query, candidates, temperature)
    if len(valid_positive_indices) != query.shape[0]:
        raise ValueError("one positive set is required per source")
    losses = []
    for i, indices in enumerate(valid_positive_indices):
        if not indices:
            raise ValueError("set-positive loss requires at least one represented positive")
        positives = torch.tensor(indices, device=logits.device)
        numerator = torch.logsumexp(logits[i, positives], dim=0) - torch.log(torch.tensor(float(len(indices)), device=logits.device))
        losses.append(-(numerator - torch.logsumexp(logits[i], dim=0)))
    return torch.stack(losses).mean()
