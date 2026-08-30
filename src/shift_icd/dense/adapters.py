from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

Pooling = Literal["cls", "mean", "native_sentence_transformer", "qwen_last_token"]


@dataclass(frozen=True)
class DenseEncoderSpec:
    model_id: str
    revision: str
    pooling: Pooling
    max_length: int
    embedding_dim: int


def cls_pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
    return last_hidden_state[:, 0, :]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def qwen_last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[batch_indices, sequence_lengths]


def numpy_embeddings(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()
