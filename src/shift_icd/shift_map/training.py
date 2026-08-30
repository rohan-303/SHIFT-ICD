from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Literal

Policy = Literal["P0", "P1", "P2"]


@dataclass(frozen=True)
class TrainingExample:
    benchmark_id: str
    source_code: str
    source_label: str
    source_family: str
    direction: str
    split: str
    source_family_split: str
    mapping_kind: str
    valid_target_codes: tuple[str, ...]
    lexical_difficulty: str


def eligible_examples(rows: list[TrainingExample], policy: Policy) -> list[TrainingExample]:
    allowed = {
        "P0": {"SINGLE_EXACT"},
        "P1": {"SINGLE_EXACT", "SINGLE_APPROXIMATE"},
        "P2": {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"},
    }[policy]
    return [row for row in rows if row.mapping_kind in allowed and row.valid_target_codes]


def sample_positive(row: TrainingExample, seed: int, epoch: int) -> str:
    if not row.valid_target_codes:
        raise ValueError("cannot sample a positive from an empty set")
    digest = hashlib.sha256(f"{seed}:{epoch}:{row.benchmark_id}".encode()).digest()
    return row.valid_target_codes[int.from_bytes(digest[:8], "big") % len(row.valid_target_codes)]


def source_balanced_epoch(rows: list[TrainingExample], seed: int, epoch: int) -> list[TrainingExample]:
    result = list(rows)
    random.Random(f"{seed}:{epoch}").shuffle(result)
    return result
