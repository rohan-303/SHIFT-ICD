from __future__ import annotations

import json
from pathlib import Path

from scripts.step8_full_universe.r2_runner import SELECTION_FIELDS, select_best

ROOT = Path(__file__).resolve().parents[2]


def test_r2_selection_prefers_frozen_lexicographic_order_then_earliest_epoch() -> None:
    rows = [
        {"valid": True, "epoch": 2, **dict(zip(SELECTION_FIELDS, [0.8, 0.7, 0.9, 0.8, 0.5], strict=True))},
        {"valid": True, "epoch": 1, **dict(zip(SELECTION_FIELDS, [0.8, 0.7, 0.9, 0.8, 0.5], strict=True))},
        {"valid": True, "epoch": 3, **dict(zip(SELECTION_FIELDS, [0.79, 0.99, 0.99, 0.99, 0.99], strict=True))},
    ]
    assert select_best(rows)["epoch"] == 1


def test_r2_protocol_is_closed_before_outcomes() -> None:
    protocol = json.loads((ROOT / "artifacts/experiments/step8_full_universe/r2_search_protocol.json").read_text())
    assert protocol["status"] == "FROZEN_BEFORE_R2_OUTCOMES"
    assert protocol["test_access_allowed"] is False
    assert protocol["development_seed"] == 17
    assert protocol["objectives"] == ["BCE", "SET_POSITIVE_LISTWISE"]
    assert protocol["negative_strategies"] == ["TOP_RANK_HARD", "MIXED_RANK", "RANDOM_WITHIN_CANDIDATE"]
