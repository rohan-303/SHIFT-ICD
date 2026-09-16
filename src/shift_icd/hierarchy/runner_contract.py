from __future__ import annotations

import json
from pathlib import Path
from typing import cast


class Step9TestAccessError(RuntimeError):
    pass


def require_test_lock(lock_path: Path | None) -> dict[str, object]:
    if lock_path is None or not lock_path.is_file():
        raise Step9TestAccessError("STEP9_TEST_ACCESS_FORBIDDEN: valid TEST lock required")
    lock = cast(dict[str, object], json.loads(lock_path.read_text(encoding="utf-8")))
    if lock.get("status") != "STEP9_TEST_LOCKED_BEFORE_SCORING":
        raise Step9TestAccessError("STEP9_TEST_ACCESS_FORBIDDEN: lock status invalid")
    return lock


def checkpoint_path(configuration_id: str, objective: str, seed: int, epoch: int) -> Path:
    safe_objective = objective.lower().replace(" ", "_")
    return Path("checkpoints") / configuration_id / safe_objective / f"seed_{seed}" / f"epoch_{epoch}" / "model.pt"


def reserve_checkpoint(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"STEP9_CHECKPOINT_COLLISION:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=False)
    path.unlink()
