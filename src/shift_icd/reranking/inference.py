from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class ThermalState(StrEnum):
    OK = "OK"
    COOLDOWN = "COOLDOWN"
    HARD_STOP = "THERMAL_HARD_STOP"


class ThermalGuard:
    def __init__(
        self,
        read_temperature: Callable[[], float],
        *,
        soft_pause: float = 80.0,
        hard_stop: float = 83.0,
        resume: float = 72.0,
    ) -> None:
        self.read_temperature = read_temperature
        self.soft_pause = soft_pause
        self.hard_stop = hard_stop
        self.resume = resume
        self.state = ThermalState.OK
        self.pause_count = 0
        self.max_temperature = float("-inf")
        self.cooldown_seconds = 0.0
        self._cooldown_started: float | None = None
        self.hard_stop_triggered = False

    def check(self) -> ThermalState:
        temperature = float(self.read_temperature())
        self.max_temperature = max(self.max_temperature, temperature)
        if temperature >= self.hard_stop:
            self.state = ThermalState.HARD_STOP
            self.hard_stop_triggered = True
            return self.state
        if self.state == ThermalState.COOLDOWN:
            if temperature <= self.resume:
                if self._cooldown_started is not None:
                    self.cooldown_seconds += time.monotonic() - self._cooldown_started
                self._cooldown_started = None
                self.state = ThermalState.OK
            return self.state
        if temperature >= self.soft_pause:
            self.state = ThermalState.COOLDOWN
            self.pause_count += 1
            self._cooldown_started = time.monotonic()
        return self.state


@dataclass
class ChunkRecord:
    chunk_id: str
    start_pair: int
    end_pair: int
    sha256: str


@dataclass
class ChunkManifest:
    path: Path
    expected_pairs: int
    chunks: dict[str, ChunkRecord] = field(default_factory=dict)

    @property
    def completed_pairs(self) -> int:
        return sum(record.end_pair - record.start_pair for record in self.chunks.values())

    def add_chunk(self, chunk_id: str, start_pair: int, end_pair: int, payload: bytes) -> None:
        if chunk_id in self.chunks:
            raise ValueError(f"duplicate chunk: {chunk_id}")
        if start_pair < 0 or end_pair <= start_pair or end_pair > self.expected_pairs:
            raise ValueError("invalid chunk range")
        self.chunks[chunk_id] = ChunkRecord(chunk_id, start_pair, end_pair, hashlib.sha256(payload).hexdigest())

    def validate_chunk(self, chunk_id: str, payload: bytes) -> bool:
        record = self.chunks.get(chunk_id)
        return record is not None and record.sha256 == hashlib.sha256(payload).hexdigest()

    def missing_ranges(self) -> list[tuple[int, int]]:
        covered = sorted((record.start_pair, record.end_pair) for record in self.chunks.values())
        missing: list[tuple[int, int]] = []
        cursor = 0
        for start, end in covered:
            if start > cursor:
                missing.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < self.expected_pairs:
            missing.append((cursor, self.expected_pairs))
        return missing

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"expected_pairs": self.expected_pairs, "chunks": [record.__dict__ for record in self.chunks.values()]}
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")

    @classmethod
    def load(cls, path: Path) -> ChunkManifest:
        payload = json.loads(path.read_text(encoding="utf8"))
        manifest = cls(path, int(payload["expected_pairs"]))
        for row in payload.get("chunks", []):
            record = ChunkRecord(**row)
            if record.chunk_id in manifest.chunks:
                raise ValueError(f"duplicate chunk: {record.chunk_id}")
            manifest.chunks[record.chunk_id] = record
        return manifest


def deterministic_pair_order(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("benchmark_id", "")),
            int(str(row.get("retriever_rank", 0))),
        ),
    )


def resumable_score(
    pairs: Sequence[dict[str, object]],
    score_fn: Callable[[Sequence[dict[str, object]]], bytes],
    output_dir: Path,
    manifest: ChunkManifest,
    chunk_pairs: int,
    thermal_guard: ThermalGuard | None = None,
) -> ThermalState:
    ordered = deterministic_pair_order(pairs)
    if len(ordered) != manifest.expected_pairs:
        raise ValueError("pair count does not match manifest")
    for start in range(0, len(ordered), chunk_pairs):
        end = min(start + chunk_pairs, len(ordered))
        chunk_id = f"{start:012d}-{end:012d}"
        if chunk_id in manifest.chunks:
            continue
        if thermal_guard is not None:
            state = thermal_guard.check()
            if state == ThermalState.HARD_STOP:
                manifest.save()
                return state
            if state == ThermalState.COOLDOWN:
                manifest.save()
                return state
        payload = score_fn(ordered[start:end])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"chunk_{chunk_id}.bin").write_bytes(payload)
        manifest.add_chunk(chunk_id, start, end, payload)
        manifest.save()
    return ThermalState.HARD_STOP if thermal_guard and thermal_guard.hard_stop_triggered else ThermalState.OK
