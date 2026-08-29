from __future__ import annotations

from typing import Any

from pydantic import Field

from shift_icd.data.schemas import GemSourceMapping


class BenchmarkExample(GemSourceMapping):
    benchmark_id: str
    benchmark_version: str = "1.0"
    source_family: str
    valid_target_codes: list[str] = Field(default_factory=list)
    split: str | None = None
    source_family_split: str | None = None
    difficulty_slices: list[str] = Field(default_factory=list)
    lexical_metadata: dict[str, Any] = Field(default_factory=dict)
    audit_metadata: dict[str, Any] = Field(default_factory=dict)
