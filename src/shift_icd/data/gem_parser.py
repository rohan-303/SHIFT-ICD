from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .schemas import Direction, GemRawRow

DIRECTION_LAYOUT: dict[Direction, tuple[int, int]] = {
    "ICD9CM_TO_ICD10CM": (6, 8),
    "ICD10CM_TO_ICD9CM": (8, 6),
}


def parse_gem_record(raw_record: str, direction: Direction, row_id: str) -> GemRawRow:
    line = raw_record.rstrip("\r\n")
    source_width, target_width = DIRECTION_LAYOUT[direction]
    if len(line) != 19:
        raise ValueError(f"GEM record must be 19 characters, got {len(line)}")
    source = line[:source_width]
    target = line[source_width : source_width + target_width]
    flags = line[source_width + target_width :]
    return GemRawRow(
        row_id=row_id,
        direction=direction,
        source_code_raw=source,
        target_code_raw=target,
        approximate=flags[0] == "1",
        no_map=flags[1] == "1",
        combination=flags[2] == "1",
        scenario=int(flags[3]),
        choice_list=int(flags[4]),
        raw_flag_string=flags,
        raw_record=line,
    )


def parse_gem_lines(lines: Iterable[str], direction: Direction) -> list[GemRawRow]:
    return [parse_gem_record(line, direction, f"{direction}:{index:06d}") for index, line in enumerate(lines, 1) if line.rstrip("\r\n")]


def parse_gem_member(path: Path, direction: Direction) -> list[GemRawRow]:
    return parse_gem_lines(path.read_text(encoding="ascii").splitlines(), direction)
