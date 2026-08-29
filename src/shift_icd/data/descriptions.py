from __future__ import annotations

from pathlib import Path

from .schemas import CodeDescription


def _read_text(path: Path) -> list[str]:
    data = path.read_bytes()
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        text = data.decode("cp1252")
    return text.splitlines()


def parse_icd9_long_titles(path: Path) -> dict[str, CodeDescription]:
    descriptions: dict[str, CodeDescription] = {}
    for line in _read_text(path):
        if not line.strip():
            continue
        code = line[:6].strip()
        description = line[6:].strip()
        if code and description:
            descriptions[code] = CodeDescription(
                code=code,
                source="CMS ICD-9-CM",
                release="Version 32",
                long_description=description,
            )
    return descriptions


def parse_icd9_short_titles(path: Path) -> dict[str, CodeDescription]:
    descriptions: dict[str, CodeDescription] = {}
    for line in _read_text(path):
        if not line.strip():
            continue
        code = line[:6].strip()
        description = line[6:].strip()
        if code and description:
            descriptions[code] = CodeDescription(
                code=code,
                source="CMS ICD-9-CM",
                release="Version 32",
                short_description=description,
            )
    return descriptions


def parse_icd10_codes(path: Path) -> dict[str, CodeDescription]:
    descriptions: dict[str, CodeDescription] = {}
    for line in _read_text(path):
        if not line.strip():
            continue
        code = line[:7].strip()
        description = line[7:].strip()
        if code and description:
            descriptions[code] = CodeDescription(
                code=code,
                source="CMS ICD-10-CM",
                release="FY 2018",
                long_description=description,
            )
    return descriptions


def parse_icd10_order_short_titles(path: Path) -> dict[str, CodeDescription]:
    descriptions: dict[str, CodeDescription] = {}
    for line in _read_text(path):
        if not line.strip() or len(line) < 16:
            continue
        code = line[6:14].strip()
        short_description = line[16:78].strip()
        if code and short_description:
            descriptions[code] = CodeDescription(
                code=code,
                source="CMS ICD-10-CM",
                release="FY 2018",
                short_description=short_description,
            )
    return descriptions


def merge_descriptions(
    long_descriptions: dict[str, CodeDescription], short_descriptions: dict[str, CodeDescription]
) -> dict[str, CodeDescription]:
    merged: dict[str, CodeDescription] = {}
    for code in set(long_descriptions) | set(short_descriptions):
        long_item = long_descriptions.get(code)
        short_item = short_descriptions.get(code)
        base = long_item or short_item
        if base is None:
            continue
        merged[code] = CodeDescription(
            code=code,
            source=base.source,
            release=base.release,
            long_description=long_item.long_description if long_item else None,
            short_description=short_item.short_description if short_item else None,
        )
    return merged
