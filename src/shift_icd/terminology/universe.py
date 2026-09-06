from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from shift_icd.data.descriptions import (
    parse_icd9_long_titles,
    parse_icd9_short_titles,
    parse_icd10_codes,
)

_CODE_DOT_RE = re.compile(r"\.")
_SPACE_RE = re.compile(r"\s+")
PARSER_VERSION = "terminology-universe-builder-v2"
NORMALIZATION_VERSION = "uppercase-trim-dotless-whitespace-v1"


@dataclass(frozen=True, slots=True)
class TerminologyRecord:
    canonical_code: str
    display_code: str
    long_description: str | None
    short_description: str | None
    terminology: str
    version: str
    source_member: str


@dataclass(frozen=True, slots=True)
class TerminologyCorpus:
    terminology: str
    version: str
    records: tuple[TerminologyRecord, ...]
    provenance: dict[str, str]


def canonicalize_code(value: str) -> str:
    """Normalize terminology identity without changing benchmark identifiers."""
    return _SPACE_RE.sub("", _CODE_DOT_RE.sub("", value.strip().upper()))


def _normalize_records(
    descriptions: dict[str, Any], terminology: str, version: str, member: str
) -> tuple[TerminologyRecord, ...]:
    records: dict[str, TerminologyRecord] = {}
    for display, item in descriptions.items():
        code = canonicalize_code(display)
        long_description = getattr(item, "long_description", None)
        short_description = getattr(item, "short_description", None)
        if code and code not in records:
            records[code] = TerminologyRecord(
                canonical_code=code,
                display_code=display,
                long_description=long_description,
                short_description=short_description,
                terminology=terminology,
                version=version,
                source_member=member,
            )
    return tuple(records[code] for code in sorted(records))


def build_icd10_universe(archive: Path) -> TerminologyCorpus:
    """Build ICD-10-CM membership from the complete authoritative code file only."""
    with ZipFile(archive) as bundle:
        data = bundle.read("icd10cm_codes_2018.txt")
    path = archive.parent / ".terminology_icd10.tmp"
    try:
        path.write_bytes(data)
        records = _normalize_records(
            parse_icd10_codes(path), "ICD-10-CM", "FY 2018", "icd10cm_codes_2018.txt"
        )
    finally:
        path.unlink(missing_ok=True)
    return TerminologyCorpus(
        "ICD-10-CM", "FY 2018", records,
        {
            "provider": "CMS",
            "resource": "2018 ICD-10-CM diagnosis code descriptions",
            "member": "icd10cm_codes_2018.txt",
            "archive": archive.name,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        },
    )


def build_icd9_universe(archive: Path) -> TerminologyCorpus:
    """Build ICD-9-CM diagnosis membership from authoritative diagnosis title files only."""
    with ZipFile(archive) as bundle:
        long_data = bundle.read("CMS32_DESC_LONG_DX.txt")
        short_data = bundle.read("CMS32_DESC_SHORT_DX.txt")
    long_path = archive.parent / ".terminology_icd9_long.tmp"
    short_path = archive.parent / ".terminology_icd9_short.tmp"
    try:
        long_path.write_bytes(long_data)
        short_path.write_bytes(short_data)
        long_titles = parse_icd9_long_titles(long_path)
        short_titles = parse_icd9_short_titles(short_path)
        combined = dict(long_titles)
        for code, item in short_titles.items():
            if code in combined:
                combined[code] = combined[code].model_copy(update={"short_description": item.short_description})
            else:
                combined[code] = item
        records = _normalize_records(combined, "ICD-9-CM", "Version 32", "CMS32_DESC_LONG_DX.txt/CMS32_DESC_SHORT_DX.txt")
    finally:
        long_path.unlink(missing_ok=True)
        short_path.unlink(missing_ok=True)
    return TerminologyCorpus(
        "ICD-9-CM", "Version 32", records,
        {
            "provider": "CMS",
            "resource": "Version 32 ICD-9-CM diagnosis titles",
            "member": "CMS32_DESC_LONG_DX.txt/CMS32_DESC_SHORT_DX.txt",
            "archive": archive.name,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        },
    )


def corpus_hashes(corpus: TerminologyCorpus) -> dict[str, str | int]:
    codes = [record.canonical_code for record in corpus.records]
    descriptions = [
        f"{record.canonical_code}\t{record.long_description or record.short_description or ''}"
        for record in corpus.records
    ]
    return {
        "count": len(codes),
        "code_hash": hashlib.sha256("\n".join(codes).encode()).hexdigest(),
        "code_description_hash": hashlib.sha256("\n".join(descriptions).encode()).hexdigest(),
    }


def write_corpus(corpus: TerminologyCorpus, output: Path) -> dict[str, Any]:
    """Persist a versioned terminology corpus and its deterministic manifest."""
    output.mkdir(parents=True, exist_ok=True)
    corpus_path = output / f"{corpus.terminology.lower().replace('-', '')}_diagnosis.jsonl"
    with corpus_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in corpus.records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    hashes = corpus_hashes(corpus)
    manifest = {
        "schema": "terminology-universe-v2",
        "terminology": corpus.terminology,
        "version": corpus.version,
        "membership_policy": "authoritative_description_resources_only; no GEM dependency",
        "gem_dependency": "NONE",
        "parser_version": PARSER_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "builder_version": PARSER_VERSION,
        "provenance": corpus.provenance,
        **hashes,
        "corpus_file": corpus_path.name,
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
    }
    (output / f"{corpus.terminology.lower().replace('-', '')}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
