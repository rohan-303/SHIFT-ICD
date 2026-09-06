from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import pandas as pd  # type: ignore[import-untyped]

from shift_icd.dense.text import clean_dense_text
from shift_icd.terminology import TerminologyCorpus

FORWARD = "ICD9CM_TO_ICD10CM"
BACKWARD = "ICD10CM_TO_ICD9CM"
DIRECTIONS = (FORWARD, BACKWARD)
TERMINOLOGY_VERSION = "CMS FY2018"
EXPECTED_COUNTS = {FORWARD: 17_513, BACKWARD: 11_690}


@dataclass(frozen=True)
class TargetCorpus:
    direction: str
    terminology_version: str
    codes: tuple[str, ...]
    descriptions: tuple[str, ...]
    corpus_hash: str

    def as_dict(self) -> dict[str, str]:
        return dict(zip(self.codes, self.descriptions, strict=True))


def build_target_corpus(rows: pd.DataFrame, direction: str) -> TargetCorpus:
    """Legacy GEM-observed builder; not valid for publication target membership."""
    if direction not in DIRECTIONS:
        raise ValueError(f"unsupported mapping direction: {direction}")
    required = {"direction", "target_code", "target_label", "target_short_description", "row_id"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"canonical rows missing target-corpus columns: {sorted(missing)}")

    frame = rows[(rows["direction"] == direction) & rows["target_code"].notna()].copy()
    frame = frame.sort_values(["target_code", "row_id"], kind="mergesort")
    descriptions: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        code = str(row.target_code)
        long = "" if pd.isna(row.target_label) else str(row.target_label)
        short = "" if pd.isna(row.target_short_description) else str(row.target_short_description)
        description = clean_dense_text(long or short)
        if code not in descriptions:
            descriptions[code] = description

    codes = tuple(sorted(descriptions))
    if not codes:
        raise ValueError(f"empty target corpus for {direction}")
    if direction == FORWARD and any(not re.match(r"^[A-Z]", code.upper()) for code in codes):
        raise AssertionError("forward target corpus contains a non-ICD-10-CM code")
    if direction == BACKWARD and any(not re.match(r"^(?:[0-9]|V|E)", code.upper()) for code in codes):
        raise AssertionError("backward target corpus contains a non-ICD-9-CM code")
    if len(rows) > 10_000 and len(codes) != EXPECTED_COUNTS[direction]:
        raise AssertionError(
            f"unexpected {direction} target corpus size: {len(codes)} != {EXPECTED_COUNTS[direction]}"
        )
    ordered_descriptions = tuple(descriptions[code] for code in codes)
    payload = "\n".join(f"{code}\t{description}" for code, description in zip(codes, ordered_descriptions, strict=True))
    return TargetCorpus(
        direction=direction,
        terminology_version=TERMINOLOGY_VERSION,
        codes=codes,
        descriptions=ordered_descriptions,
        corpus_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )


def build_target_corpus_from_terminology(corpus: TerminologyCorpus, direction: str) -> TargetCorpus:
    """Build target membership exclusively from an authoritative terminology corpus."""
    expected = "ICD-10-CM" if direction == FORWARD else "ICD-9-CM" if direction == BACKWARD else None
    if expected is None:
        raise ValueError(f"unsupported mapping direction: {direction}")
    if corpus.terminology != expected:
        raise ValueError(f"terminology {corpus.terminology!r} is incompatible with {direction}")
    codes = tuple(record.canonical_code for record in corpus.records)
    descriptions = tuple(clean_dense_text(record.long_description or record.short_description or "") for record in corpus.records)
    if not codes:
        raise ValueError(f"empty target corpus for {direction}")
    payload = "\n".join(f"{code}\t{description}" for code, description in zip(codes, descriptions, strict=True))
    return TargetCorpus(
        direction,
        f"{corpus.terminology} {corpus.version}",
        codes,
        descriptions,
        hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )
