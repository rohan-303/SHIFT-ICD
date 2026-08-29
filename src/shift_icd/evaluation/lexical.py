from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def lexical_features(source: str, target: str) -> dict[str, float | int | bool]:
    source_tokens = tokens(source)
    target_tokens = tokens(target)
    union = source_tokens | target_tokens
    intersection = source_tokens & target_tokens
    return {
        "exact_string_match": source.strip().casefold() == target.strip().casefold(),
        "shared_token_count": len(intersection),
        "token_overlap": len(intersection) / max(len(source_tokens), len(target_tokens), 1),
        "jaccard_similarity": len(intersection) / max(len(union), 1),
        "character_similarity": SequenceMatcher(None, source.casefold(), target.casefold()).ratio(),
        "description_length_difference": abs(len(source) - len(target)),
    }


def lexical_score(source: str, target: str) -> float:
    """Fast, deterministic lexical audit score used during corpus construction."""
    source_tokens = tokens(source)
    target_tokens = tokens(target)
    return len(source_tokens & target_tokens) / max(len(source_tokens | target_tokens), 1)


def score_distribution_bucket(score: float) -> str:
    if score >= 0.90:
        return "LEXICAL_EXACT"
    if score >= 0.60:
        return "LEXICAL_HIGH"
    if score >= 0.30:
        return "LEXICAL_MEDIUM"
    return "LEXICAL_LOW"


def token_counter(texts: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokens(text))
    return counter
