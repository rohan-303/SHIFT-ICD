from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

TOKENIZER_VERSION = "regex-alnum-v1"
NORMALIZATION_VERSION = "nfkc-lower-punct-boundary-v1"
BM25_VERSION = "internal-robertson-v1"
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(_TOKEN_RE.findall(text))


def tokenize(text: str | None) -> list[str]:
    return normalize_text(text).split()


def token_jaccard(query: str, document: str) -> float:
    q, d = set(tokenize(query)), set(tokenize(document))
    return len(q & d) / len(q | d) if q | d else 0.0


def exact_label_rank(query: str, documents: Mapping[str, str], exact_index: Mapping[str, list[str]] | None = None) -> list[str]:
    q = normalize_text(query)
    if exact_index is not None:
        exact = sorted(exact_index.get(q, []))
    else:
        exact = sorted(code for code, text in documents.items() if normalize_text(text) == q)
    return exact + sorted(set(documents) - set(exact))


def token_overlap_rank(query: str, documents: Mapping[str, str], token_index: Mapping[str, set[str]] | None = None) -> list[str]:
    q = set(tokenize(query))
    if token_index is not None:
        candidates = {code for term in q for code in token_index.get(term, set())}
    else:
        candidates = {code for code, text in documents.items() if q.intersection(tokenize(text))}
    ranked = sorted(candidates, key=lambda code: (-token_jaccard(query, documents[code]), code))
    return ranked + sorted(set(documents) - candidates)


@dataclass(frozen=True)
class BM25Index:
    documents: dict[str, str]
    tokens: dict[str, tuple[str, ...]]
    postings: dict[str, dict[str, int]]
    document_frequency: dict[str, int]
    average_document_length: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def from_documents(cls, documents: Mapping[str, str], k1: float = 1.5, b: float = 0.75) -> BM25Index:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 >= 0 and 0 <= b <= 1")
        normalized = {str(code): normalize_text(text) for code, text in documents.items()}
        tokens = {code: tuple(tokenize(text)) for code, text in normalized.items()}
        postings: dict[str, dict[str, int]] = defaultdict(dict)
        for code in sorted(tokens):
            for term, frequency in Counter(tokens[code]).items():
                postings[term][code] = frequency
        lengths = [len(value) for value in tokens.values()]
        average = sum(lengths) / len(lengths) if lengths else 0.0
        return cls(normalized, tokens, dict(postings), {term: len(posting) for term, posting in postings.items()}, average, k1, b)

    def _idf(self, document_frequency: int) -> float:
        n = len(self.tokens)
        return math.log1p((n - document_frequency + 0.5) / (document_frequency + 0.5))

    def score(self, query: str, code: str) -> float:
        query_terms = set(tokenize(query))
        length = len(self.tokens[code])
        norm = self.k1 * (1 - self.b + self.b * length / self.average_document_length) if self.average_document_length else self.k1
        return sum(
            self._idf(self.document_frequency[term]) * frequency * (self.k1 + 1) / (frequency + norm)
            for term in query_terms
            if (frequency := self.postings.get(term, {}).get(code, 0))
        )

    def rank(self, query: str, limit: int | None = None) -> list[tuple[str, float]]:
        query_terms = set(tokenize(query))
        scores: dict[str, float] = defaultdict(float)
        for term in query_terms:
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self._idf(self.document_frequency[term])
            for code, frequency in posting.items():
                length = len(self.tokens[code])
                norm = self.k1 * (1 - self.b + self.b * length / self.average_document_length) if self.average_document_length else self.k1
                scores[code] += idf * frequency * (self.k1 + 1) / (frequency + norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if limit is None:
            return ranked + [(code, 0.0) for code in sorted(set(self.documents) - set(scores))]
        if len(ranked) < limit:
            ranked.extend((code, 0.0) for code in sorted(set(self.documents) - set(scores))[: limit - len(ranked)])
        return ranked[:limit]
