from __future__ import annotations

import re
import unicodedata


def clean_dense_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", normalized).strip()


def format_qwen_query(instruction: str, query: str) -> str:
    return f"Instruct: {instruction}\nQuery:{clean_dense_text(query)}"
