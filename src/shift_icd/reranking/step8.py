from __future__ import annotations

import gzip
import hashlib
import json
import math
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "ncbi/MedCPT-Cross-Encoder"
MODEL_REVISION = "71caf65d4927987813984f54c284405a13fcca49"
MAX_LENGTH = 96
LIST_SIZE = 8
ORDINARY_KINDS = {"SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"}
EXCLUDED_KINDS = {"NO_MAP", "COMBINATION", "COMBINATION_WITH_ALTERNATIVES"}
OBJECTIVES = ("BCE", "SET_POSITIVE_LISTWISE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_pair(source_description: str, target_description: str) -> tuple[str, str]:
    if not isinstance(source_description, str) or not isinstance(target_description, str):
        raise TypeError("descriptions must be strings")
    return source_description, target_description


def extract_relevance_scores(output: Any, *, num_labels: int) -> list[float]:
    logits = getattr(output, "logits", output)
    shape = tuple(getattr(logits, "shape", ()))
    if num_labels != 1 or len(shape) != 2 or shape[1] != 1:
        raise ValueError(f"unsupported MedCPT output contract: num_labels={num_labels}, shape={shape}")
    values = logits[:, 0]
    return [float(value) for value in values.detach().float().cpu().tolist()]


def validate_membership(before: Sequence[Sequence[str]], after: Sequence[Sequence[str]]) -> None:
    if len(before) != len(after):
        raise ValueError("reranking changed source count")
    for left, right in zip(before, after, strict=True):
        if len(left) != len(right) or set(left) != set(right):
            raise ValueError("reranking changed candidate membership")


def hit_at_k(ranked_codes: Sequence[Sequence[str]], gold_codes: Sequence[set[str]], k: int) -> float:
    if k <= 0 or len(ranked_codes) != len(gold_codes) or not ranked_codes:
        raise ValueError("invalid evaluator inputs")
    return sum(bool(set(ranked[:k]) & gold) for ranked, gold in zip(ranked_codes, gold_codes, strict=True)) / len(ranked_codes)


def structural_coverage_at_k(ranked_codes: Sequence[Sequence[str]], gold_codes: Sequence[set[str]], k: int) -> float:
    """Measure fraction of each source's valid target set present in the top-k."""
    if k <= 0 or len(ranked_codes) != len(gold_codes) or not ranked_codes or any(not gold for gold in gold_codes):
        raise ValueError("invalid structural evaluator inputs")
    return sum(len(set(ranked[:k]) & gold) / len(gold) for ranked, gold in zip(ranked_codes, gold_codes, strict=True)) / len(ranked_codes)


def evaluator_membership_invariance(
    before: Sequence[Sequence[str]], after: Sequence[Sequence[str]], gold_codes: Sequence[set[str]], k: int = 100
) -> dict[str, float]:
    validate_membership(before, after)
    return {
        "hit_at_100_before": hit_at_k(before, gold_codes, k),
        "hit_at_100_after": hit_at_k(after, gold_codes, k),
        "structural_at_100_before": structural_coverage_at_k(before, gold_codes, k),
        "structural_at_100_after": structural_coverage_at_k(after, gold_codes, k),
    }


def bce_loss(logits: Sequence[float], labels: Sequence[float]) -> float:
    if len(logits) != len(labels) or not logits:
        raise ValueError("BCE shape mismatch")
    return sum(max(x, 0.0) - x * y + math.log1p(math.exp(-abs(x))) for x, y in zip(logits, labels, strict=True)) / len(logits)


def set_positive_listwise_loss(scores: Sequence[float], positive_indices: Iterable[int]) -> float:
    values = [float(x) for x in scores]
    positives = sorted(set(int(i) for i in positive_indices))
    if not values or not positives or any(i < 0 or i >= len(values) for i in positives):
        raise ValueError("invalid listwise inputs")
    denominator = _logsumexp(values)
    numerator = _logsumexp([values[i] for i in positives])
    return denominator - numerator


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def ordinary_training_eligible(row: dict[str, Any]) -> bool:
    return row.get("mapping_kind") in ORDINARY_KINDS and bool(row.get("gold_codes"))


def construct_training_list(rows: Sequence[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Construct one source-balanced list-8; positives are never negatives."""
    import random

    positives = [r for r in rows if r.get("candidate_is_gold")]
    if not positives or len(positives) > LIST_SIZE:
        raise ValueError("source cannot be represented by a list-8 without dropping positives")
    negatives = [r for r in rows if not r.get("candidate_is_gold")]
    rng = random.Random(seed)
    rng.shuffle(negatives)
    selected = positives + negatives[: LIST_SIZE - len(positives)]
    return sorted(selected, key=lambda r: int(r["candidate_rank"]))


def load_authoritative_descriptions(root: Path, direction: str) -> tuple[dict[str, str], dict[str, set[str]]]:
    benchmark = root / "data/benchmarks/cms_track_a/v1.0/forward/all.jsonl"
    source: dict[str, str] = {}
    gold: dict[str, set[str]] = {}
    with benchmark.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["direction"] == direction:
                source[row["benchmark_id"]] = row["source_label"]
                gold[row["benchmark_id"]] = set(row.get("valid_target_codes", []))
    return source, gold


def load_code_descriptions(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    archives = [
        (root / "data/raw/cms/2018_gem/2018-icd-10-code-descriptions.zip", "icd10cm_codes_2018.txt"),
        (root / "data/raw/cms/2018_gem/icd-9-cm-v32-master-descriptions.zip", "CMS32_DESC_LONG_DX.txt"),
    ]
    for archive, member in archives:
        with zipfile.ZipFile(archive) as zipped, zipped.open(member) as stream:
            for raw in stream:
                text = raw.decode("latin-1").rstrip("\r\n")
                code, separator, description = text.partition("    ")
                if not separator:
                    code, separator, description = text.partition("  ")
                if code and description:
                    result[code.strip()] = description.strip()
    return result


def load_candidate_groups(path: Path) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            groups.setdefault(str(row["source_id"]), []).append(row)
    ordered = []
    for source_id in sorted(groups):
        rows = sorted(groups[source_id], key=lambda row: int(row["candidate_rank"]))
        ordered.append(rows)
    return ordered


@dataclass(frozen=True)
class Step8TestPolicy:
    allowed: bool = False
    lock_path: Path | None = None

    def require(self) -> None:
        if not self.allowed:
            raise PermissionError("STEP8_TEST_ACCESS_FORBIDDEN")
        if self.lock_path is None or not self.lock_path.is_file():
            raise PermissionError("STEP8_TEST_LOCK_REQUIRED")


def deterministic_candidate_order(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row["source_id"]), int(row["candidate_rank"])))
