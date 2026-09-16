from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HierarchyRecord:
    code: str
    ontology: str
    parent_code: str | None
    ancestor_chain: tuple[str, ...]
    depth: int
    root_code: str
    chapter: str
    block: str
    family: str
    provenance: str
    audit_status: str = "PASS"


def _family(code: str) -> str:
    return code[:3]


def _chapter(code: str) -> str:
    return code[:1]


def _block(code: str) -> str:
    return code[:2] if len(code) >= 2 else code


def build_prefix_hierarchy(codes: Iterable[str], *, ontology: str, provenance: str) -> dict[str, HierarchyRecord]:
    """Build a deterministic prefix hierarchy from terminology identities only.

    Prefix links are used because they are available in both committed ICD code
    corpora without consulting GEM mappings. Missing intermediate prefixes are
    materialized as virtual structural nodes with no learned description.
    """
    normalized_codes = tuple(sorted({str(code).strip().upper().replace(".", "") for code in codes if str(code).strip()}))
    virtual_prefixes = {code[:i] for code in normalized_codes for i in range(1, len(code) + 1)}
    normalized = tuple(sorted(virtual_prefixes))
    code_set = set(normalized)
    parents: dict[str, str | None] = {}
    for code in normalized:
        parents[code] = max((code[:i] for i in range(1, len(code)) if code[:i] in code_set), key=len, default=None)
    records: dict[str, HierarchyRecord] = {}
    for code in normalized:
        chain: list[str] = []
        cursor = parents[code]
        seen: set[str] = set()
        while cursor is not None:
            if cursor in seen:
                raise ValueError(f"HIERARCHY_CYCLE:{code}")
            seen.add(cursor)
            chain.append(cursor)
            cursor = parents[cursor]
        root = chain[-1] if chain else code
        records[code] = HierarchyRecord(
            code=code,
            ontology=ontology,
            parent_code=parents[code],
            ancestor_chain=tuple(chain),
            depth=len(chain),
            root_code=root,
            chapter=_chapter(code),
            block=_block(code),
            family=_family(code),
            provenance=provenance,
        )
    return records


def audit_hierarchy(records: dict[str, HierarchyRecord]) -> dict[str, int | bool]:
    invalid_parent = sum(int(record.parent_code is not None and record.parent_code not in records) for record in records.values())
    cycle_count = 0
    chain_errors = 0
    depth_errors = 0
    for record in records.values():
        seen: set[str] = set()
        cursor: str | None = record.code
        while cursor is not None:
            if cursor in seen:
                cycle_count += 1
                break
            seen.add(cursor)
            cursor = records[cursor].parent_code if cursor in records else None
        expected_chain: list[str] = []
        cursor = record.parent_code
        while cursor is not None and cursor in records:
            expected_chain.append(cursor)
            cursor = records[cursor].parent_code
        chain_errors += int(tuple(expected_chain) != record.ancestor_chain)
        depth_errors += int(record.depth != len(record.ancestor_chain))
    return {
        "node_count": len(records),
        "root_count": sum(int(record.parent_code is None) for record in records.values()),
        "invalid_parent_count": invalid_parent,
        "cycle_count": cycle_count,
        "ancestor_chain_error_count": chain_errors,
        "depth_consistency_error_count": depth_errors,
        "target_universe_coverage_count": len(records),
        "integrity_pass": not any((invalid_parent, cycle_count, chain_errors, depth_errors)),
    }
