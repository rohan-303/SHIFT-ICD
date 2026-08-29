from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import product
from math import prod
from typing import cast

from .schemas import (
    CodeDescription,
    GemChoiceAlternative,
    GemChoiceList,
    GemMappingRow,
    GemRawRow,
    GemScenario,
    GemSourceMapping,
    MappingKind,
)

SOURCE_VERSION = {
    "ICD9CM_TO_ICD10CM": "ICD-9-CM",
    "ICD10CM_TO_ICD9CM": "ICD-10-CM",
}
TARGET_VERSION = {
    "ICD9CM_TO_ICD10CM": "ICD-10-CM",
    "ICD10CM_TO_ICD9CM": "ICD-9-CM",
}


def normalize_mapping_row(
    row: GemRawRow,
    source_descriptions: Mapping[str, CodeDescription] | None = None,
    target_descriptions: Mapping[str, CodeDescription] | None = None,
) -> GemMappingRow:
    source_code = row.source_code_raw.strip()
    target_code = None if row.no_map else (row.target_code_raw.strip() or None)
    source = (source_descriptions or {}).get(source_code)
    target = (target_descriptions or {}).get(target_code) if target_code else None
    return GemMappingRow(
        row_id=row.row_id,
        direction=row.direction,
        source_version=SOURCE_VERSION[row.direction],
        target_version=TARGET_VERSION[row.direction],
        source_code=source_code,
        target_code=target_code,
        source_label=source.long_description if source else None,
        target_label=target.long_description if target else None,
        source_short_description=source.short_description if source else None,
        target_short_description=target.short_description if target else None,
        approximate=row.approximate,
        no_map=row.no_map,
        combination=row.combination,
        scenario=row.scenario,
        choice_list=row.choice_list,
    )


def count_valid_mapping_sets(scenarios: Iterable[GemScenario]) -> int:
    return sum(prod(len(choice.alternatives) for choice in scenario.choice_lists) for scenario in scenarios)


def enumerate_mapping_sets(scenarios: Iterable[GemScenario], max_sets: int = 1000) -> list[list[str]]:
    scenarios = list(scenarios)
    count = count_valid_mapping_sets(scenarios)
    if count > max_sets:
        raise ValueError(f"refusing enumeration: {count} sets exceed safety limit {max_sets}")
    result: list[list[str]] = []
    for scenario in scenarios:
        choices = [choice.alternatives for choice in scenario.choice_lists]
        result.extend([[alternative.target_code for alternative in selection] for selection in product(*choices)])
    return result


def _build_scenarios(rows: list[GemMappingRow]) -> list[GemScenario]:
    grouped: dict[int, dict[int, dict[str, GemChoiceAlternative]]] = {}
    for row in rows:
        if row.no_map or not row.combination or not row.target_code:
            continue
        scenario = grouped.setdefault(row.scenario, {})
        choices = scenario.setdefault(row.choice_list, {})
        existing = choices.get(row.target_code)
        if existing:
            existing.raw_row_ids.append(row.row_id)
            existing.approximate = existing.approximate or row.approximate
        else:
            choices[row.target_code] = GemChoiceAlternative(
                target_code=row.target_code,
                target_label=row.target_label,
                raw_row_ids=[row.row_id],
                approximate=row.approximate,
            )
    return [
        GemScenario(
            scenario_id=scenario_id,
            choice_lists=[
                GemChoiceList(choice_list_id=choice_id, alternatives=list(alternatives.values()))
                for choice_id, alternatives in sorted(choice_lists.items())
            ],
            raw_row_ids=[row.row_id for row in rows if row.scenario == scenario_id],
            approximate=any(row.approximate for row in rows if row.scenario == scenario_id),
        )
        for scenario_id, choice_lists in sorted(grouped.items())
    ]


def canonicalize_source(
    rows: list[GemRawRow] | list[GemMappingRow],
    source_descriptions: Mapping[str, CodeDescription] | None = None,
    target_descriptions: Mapping[str, CodeDescription] | None = None,
    enumeration_limit: int = 1000,
) -> GemSourceMapping:
    if not rows:
        raise ValueError("cannot canonicalize an empty source group")
    normalized = [
        row if isinstance(row, GemMappingRow) else normalize_mapping_row(row, source_descriptions, target_descriptions) for row in rows
    ]
    source_codes = {row.source_code for row in normalized}
    if len(source_codes) != 1:
        raise ValueError("canonicalize_source requires one source code")
    direction = normalized[0].direction
    targets = {row.target_code for row in normalized if row.target_code}
    combinations = [row for row in normalized if row.combination and not row.no_map]
    scenarios = _build_scenarios(normalized)
    no_map = all(row.no_map for row in normalized)
    scenario_count = len(scenarios)
    choice_list_count = sum(len(s.choice_lists) for s in scenarios)
    required_components = max((len(s.choice_lists) for s in scenarios), default=0)
    valid_count = count_valid_mapping_sets(scenarios)
    if not combinations and not no_map:
        valid_count = len(targets)
    approximate_any = any(row.approximate for row in normalized)
    approximate_all = all(row.approximate for row in normalized)
    if no_map:
        kind = "NO_MAP"
    elif combinations and any(len(c.alternatives) > 1 for s in scenarios for c in s.choice_lists):
        kind = "COMBINATION_WITH_ALTERNATIVES"
    elif combinations:
        kind = "COMBINATION"
    elif scenario_count > 1:
        kind = "MULTI_SCENARIO"
    elif len(targets) > 1:
        kind = "ALTERNATIVE"
    elif approximate_all:
        kind = "SINGLE_APPROXIMATE"
    else:
        kind = "SINGLE_EXACT"
    unusually_large = sum(1 for row in normalized if row.target_code) > 100 or valid_count > enumeration_limit
    source_label = normalized[0].source_label
    return GemSourceMapping(
        source_code=normalized[0].source_code,
        source_label=source_label,
        direction=direction,
        source_version=normalized[0].source_version,
        target_version=normalized[0].target_version,
        mapping_kind=cast(MappingKind, kind),
        raw_row_ids=[row.row_id for row in normalized],
        scenarios=scenarios,
        target_row_count=sum(1 for row in normalized if row.target_code),
        alternative_count=(len(targets) if not combinations else sum(len(c.alternatives) for s in scenarios for c in s.choice_lists)),
        required_component_count=required_components,
        scenario_count=scenario_count,
        choice_list_count=choice_list_count,
        unique_target_count=len(targets),
        valid_mapping_set_count=valid_count,
        approximate_any=approximate_any,
        approximate_all=approximate_all,
        no_map=no_map,
        combination=bool(combinations),
        eligible_simple_mapping=(not no_map and not combinations and len(targets) == 1 and not approximate_any),
        eligible_approximate_mapping=approximate_any,
        eligible_no_map=no_map,
        eligible_alternative_mapping=(not combinations and len(targets) > 1),
        eligible_combination_mapping=bool(combinations),
        eligible_multiscenario_mapping=scenario_count > 1,
        description_available=source_label is not None,
        unusually_large_mapping_structure=unusually_large,
    )


def canonicalize_rows(
    rows: list[GemRawRow],
    source_descriptions: Mapping[str, CodeDescription],
    target_descriptions: Mapping[str, CodeDescription],
    enumeration_limit: int = 1000,
) -> tuple[list[GemMappingRow], list[GemSourceMapping]]:
    normalized = [normalize_mapping_row(row, source_descriptions, target_descriptions) for row in rows]
    grouped: dict[str, list[GemMappingRow]] = {}
    for row in normalized:
        grouped.setdefault(row.source_code, []).append(row)
    sources = [canonicalize_source(group, enumeration_limit=enumeration_limit) for group in grouped.values()]
    return normalized, sources
