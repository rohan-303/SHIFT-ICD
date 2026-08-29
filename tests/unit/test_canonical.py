from pathlib import Path

import pytest

from shift_icd.data.canonical import (
    canonicalize_source,
    count_valid_mapping_sets,
    enumerate_mapping_sets,
)
from shift_icd.data.descriptions import parse_icd9_long_titles, parse_icd10_codes
from shift_icd.data.gem_parser import parse_gem_record
from shift_icd.data.schemas import GemRawRow, GemSourceMapping


def row(raw: str, direction: str = "ICD9CM_TO_ICD10CM") -> GemRawRow:
    return parse_gem_record(raw, direction=direction, row_id="r1")


def test_exact_single_mapping():
    result = canonicalize_source([row("0010  A000    00000")])
    assert result.mapping_kind == "SINGLE_EXACT"
    assert result.scenario_count == 0
    assert result.valid_mapping_set_count == 1


def test_approximate_single_mapping():
    result = canonicalize_source([row("0010  A000    10000")])
    assert result.mapping_kind == "SINGLE_APPROXIMATE"
    assert result.approximate_any is True


def test_no_map_rejects_normal_target():
    no_map = row("0010          11000")
    assert no_map.no_map and no_map.target_code_raw.strip() == ""
    with pytest.raises(ValueError):
        row("0010  A000    11000")


def test_alternatives_and_multiple_scenarios():
    rows = [
        row("0010  A000    10000"),
        row("0010  A001    10000"),
    ]
    result = canonicalize_source(rows)
    assert result.mapping_kind == "ALTERNATIVE"
    assert result.unique_target_count == 2


def test_combination_cartesian_product_and_symbolic_structure():
    rows = [
        row("0010  A000    10111"),
        row("0010  A001    10111"),
        row("0010  A009    10112"),
        row("0010  A010    10112"),
    ]
    result = canonicalize_source(rows)
    assert result.mapping_kind == "COMBINATION_WITH_ALTERNATIVES"
    assert result.required_component_count == 2
    assert result.valid_mapping_set_count == 4
    assert count_valid_mapping_sets(result.scenarios) == 4
    assert len(enumerate_mapping_sets(result.scenarios, max_sets=10)) == 4


def test_enumeration_safety_threshold():
    rows = [row(f"0010  A{i:03d}    1011{j}") for i in range(5) for j in range(1, 6)]
    result = canonicalize_source(rows)
    assert result.valid_mapping_set_count > 2
    with pytest.raises(ValueError, match="safety limit"):
        enumerate_mapping_sets(result.scenarios, max_sets=2)


def test_direction_and_traceability():
    result = canonicalize_source([row("A000    0010  00000", "ICD10CM_TO_ICD9CM")])
    assert result.direction == "ICD10CM_TO_ICD9CM"
    assert result.raw_row_ids == ["r1"]


def test_description_parsers(tmp_path: Path):
    icd9 = tmp_path / "icd9.txt"
    icd9.write_bytes("0010  Choléra due to Vibrio cholerae\n".encode("cp1252"))
    assert parse_icd9_long_titles(icd9)["0010"].long_description.startswith("Chol")

    icd10 = tmp_path / "icd10.txt"
    icd10.write_text("A000    Cholera due to Vibrio cholerae\n", encoding="ascii")
    assert parse_icd10_codes(icd10)["A000"].long_description.startswith("Cholera")


def test_serialization_round_trip():
    result = canonicalize_source([row("0010  A000    00000")])
    restored = GemSourceMapping.model_validate_json(result.model_dump_json())
    assert restored == result
