from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Direction = Literal["ICD9CM_TO_ICD10CM", "ICD10CM_TO_ICD9CM"]
MappingKind = Literal[
    "NO_MAP",
    "SINGLE_EXACT",
    "SINGLE_APPROXIMATE",
    "ALTERNATIVE",
    "COMBINATION",
    "COMBINATION_WITH_ALTERNATIVES",
    "MULTI_SCENARIO",
]


class CodeDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    source: str
    release: str
    long_description: str | None = None
    short_description: str | None = None


class GemRawRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row_id: str
    direction: Direction
    source_code_raw: str
    target_code_raw: str
    approximate: bool
    no_map: bool
    combination: bool
    scenario: int
    choice_list: int
    raw_flag_string: str
    raw_record: str

    @field_validator("raw_flag_string")
    @classmethod
    def flags_are_five_characters(cls, value: str) -> str:
        if len(value) != 5 or any(not char.isdigit() for char in value):
            raise ValueError("GEM flags must contain exactly five digits")
        return value

    @field_validator("scenario", "choice_list")
    @classmethod
    def identifiers_are_nonnegative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("GEM identifiers cannot be negative")
        return value

    @field_validator("target_code_raw")
    @classmethod
    def no_map_target_is_blank_or_marker(cls, value: str) -> str:
        return value

    def model_post_init(self, __context: object) -> None:
        if self.no_map and self.target_code_raw.strip() not in {"", "NoDx", "NoPCS"}:
            raise ValueError("no-map rows cannot contain a normal target code")
        if not self.combination and (self.scenario != 0 or self.choice_list != 0):
            raise ValueError("non-combination GEM rows must have scenario and choice list zero")


class GemMappingRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row_id: str
    direction: Direction
    source_version: str
    target_version: str
    source_code: str
    target_code: str | None
    source_label: str | None = None
    target_label: str | None = None
    source_short_description: str | None = None
    target_short_description: str | None = None
    approximate: bool
    no_map: bool
    combination: bool
    scenario: int
    choice_list: int


class GemChoiceAlternative(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_code: str
    target_label: str | None = None
    raw_row_ids: list[str] = Field(min_length=1)
    approximate: bool


class GemChoiceList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    choice_list_id: int = Field(ge=1)
    alternatives: list[GemChoiceAlternative] = Field(min_length=1)


class GemScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: int = Field(ge=1)
    choice_lists: list[GemChoiceList] = Field(min_length=1)
    raw_row_ids: list[str] = Field(min_length=1)
    approximate: bool
    combination: bool = True


class GemSourceMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canonical_schema_version: str = "1.0"
    source_code: str
    source_label: str | None = None
    direction: Direction
    source_version: str
    target_version: str
    mapping_kind: MappingKind
    raw_row_ids: list[str] = Field(min_length=1)
    scenarios: list[GemScenario] = Field(default_factory=list)
    target_row_count: int = Field(ge=0)
    alternative_count: int = Field(ge=0)
    required_component_count: int = Field(ge=0)
    scenario_count: int = Field(ge=0)
    choice_list_count: int = Field(ge=0)
    unique_target_count: int = Field(ge=0)
    valid_mapping_set_count: int = Field(ge=0)
    approximate_any: bool
    approximate_all: bool
    no_map: bool
    combination: bool
    eligible_simple_mapping: bool
    eligible_approximate_mapping: bool
    eligible_no_map: bool
    eligible_alternative_mapping: bool
    eligible_combination_mapping: bool
    eligible_multiscenario_mapping: bool
    description_available: bool
    unusually_large_mapping_structure: bool
