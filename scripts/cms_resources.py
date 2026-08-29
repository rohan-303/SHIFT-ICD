"""Authoritative CMS resources used by the initial SHIFT-ICD benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://www.cms.gov"
LANDING_PAGE = (
    "https://www.cms.gov/medicare/coding-billing/icd-10-codes/"
    "icd-10-cm-icd-10-pcs-gem-archive"
)
ICD9_TITLES_PAGE = (
    "https://www.cms.gov/medicare/coding-billing/icd-10-codes/"
    "icd-9-cm-diagnosis-procedure-codes-abbreviated-and-full-code-titles"
)


@dataclass(frozen=True)
class Resource:
    name: str
    filename: str
    url: str
    release_year: int | str
    direction: str
    sha256: str
    license_terms: str
    notes: str


RESOURCES = (
    Resource(
        name="2018 General Equivalence Mappings (GEMs)",
        filename="2018-icd-10-cm-general-equivalence-mappings.zip",
        url=f"{BASE_URL}/medicare/coding/icd10/downloads/2018-icd-10-cm-general-equivalence-mappings.zip",
        release_year=2018,
        direction="contains forward and reverse diagnosis GEM files",
        sha256="1c5e5f14026ace48437a0d1c485d282fb5d171c357c6ab4f7798fc0a1e3624a2",
        license_terms="CMS public distribution; retain official provenance and consult CMS terms.",
        notes="Archive contains diagnosis GEM text files and two CMS technical PDF documents.",
    ),
    Resource(
        name="2018 ICD-10-CM Code Descriptions in Tabular Order",
        filename="2018-icd-10-code-descriptions.zip",
        url=f"{BASE_URL}/medicare/coding/icd10/downloads/2018-icd-10-code-descriptions.zip",
        release_year=2018,
        direction="ICD-10-CM diagnosis descriptions",
        sha256="d4954a3fa02e0bfbecb10b20864198dfa50862ded93f2d9d7313e9fe07ef0dfc",
        license_terms=(
            "CMS/NCHS public distribution; retain official provenance and consult source terms."
        ),
        notes="Includes tabular-order text files, addenda, and PDF representations.",
    ),
    Resource(
        name="ICD-9-CM Version 32 Full and Abbreviated Code Titles",
        filename="icd-9-cm-v32-master-descriptions.zip",
        url=f"{BASE_URL}/medicare/coding/icd9providerdiagnosticcodes/downloads/icd-9-cm-v32-master-descriptions.zip",
        release_year="Version 32 / effective October 1, 2014",
        direction="ICD-9-CM diagnosis and procedure descriptions",
        sha256="45a7d05ddcadf124af88375b64cdf068bb1e3f999ce7fdacb91f65f4e6d55f08",
        license_terms=(
            "CMS public distribution; retain official provenance and consult source terms."
        ),
        notes="Diagnosis and surgery title files are bundled; diagnosis text is relevant here.",
    ),
)


def raw_directory(project_root: Path) -> Path:
    return project_root / "data" / "raw" / "cms" / "2018_gem"
