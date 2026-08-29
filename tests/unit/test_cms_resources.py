import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from cms_resources import Resource
from download_cms_gem import download_resource, sha256_file
from inspect_cms_gem import inspect_archive, inspect_gem_member
from verify_manifest import verify_manifest


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"shift-icd")
    expected = hashlib.sha256(b"shift-icd").hexdigest()
    assert sha256_file(path) == expected


def test_manifest_loads_and_verifies_obtained_archives() -> None:
    manifest = Path("data/raw/cms/2018_gem/manifest.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(payload["artifacts"]) == 3
    assert verify_manifest(manifest) == []


def test_download_refuses_existing_checksum_mismatch(tmp_path: Path) -> None:
    destination = tmp_path / "resource.zip"
    destination.write_bytes(b"wrong")
    resource = Resource(
        name="fixture",
        filename=destination.name,
        url="https://example.invalid/fixture.zip",
        release_year=2018,
        direction="fixture",
        sha256=hashlib.sha256(b"right").hexdigest(),
        license_terms="test",
        notes="test",
    )
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        download_resource(resource, destination)


def test_fixed_width_gem_inspection() -> None:
    report = inspect_gem_member(
        b"0010  A000    00000\r\n7796  NoDx    11000\r\n",
        source_width=6,
        target_width=8,
        direction="ICD-9-CM to ICD-10-CM",
    )
    assert report["total_raw_mapping_rows"] == 2
    assert report["line_lengths"] == {"19": 2}
    assert report["flag_values"] == {"00000": 1, "11000": 1}
    assert report["no_map_records"] == 1


def test_archive_inspection_reports_members(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("2018_I9gem.txt", "0010  A000    00000\r\n")
        bundle.writestr("notes.txt", "unchanged")
    report = inspect_archive(archive)
    assert {member["filename"] for member in report["members"]} == {
        "2018_I9gem.txt",
        "notes.txt",
    }
