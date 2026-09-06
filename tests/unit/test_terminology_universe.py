from pathlib import Path

from shift_icd.terminology.universe import (
    build_icd9_universe,
    build_icd10_universe,
    canonicalize_code,
    corpus_hashes,
)

ROOT = Path(__file__).parents[2]
RAW = ROOT / "data/raw/cms/2018_gem"


def test_code_normalization_is_uppercase_dotless_and_trimmed() -> None:
    assert canonicalize_code(" v58.89 ") == "V5889"
    assert canonicalize_code("e11.9") == "E119"
    assert canonicalize_code(" 0010 ") == "0010"


def test_icd10_universe_uses_only_authoritative_description_member() -> None:
    corpus = build_icd10_universe(RAW / "2018-icd-10-code-descriptions.zip")
    assert len(corpus.records) == 71_704
    assert corpus.records[0].canonical_code == "A000"
    assert corpus.records[-1].canonical_code == "Z9989"
    assert corpus.records[-1].long_description


def test_icd9_universe_uses_only_authoritative_diagnosis_members() -> None:
    corpus = build_icd9_universe(RAW / "icd-9-cm-v32-master-descriptions.zip")
    assert len(corpus.records) == 14_567
    assert any(record.canonical_code == "V5889" for record in corpus.records)


def test_universe_hashes_are_deterministic_and_gem_independent() -> None:
    icd10_a = build_icd10_universe(RAW / "2018-icd-10-code-descriptions.zip")
    icd10_b = build_icd10_universe(RAW / "2018-icd-10-code-descriptions.zip")
    assert corpus_hashes(icd10_a) == corpus_hashes(icd10_b)
    assert corpus_hashes(icd10_a)["code_hash"] == "8ca0d0cf0213581645fe64d9c1e599552a2fb207eb740f32ca09ffeaaf9f8e26"
    assert all("gem" not in str(field).lower() for field in icd10_a.provenance.values())
