from pathlib import Path

from shift_icd.dense.corpus import FORWARD, build_target_corpus_from_terminology
from shift_icd.terminology import build_icd10_universe

ROOT = Path(__file__).parents[2]
RAW = ROOT / "data/raw/cms/2018_gem"


def test_target_corpus_membership_requires_authoritative_terminology() -> None:
    corpus = build_icd10_universe(RAW / "2018-icd-10-code-descriptions.zip")
    target = build_target_corpus_from_terminology(corpus, FORWARD)
    assert len(target.codes) == 71_704
    assert target.codes[0] == "A000"
    assert target.codes[-1] == "Z9989"
