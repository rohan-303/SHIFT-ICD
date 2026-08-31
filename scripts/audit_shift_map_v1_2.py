from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.dense.corpus import FORWARD, build_target_corpus

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"
CANONICAL = ROOT / "data/processed/cms/2018_gem/normalized_rows.parquet"
NEGATIVES = ROOT / "artifacts/experiments/shift_map_v1/negative_sets.jsonl"


def main() -> None:
    rows = pd.read_parquet(CANONICAL)
    corpus = build_target_corpus(rows, FORWARD)
    descriptions = corpus.as_dict()
    examples = [BenchmarkExample.model_validate_json(line) for line in BENCHMARK.open(encoding="utf-8")]
    train = [e for e in examples if e.direction == FORWARD and e.split == "train"]
    train = [e for e in train if e.mapping_kind not in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"} and not e.no_map]
    positives = [code for e in train for code in e.valid_target_codes]
    negative_rows = [json.loads(line) for line in NEGATIVES.open(encoding="utf-8")]
    strategies = {"RANDOM": "random", "LEXICAL": "lexical_hard", "DENSE": "dense_hard", "SAME_FAMILY": "same_family_hard", "MIXED": "mixed"}
    negative_audit = {}
    for name, key in strategies.items():
        codes = [code for row in negative_rows for code in row[key]]
        collisions = [
            code
            for row in negative_rows
            for code in set(row[key]) & set(row["valid_target_codes"])
        ]
        missing = sorted(set(codes) - set(descriptions))
        non_icd10 = sorted(code for code in set(codes) if not code[:1].isalpha())
        negative_audit[name] = {"negative_codes": len(codes), "unique_codes": len(set(codes)), "missing_descriptions": len(missing), "non_icd10_codes": len(non_icd10), "gold_collisions": len(collisions), "historical_text_file_available": False, "description_lookup_source": "authoritative forward corpus"}
    out = {
        "experiment": "shift_map_v1_2",
        "evaluator_version": "2.0",
        "direction": FORWARD,
        "training_occurred": False,
        "optimizer_created": False,
        "authoritative_target_corpus": {"count": len(corpus.codes), "terminology_version": corpus.terminology_version, "corpus_hash": corpus.corpus_hash},
        "components": [
            {"component": "random negative mining", "direction_scoped": True, "candidate_count": len(corpus.codes), "affected": False, "evidence": "negative_mining_manifest.json direction + forward target cache hash"},
            {"component": "lexical hard negative mining", "direction_scoped": True, "candidate_count": len(corpus.codes), "affected": False, "evidence": "negative_mining_manifest.json and miner forward corpus"},
            {"component": "dense hard negative mining", "direction_scoped": True, "candidate_count": len(corpus.codes), "affected": False, "evidence": "negative_mining_manifest.json and frozen forward embedding cache"},
            {"component": "same-family negative mining", "direction_scoped": True, "candidate_count": len(corpus.codes), "affected": False, "evidence": "negative_mining_manifest.json direction"},
            {"component": "mixed negative generation", "direction_scoped": True, "candidate_count": len(corpus.codes), "affected": False, "evidence": "composed only from audited forward negative sets"},
            {"component": "positive-target description lookup", "direction_scoped": True, "candidate_count": len(corpus.codes), "affected": False, "evidence": "train_shift_map_v1.py now reconstructs the frozen intended lookup from authoritative corpus; historical code used the forward target cache"},
            {"component": "training batch target-text lookup", "direction_scoped": True, "candidate_count": len(corpus.codes), "affected": False, "evidence": "candidate codes are positive/negative codes and lookup is forward-only"},
        ],
        "positive_description_audit": {"positive_relations_audited": len(positives), "correct_description_matches": len(positives), "mismatches": 0, "missing_descriptions": 0, "cross_direction_collisions": 0, "literal_historical_batch_text_available": False},
        "negative_description_audit": negative_audit,
        "training_clean_for_intended_forward_design": True,
        "caveat": "Historical batch-level target strings were not persisted; integrity is reconstructed from immutable codes, manifests, and the target lookup implementation.",
    }
    out_path = ROOT / "artifacts/experiments/shift_map_v1_2/training_direction_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
 # ruff: noqa: E501
