# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path

from run_bm25_v2 import bootstrap, evaluate, load_examples, load_full_corpora, sha256, summarize, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    out = ROOT / "artifacts/experiments/bm25_v2"
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    lock = json.loads((out / "test_lock.json").read_text(encoding="utf-8"))
    if sha256(out / "selection.json") != lock["selection_sha256"]:
        raise RuntimeError("BM25_SELECTION_LOCK_MISMATCH")
    frozen = (float(selection["winner"]["k1"]), float(selection["winner"]["b"]))
    examples = load_examples(ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl")
    corpora, _profiles = load_full_corpora()
    datasets = {
        "forward_stratified_test": [e for e in examples if e.direction == "ICD9CM_TO_ICD10CM" and e.split == "test"],
        "forward_family_held_out_test": [e for e in examples if e.direction == "ICD9CM_TO_ICD10CM" and e.source_family_split == "test"],
        "backward_stratified_test": [e for e in examples if e.direction == "ICD10CM_TO_ICD9CM" and e.split == "test"],
        "backward_family_held_out_test": [e for e in examples if e.direction == "ICD10CM_TO_ICD9CM" and e.source_family_split == "test"],
    }
    metrics = {}
    rows = []
    for name, dataset in datasets.items():
        evaluated, vectors, _diagnostics = evaluate(dataset, corpora, "BM25 Dev-Tuned", frozen, 20260829, out / "rankings", collect_rows=True)
        for row in evaluated:
            row.update({"dataset": name, "system": "BM25 Dev-Tuned"})
        rows.extend(evaluated)
        metrics[name] = {metric: bootstrap(values, 20260829) for metric, values in vectors.items()}
    write_json(out / "test_metrics.json", metrics)
    with (out / "scoped_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tables = ROOT / "reports/tables/bm25_full_universe"
    write_csv(tables / "bm25_forward_test.csv", [r for r in rows if r["dataset"] == "forward_stratified_test"])
    write_csv(tables / "bm25_backward_stratified.csv", [r for r in rows if r["dataset"] == "backward_stratified_test"])
    write_csv(tables / "bm25_backward_family.csv", [r for r in rows if r["dataset"] == "backward_family_held_out_test"])
    write_csv(tables / "bm25_forward_mapping_kind.csv", summarize([r for r in rows if r["dataset"] == "forward_stratified_test"], "mapping_kind"))
    write_csv(tables / "bm25_combination_structural.csv", summarize([r for r in rows if r["dataset"] == "forward_stratified_test" and r["mapping_kind"] in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}], "mapping_kind"))
    write_json(out / "manifest.json", {"experiment": "bm25_full_universe", "selection_sha256": lock["selection_sha256"], "test_lock_sha256": sha256(out / "test_lock.json"), "files": {str(path.relative_to(out)).replace("\\", "/"): sha256(path) for path in out.rglob("*") if path.is_file() and path.name != "manifest.json"}})
    print(json.dumps({"selection": selection["winner"], "datasets": {name: len(data) for name, data in datasets.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
