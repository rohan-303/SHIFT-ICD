# ruff: noqa: E501
from __future__ import annotations

import json
import statistics

import pandas as pd  # type: ignore[import-untyped]
from run_bm25_v1 import ROOT, build_corpora, evaluate, load_examples, write_json


def main() -> None:
    config = json.loads((ROOT / "artifacts/experiments/bm25_v1/config.json").read_text())
    selection = json.loads((ROOT / "artifacts/experiments/bm25_v1/selection.json").read_text())
    params = (selection["winner"]["k1"], selection["winner"]["b"])
    examples = load_examples(ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl")
    rows = pd.read_parquet(ROOT / "data/processed/cms/2018_gem/normalized_rows.parquet")
    dev = [e for e in examples if e.direction == "ICD9CM_TO_ICD10CM" and e.split == "dev"]
    forward_test = [e for e in examples if e.direction == "ICD9CM_TO_ICD10CM" and e.split == "test"]
    corpora_q2, profiles_q2 = build_corpora(rows, "q2_short_plus_long")
    q2_rows, q2_vectors, _ = evaluate(dev, corpora_q2, "BM25 Dev-Tuned", params, config["seed"])
    noncomb = [r for r in q2_rows if not r["no_map"] and r["mapping_kind"] not in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}]
    q2_hit10 = statistics.mean(r["Hit@10"] for r in noncomb)
    write_json(ROOT / "artifacts/experiments/bm25_v1/q2_dev_ablation.json", {"representation": "q2_short_plus_long", "frozen_parameters": {"k1": params[0], "b": params[1]}, "dev_noncombination_answerable_hit_at_10": q2_hit10, "q1_selected_dev_hit_at_10": selection["winner"]["noncombination_answerable_hit_at_10"], "q2_profiles": profiles_q2})
    corpora_q1, _ = build_corpora(rows, config["primary_text"])
    q1_rows, _, no_map = evaluate(forward_test, corpora_q1, "BM25 Dev-Tuned", params, config["seed"], ROOT / "artifacts/experiments/bm25_v1/rankings_forward_stratified", collect_rows=True)
    write_json(ROOT / "artifacts/experiments/bm25_v1/no_map_diagnostics.json", {"count": len(no_map), "mean_max_bm25_score": statistics.mean(r["max_bm25_score"] for r in no_map) if no_map else None, "median_max_bm25_score": statistics.median(r["max_bm25_score"] for r in no_map) if no_map else None, "rows": no_map})
    errors = [r for r in q1_rows if not r["no_map"] and not r["Hit@10"]]
    error_ids = {r["benchmark_id"] for r in sorted(errors, key=lambda r: (r["mapping_kind"], r["lexical_difficulty"], r["benchmark_id"]))[:100]}
    selected = {e.benchmark_id: e for e in forward_test if e.benchmark_id in error_ids}
    ranking_path = ROOT / "artifacts/experiments/bm25_v1/rankings_forward_stratified/bm25_dev-tuned.jsonl"
    rankings: dict[str, list[dict[str, object]]] = {}
    for line in ranking_path.open(encoding="utf-8"):
        row = json.loads(line)
        rankings.setdefault(row["benchmark_id"], []).append(row)
    error_rows = []
    for bid, ex in selected.items():
        top = rankings.get(bid, [])[:10]
        best = next((r["rank"] for r in rankings.get(bid, []) if r["target_code"] in ex.valid_target_codes), None)
        error_rows.append({"benchmark_id": bid, "source_code": ex.source_code, "source_description": ex.source_label, "mapping_kind": ex.mapping_kind, "valid_targets": ex.valid_target_codes, "top10": top, "lexical_difficulty": ex.lexical_metadata.get("lexical_difficulty"), "best_valid_target_rank": best, "failure_category": None})
    error_path = ROOT / "artifacts/error_analysis/bm25_v1_forward_errors.jsonl"
    error_path.parent.mkdir(parents=True, exist_ok=True)
    with error_path.open("w", encoding="utf-8") as f:
        for row in error_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    v = next((e for e in forward_test if e.source_code == "V54.12"), None)
    vrow = next((r for r in q1_rows if r["benchmark_id"] == v.benchmark_id), None) if v else None
    vrank = rankings.get(v.benchmark_id, []) if v else []
    if v and vrow:
        valid = set(v.valid_target_codes)
        write_json(ROOT / "artifacts/experiments/bm25_v1/v54_12_analysis.json", {"benchmark_id": v.benchmark_id, "source_code": v.source_code, "source_description": v.source_label, "valid_target_count": len(valid), "best_valid_target_rank": next((r["rank"] for r in vrank if r["target_code"] in valid), None), "valid_targets_within_k": {str(k): sum(r["target_code"] in valid for r in vrank[:k]) for k in (5,10,25,50,100)}, "top10": vrank[:10], "warning": "Large acceptable target sets increase Hit@K probability; gold semantics are unchanged."})
    print(json.dumps({"q2_dev_hit10": q2_hit10, "q1_dev_hit10": selection["winner"]["noncombination_answerable_hit_at_10"], "forward_test_rows": len(q1_rows), "no_map_rows": len(no_map), "error_rows": len(error_rows), "v54_12_found": v is not None}, sort_keys=True))


if __name__ == "__main__":
    main()
