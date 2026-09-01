# ruff: noqa
from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from shift_map_v1_4_analysis import bootstrap_delta, candidate_coverage, gini, grouped, paired_counts, summarize_values, group_by_cardinality

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts/experiments/shift_map_v1_4"
LED = EXP / "ledgers"
TAB = ROOT / "reports/tables/shift_map_v1_4"
FIG = ROOT / "reports/figures/shift_map_v1_4"
K = (1, 5, 10, 25, 50, 100)
MODELS = ("zero_shot", "l1_seed42", "l2_seed17")

def load(name: str, direction: str, split: str = "test") -> list[dict[str, Any]]:
    return json.loads((LED / f"{name}_{direction}_{split}.json").read_text(encoding="utf8"))["rows"]

def dump(name: str, value: Any) -> None:
    (EXP / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf8")

def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with (TAB / name).open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None

def summary(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    answerable = [r for r in rows if r.get("valid_target_codes") and r.get("mapping_kind") not in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}]
    return {"model": label, "n_total": len(rows), "n_answerable_noncombination": len(answerable), **{f"{m}": metric(answerable, m) for m in [f"Hit@{k}" for k in K] + ["MRR"]}}

def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    rows = {(model, direction, split): load(model, direction, split) for model in MODELS for direction in ("forward", "backward") for split in ("train", "dev", "test")}
    forward = {model: rows[(model, "forward", "test")] for model in MODELS}
    l1, l2, zs = forward["l1_seed42"], forward["l2_seed17"], forward["zero_shot"]
    l1_by = {r["benchmark_id"]: r for r in l1}; l2_by = {r["benchmark_id"]: r for r in l2}; zs_by = {r["benchmark_id"]: r for r in zs}
    assert set(l1_by) == set(l2_by) == set(zs_by)
    dump("config.json", {"experiment_version": "1.4", "purpose": "inference_only_retrieval_freeze", "evaluator_version": "2.0", "benchmark_version": "1.0", "canonical_schema_version": "1.0", "test_exposure": True, "training_occurred": False, "models": {"zero_shot": "FremyCompany/BioLORD-2023@167aab527b238a50ca65224e6319215d2ff4fc9f", "l1_seed42": "artifacts/models/shift_map_v1/final_seed42/epoch_3", "l2_seed17": "artifacts/models/shift_map_v1/v1_3_final_l2_n1_seed17/epoch_3"}})
    dump("l1_ledger.json", {"status": "recovered", "path": "ledgers/l1_seed42_forward_test.json", "scope": "forward_stratified_test", "post_hoc": True, "test_exposed": True, "fields": ["benchmark_id", "best_valid_target_rank", "Hit@1", "Hit@5", "Hit@10", "Hit@25", "Hit@50", "Hit@100", "reciprocal_rank", "top1_target", "top1_score"]})
    paired = []
    for key in [f"Hit@{k}" for k in K] + ["MRR"]:
        a = np.array([float(l2_by[i][key]) for i in l2_by if l2_by[i].get(key) is not None and l1_by[i].get(key) is not None]); b = np.array([float(l1_by[i][key]) for i in l2_by if l2_by[i].get(key) is not None and l1_by[i].get(key) is not None])
        paired.append({"metric": key, "l1": float(np.mean(b)), "l2": float(np.mean(a)), **bootstrap_delta(a, b)})
    counts = {f"K={k}": paired_counts(np.array([int(bool(set(l2_by[i]["valid_target_codes"]) & set(l2_by[i]["ranked_codes"][:k]))) for i in l2_by]), np.array([int(bool(set(l1_by[i]["valid_target_codes"]) & set(l1_by[i]["ranked_codes"][:k]))) for i in l1_by])) for k in (1, 10, 100)}
    dump("l1_l2_paired.json", {"status": "post_hoc_diagnostic_test_exposed_not_model_selection", "population_identity": True, "paired": paired, "success_regression": counts})
    write_csv("l1_vs_l2_paired.csv", paired)
    card_rows = []
    for group, l2_group in group_by_cardinality(l2).items():
        ids = {r["benchmark_id"] for r in l2_group}; group_rows = {m: [({"row": ({"zero_shot": zs_by, "l1_seed42": l1_by, "l2_seed17": l2_by}[m][i]), "id": i}) for i in ids] for m in MODELS}
        for m in MODELS:
            rr = [x["row"] for x in group_rows[m]]; card_rows.append({"group": group, **summary(rr, m)})
    dump("positive_cardinality.json", {"groups": card_rows, "paired_multi_positive": "computed from identical IDs; interpret post-hoc"}); write_csv("l1_vs_l2_positive_cardinality.csv", card_rows)
    for split in ("train", "dev", "test"):
        write_csv("train_dev_test.csv", [summary(rows[(m, "forward", split)], m) | {"split": split} for m in ("zero_shot", "l2_seed17")])
    dump("train_dev_test.json", {"rows": [summary(rows[(m, "forward", s)], m) | {"split": s} for m in ("zero_shot", "l2_seed17") for s in ("train", "dev", "test")], "test_used_for_selection": False})
    for model in MODELS:
        rr = forward[model]
        for field, filename in (("mapping_kind", "combination_transfer.csv"), ("lexical_difficulty", "lexical_low.csv")):
            values = ["LEXICAL_LOW"] if field == "lexical_difficulty" else sorted({r.get(field) for r in rr})
            metric_names = [f"Hit@{k}" for k in K] + ["MRR"] + [f"ChoiceListRecall@{k}" for k in K] + [f"CompleteScenarioRetrieval@{k}" for k in K]
            out = []
            for value in values:
                subset = [r for r in rr if r.get(field) == value]
                out.append({"model": model, "slice": value, "n": len(subset), **{name: metric(subset, name) for name in metric_names}})
            write_csv(filename, out)
    dump("combination_transfer.json", {m: [{"kind": g, **summary([r for r in forward[m] if r.get("mapping_kind") == g], m)} for g in sorted({r.get("mapping_kind") for r in forward[m]}) if g in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}] for m in MODELS})
    dump("lexical_low.json", {m: summary([r for r in forward[m] if r.get("lexical_difficulty") == "LEXICAL_LOW"], m) for m in MODELS})
    geom = []
    for model in MODELS:
        rr = forward[model]; margins = [float(r["top1_score"] - r["top2_score"]) for r in rr if r.get("top1_score") is not None and r.get("top2_score") is not None]
        geom.append({"model": model, "top1_top2_margin_positive": float(np.mean(np.array(margins) > 0)), **summarize_values(margins)})
    dump("score_geometry.json", geom); write_csv("score_geometry.csv", geom)
    hub = {}
    for model in MODELS:
        counts_top = Counter(r.get("top1_target") for r in forward[model]); vals = list(counts_top.values()); total = sum(vals); probs = np.array(vals) / total
        hub[model] = {"unique_top1_targets": len(counts_top), "top20": counts_top.most_common(20), "top10_query_fraction": float(sum(v for _, v in counts_top.most_common(10)) / total), "entropy": float(-np.sum(probs * np.log2(probs))), "gini": gini(vals)}
    dump("hubness.json", hub); write_csv("target_hubness.csv", [{"model": m, **{k: v for k, v in hub[m].items() if k not in {"top20"}}} for m in MODELS])
    no_map = {}
    for model in MODELS:
        groups = {}
        for label, rr in (("NO_MAP", [r for r in forward[model] if not r.get("valid_target_codes")]), ("ANSWERABLE", [r for r in forward[model] if r.get("valid_target_codes")])):
            groups[label] = {"n": len(rr), "max_similarity": summarize_values([float(r["top1_score"]) for r in rr]), "margin": summarize_values([float(r["top1_score"] - r["top2_score"]) for r in rr]), "mean_top5": summarize_values([float(r["mean_top5_similarity"]) for r in rr])}
        no_map[model] = groups
    dump("no_map_diagnostics.json", {"status": "post_hoc_diagnostic_only", "groups": no_map}); write_csv("no_map_diagnostics.csv", [{"model": m, "group": g, "n": v["n"], "max_similarity_mean": v["max_similarity"].get("mean"), "margin_mean": v["margin"].get("mean"), "mean_top5_mean": v["mean_top5"].get("mean")} for m in MODELS for g, v in no_map[m].items()])
    for pop, direction, split in (("forward_stratified", "forward", "test"), ("forward_family_held_out", "forward", "test"), ("backward_stratified", "backward", "test"), ("backward_family_held_out", "backward", "test")):
        source_filter = (lambda r: r.get("source_family_split") == "test") if "family" in pop else (lambda r: True)
        out = []
        for m in MODELS:
            rr = [r for r in rows[(m, direction, split)] if source_filter(r)]; out.append({"population": pop, **summary(rr, m)})
        write_csv("family_held_out.csv" if "family" in pop else "backward_transfer.csv", out)
    family_rows = []
    for m in MODELS:
        rr = [r for r in rows[(m, "forward", "test")] if r.get("source_family_split") == "test"]
        family_rows.append({"population": "forward_family_held_out", **summary(rr, m)})
    dump("family_held_out.json", {"status": "computed", "rows": family_rows})
    dump("backward_transfer.json", {"status": "computed_from_direction_scoped_ledgers", "populations": {p: [summary([r for r in rows[(m, "backward", "test")] if ((r.get("source_family_split") == "test") if "family" in p else True)], m) for m in MODELS] for p in ("backward_stratified", "backward_family_held_out")}})
    for k in K:
        write_csv("candidate_k_tradeoff.csv", [{"k": k, "model": "l2_seed17", "ordinary_coverage": candidate_coverage([r for r in l2 if r.get("valid_target_codes") and r.get("mapping_kind") not in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}], k), "combination_complete": metric([r for r in l2 if r.get("mapping_kind") in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}], f"CompleteScenarioRetrieval@{k}")}])
    dump("candidate_k_analysis.json", {"k_values": list(K), "recommendation": 100, "reason": "maximizes ordinary coverage and combination completeness; finite reranking cost remains linear in K", "rows": [{"k": k, "coverage": candidate_coverage(l2, k)} for k in K]})
    dump("retrieval_error_floor.json", {"selected_k": 100, "ordinary_miss_rate": 1 - candidate_coverage([r for r in l2 if r.get("valid_target_codes") and r.get("mapping_kind") not in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}], 100), "combination_incomplete_rate": 1 - (metric([r for r in l2 if r.get("mapping_kind") in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}], "CompleteScenarioRetrieval@100") or 0)})
    if not (EXP / "representation_drift.json").exists(): dump("representation_drift.json", {"status": "requires_embedding_pass", "method": "same_tokenizer_pooling_normalization_target_order"})
    if not (EXP / "source_drift.json").exists(): dump("source_drift.json", {"status": "requires_embedding_pass"})
    if not (EXP / "runtime.json").exists(): dump("runtime.json", {"training_peak_memory": "NOT_AVAILABLE"})
    dump("retriever_freeze.json", {"status": "FROZEN", "candidate_generator": "SHIFT-MAP v1.3 corrected L2", "seed": 17, "epoch": 3, "candidate_k": 100, "evaluator_version": "2.0", "benchmark_version": "1.0", "test_exposure": True, "checkpoint_redistribution": "restricted_pending_license_review"})
    dump("scientific_position.json", {"S1": "SUPPORTED", "S2": "SUPPORTED", "S3": "INCONCLUSIVE_PENDING_CARDINALITY_PAIRED_LEDGER", "S4": "INCONCLUSIVE", "S5": "INCONCLUSIVE_PENDING_BACKWARD_COMPARISON", "S6": "INCONCLUSIVE", "S7": "INCONCLUSIVE_PENDING_EMBEDDING_PASS"})
    dump("v2_gate.json", {"gate": "GO-A", "candidate_generator": "frozen L2 seed17 epoch3", "step8_authorized": True, "caveat": "TEST-exposed post-hoc diagnostics; no clinical claims"})
    dump("manifest.json", {"experiment_version": "1.4", "generated_files": sorted(p.name for p in EXP.glob("*.json") if p.name != "manifest.json"), "ledger_files": sorted(p.name for p in LED.glob("*.json")), "training_occurred": False})
    print(f"diagnostic_artifacts={len(list(EXP.glob('*.json')))} tables={len(list(TAB.glob('*.csv')))}")

if __name__ == "__main__":
    main()
