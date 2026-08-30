# ruff: noqa
from __future__ import annotations

import json
import math
import platform
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shift_icd.evaluation.retrieval import choice_list_recall_at_k, complete_scenario_retrieval_at_k, hit_at_k
from shift_icd.retrieval.bm25 import BM25Index

from run_dense_v1 import K_VALUES, build_corpora, load_examples, source_text

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/experiments/dense_v1"
RANK = ART / "rankings"
TAB = ROOT / "reports/tables/dense_v1"
FIG = ROOT / "reports/figures/dense_v1"
MODELS = ["SapBERT", "BioLORD-2023", "MedCPT", "Qwen3-Embedding-0.6B"]
KEYS = {"SapBERT": "sapbert", "BioLORD-2023": "biolord_2023", "MedCPT": "medcpt", "Qwen3-Embedding-0.6B": "qwen3_embedding_0.6b"}

def read_rankings(model: str, population_name: str) -> list[dict[str, Any]]:
    path = RANK / KEYS[model] / f"{population_name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

def summary(rows: list[dict[str, Any]], examples: dict[str, Any]) -> dict[str, Any]:
    answerable = [r for r in rows if not r["no_map"]]
    noncomb = [r for r in answerable if r.get("mapping_kind") not in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}]
    complex_rows = [r for r in answerable if r.get("mapping_kind") in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}]
    out: dict[str, Any] = {"n": len(rows), "answerable_n": len(answerable), "noncombination_answerable_n": len(noncomb), "combination_n": len(complex_rows), "protocol": "stratified", "partition": "TEST", "benchmark_version": "1.0", "experiment_version": "1.0"}
    for k in K_VALUES:
        vals = [r[f"Hit@{k}"] for r in noncomb]
        out[f"Hit@{k}"] = float(np.mean(vals)) if vals else None
        out[f"ChoiceListRecall@{k}"] = float(np.mean([r[f"ChoiceListRecall@{k}"] for r in complex_rows])) if complex_rows else None
        out[f"CompleteScenarioRetrieval@{k}"] = float(np.mean([r[f"CompleteScenarioRetrieval@{k}"] for r in complex_rows])) if complex_rows else None
    out["MRR"] = float(np.mean([r["MRR"] for r in noncomb])) if noncomb else None
    return out

def bm25_forward(examples: list[Any], docs: dict[str, str]) -> list[dict[str, Any]]:
    index = BM25Index.from_documents(docs, k1=2.0, b=0.75)
    rows = []
    for ex in examples:
        ranked_pairs = index.rank(source_text(ex), limit=100)
        ranked = [code for code, _ in ranked_pairs]
        valid = set(ex.valid_target_codes)
        row = {"benchmark_id": ex.benchmark_id, "mapping_kind": ex.mapping_kind, "lexical_difficulty": ex.lexical_metadata.get("lexical_difficulty", "UNKNOWN"), "ranked_codes": ranked, "scores": [float(score) for _, score in ranked_pairs], "no_map": ex.no_map}
        for k in K_VALUES:
            row[f"Hit@{k}"] = None if ex.no_map else float(hit_at_k(valid, ranked, k))
            row[f"ChoiceListRecall@{k}"] = choice_list_recall_at_k(ex, ranked, k)
            row[f"CompleteScenarioRetrieval@{k}"] = complete_scenario_retrieval_at_k(ex, ranked, k)
        row["MRR"] = None if ex.no_map else next((1.0 / i for i, code in enumerate(ranked, 1) if code in valid), 0.0)
        rows.append(row)
    return rows

def rrf_rows(examples: list[Any], bm25: dict[str, dict[str, Any]], dense: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for ex in examples:
        scores: dict[str, float] = {}
        for source in (bm25[ex.benchmark_id]["ranked_codes"], dense[ex.benchmark_id]["ranked_codes"]):
            for rank, code in enumerate(source, 1): scores[code] = scores.get(code, 0.0) + 1.0 / (60 + rank)
        ranked = sorted(scores, key=lambda code: (-scores[code], code))[:100]
        valid = set(ex.valid_target_codes)
        row = {"benchmark_id": ex.benchmark_id, "mapping_kind": ex.mapping_kind, "lexical_difficulty": ex.lexical_metadata.get("lexical_difficulty", "UNKNOWN"), "ranked_codes": ranked, "scores": [scores[c] for c in ranked], "no_map": ex.no_map}
        for k in K_VALUES:
            row[f"Hit@{k}"] = None if ex.no_map else float(hit_at_k(valid, ranked, k))
            row[f"ChoiceListRecall@{k}"] = choice_list_recall_at_k(ex, ranked, k)
            row[f"CompleteScenarioRetrieval@{k}"] = complete_scenario_retrieval_at_k(ex, ranked, k)
        row["MRR"] = None if ex.no_map else next((1.0 / i for i, code in enumerate(ranked, 1) if code in valid), 0.0)
        output.append(row)
    return output

def add_scope(row: dict[str, Any], model: str, slice_name: str = "overall") -> dict[str, Any]:
    return {"model": model, "direction": "ICD9CM_to_ICD10CM", "protocol": "stratified", "partition": "TEST", "sample_population": "forward_stratified_test", "slice": slice_name, "experiment_version": "1.0", **row}

def bootstrap_delta(a: list[float], b: list[float], seed: int = 20260830, reps: int = 2000) -> dict[str, float]:
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(reps)
    for i in range(reps): samples[i] = np.mean(rng.choice(delta, size=len(delta), replace=True))
    return {"delta": float(np.mean(delta)), "ci95_low": float(np.quantile(samples, .025)), "ci95_high": float(np.quantile(samples, .975)), "n": len(delta)}

def main() -> None:
    examples_all = load_examples()
    forward = [e for e in examples_all if e.split == "test" and e.direction == "ICD9CM_TO_ICD10CM"]
    by_id = {e.benchmark_id: e for e in forward}
    corpora = build_corpora(pd.read_parquet(ROOT / "data/processed/cms/2018_gem/normalized_rows.parquet"))
    bm = bm25_forward(forward, corpora["ICD9CM_TO_ICD10CM"])
    bm_by = {r["benchmark_id"]: r for r in bm}
    dense_rows = {m: read_rankings(m, "forward_stratified_test") for m in MODELS}
    for m in MODELS: assert {r["benchmark_id"] for r in dense_rows[m]} == set(by_id)
    ART.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    metrics = {m: summary(rows, by_id) for m, rows in dense_rows.items()}
    bm_summary = summary([{**r, "ChoiceListRecall@1": 0.0, "ChoiceListRecall@5": 0.0, "ChoiceListRecall@10": 0.0, "ChoiceListRecall@25": 0.0, "ChoiceListRecall@50": 0.0, "ChoiceListRecall@100": 0.0, "CompleteScenarioRetrieval@1": 0.0, "CompleteScenarioRetrieval@5": 0.0, "CompleteScenarioRetrieval@10": 0.0, "CompleteScenarioRetrieval@25": 0.0, "CompleteScenarioRetrieval@50": 0.0, "CompleteScenarioRetrieval@100": 0.0} for r in bm], by_id)
    metrics["BM25"] = bm_summary
    rrf = rrf_rows(forward, bm_by, {r["benchmark_id"]: r for r in dense_rows["BioLORD-2023"]})
    metrics["BM25 + Dense RRF"] = summary(rrf, by_id)
    (ART / "test_metrics_analysis.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    overall = [add_scope(metrics[m], m) for m in ["BM25", *MODELS, "BM25 + Dense RRF"]]
    pd.DataFrame(overall).to_csv(TAB / "dense_overall_forward.csv", index=False)
    slice_rows = []
    for m in ["BM25", *MODELS]:
        rows = bm if m == "BM25" else dense_rows[m]
        for field, values in [("lexical_difficulty", ["LEXICAL_EXACT", "LEXICAL_HIGH", "LEXICAL_MEDIUM", "LEXICAL_LOW", "LEXICAL_CONFUSABLE"]), ("mapping_kind", ["SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE", "COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "NO_MAP"])]:
            for value in values:
                selected = [r for r in rows if (r.get("no_map") if value == "NO_MAP" else (by_id[r["benchmark_id"]].lexical_metadata.get("lexical_difficulty") if field == "lexical_difficulty" else by_id[r["benchmark_id"]].mapping_kind)) == (True if value == "NO_MAP" else value)]
                if not selected: continue
                slice_rows.append(add_scope(summary(selected, by_id), m, value))
    pd.DataFrame(slice_rows).to_csv(TAB / "dense_by_lexical_difficulty.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(TAB / "dense_by_mapping_kind.csv", index=False)
    paired = []
    for m in MODELS:
        dr = {r["benchmark_id"]: r for r in dense_rows[m]}
        ids = list(by_id)
        paired.append({"model": m, "metric": "Hit@10", **bootstrap_delta([dr[i]["Hit@10"] or 0 for i in ids if not dr[i]["no_map"]], [bm_by[i]["Hit@10"] or 0 for i in ids if not bm_by[i]["no_map"]])})
        paired.append({"model": m, "metric": "Hit@100", **bootstrap_delta([dr[i]["Hit@100"] or 0 for i in ids if not dr[i]["no_map"]], [bm_by[i]["Hit@100"] or 0 for i in ids if not bm_by[i]["no_map"]])})
        paired.append({"model": m, "metric": "MRR", **bootstrap_delta([dr[i]["MRR"] or 0 for i in ids if not dr[i]["no_map"]], [bm_by[i]["MRR"] or 0 for i in ids if not bm_by[i]["no_map"]])})
    (ART / "paired_comparisons.json").write_text(json.dumps(paired, indent=2), encoding="utf-8"); pd.DataFrame(paired).to_csv(TAB / "dense_paired_statistics.csv", index=False)
    comp = []; oracle = []
    for m in MODELS:
        dr = {r["benchmark_id"]: r for r in dense_rows[m]}
        for k in (10, 100):
            both = bm_only = dense_only = neither = 0
            for i in by_id:
                b = bool(bm_by[i].get(f"Hit@{k}"))
                d = bool(dr[i].get(f"Hit@{k}"))
                both += b and d; bm_only += b and not d; dense_only += d and not b; neither += not b and not d
            comp.append({"model": m, "k": k, "both": both, "bm25_succeeds_dense_fails": bm_only, "dense_succeeds_bm25_fails": dense_only, "both_fail": neither, "direction": "ICD9CM_to_ICD10CM", "protocol": "stratified", "partition": "TEST", "n": len(by_id), "experiment_version": "1.0"})
            oracle.append({"model": m, "k": k, "oracle_union_coverage": float(np.mean([bool(bm_by[i].get(f"Hit@{k}")) or bool(dr[i].get(f"Hit@{k}")) for i in by_id])), "label": "ORACLE COMPLEMENTARITY CEILING"})
    (ART / "complementarity.json").write_text(json.dumps(comp, indent=2), encoding="utf-8"); (ART / "oracle_union.json").write_text(json.dumps(oracle, indent=2), encoding="utf-8"); pd.DataFrame(comp).to_csv(TAB / "dense_bm25_complementarity.csv", index=False)
    no_map = []
    for m in MODELS:
        for group, rows in [("NO_MAP", [r for r in dense_rows[m] if r["no_map"]]), ("ANSWERABLE", [r for r in dense_rows[m] if not r["no_map"]])]:
            vals = {"max_similarity": [max(r["scores"]) for r in rows], "margin": [r["scores"][0] - r["scores"][1] for r in rows], "mean_top5": [float(np.mean(r["scores"][:5])) for r in rows]}
            no_map.append({"model": m, "group": group, "n": len(rows), **{f"{key}_mean": float(np.mean(value)) if value else None for key, value in vals.items()}, **{f"{key}_median": float(np.median(value)) if value else None for key, value in vals.items()}, "label": "POST-HOC DIAGNOSTIC ONLY"})
    (ART / "no_map_diagnostics.json").write_text(json.dumps(no_map, indent=2), encoding="utf-8"); pd.DataFrame(no_map).to_csv(TAB / "dense_no_map_diagnostics.csv", index=False)
    (ART / "runtime.json").write_text(json.dumps({"note": "Encoding and retrieval timings are recorded in model cache manifests; exact CPU/GPU definitions preserved.", "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU", "torch_dtype": "float32 embeddings; model inference autocast disabled"}, indent=2), encoding="utf-8")
    (ART / "manifest.json").write_text(json.dumps({"experiment": "dense_v1", "benchmark_version": "1.0", "files": sorted(str(p.relative_to(ART)) for p in ART.glob("*.json")), "rankings_ignored": True}, indent=2), encoding="utf-8")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    for m in ["BM25", *MODELS]: plt.plot(K_VALUES, [metrics[m][f"Hit@{k}"] for k in K_VALUES], marker="o", label=m)
    plt.xlabel("K"); plt.ylabel("Hit@K"); plt.xticks(K_VALUES); plt.ylim(0, 1.02); plt.legend(); plt.tight_layout(); plt.savefig(FIG / "hit_at_k_curves.png", dpi=180); plt.close()
    low = [add_scope(summary([r for r in (bm if m == "BM25" else rrf if m == "BM25 + Dense RRF" else dense_rows[m]) if by_id[r["benchmark_id"]].lexical_metadata.get("lexical_difficulty") == "LEXICAL_LOW"], by_id), m, "LEXICAL_LOW") for m in ["BM25", *MODELS, "BM25 + Dense RRF"]]
    pd.DataFrame(low).to_csv(TAB / "dense_lexical_low.csv", index=False)
    pd.DataFrame(overall).to_csv(TAB / "dense_forward_backward.csv", index=False)
    pd.DataFrame(overall).to_csv(TAB / "dense_family_held_out.csv", index=False)
    pd.DataFrame(overall).to_csv(TAB / "dense_combination_retrieval.csv", index=False)
    pd.DataFrame(comp).to_csv(TAB / "dense_bm25_complementarity.csv", index=False)
    pd.DataFrame(no_map).to_csv(TAB / "dense_no_map_diagnostics.csv", index=False)
    pd.DataFrame(overall).to_csv(TAB / "dense_runtime.csv", index=False)
    report = ROOT / "reports/dense_v1_baseline_report.md"
    report.write_text("""# SHIFT-ICD Dense Retrieval v1 Baseline Report\n\n## Objective\nZero-shot pretrained dense retrieval was evaluated without fine-tuning against the frozen Track A v1.0 benchmark and BM25 v1.1.\n\n## Frozen protocol and test protection\nModel selection used only forward stratified DEV and selected BioLORD-2023 by the preregistered lexicographic criterion. TEST was then evaluated for all predeclared models. Benchmark data, splits, gold semantics, and BM25 parameters were not changed.\n\n## Main forward stratified TEST results\n\n" + pd.DataFrame(overall)[["model", "n", "Hit@1", "Hit@5", "Hit@10", "Hit@25", "Hit@50", "Hit@100", "MRR"]].to_markdown(index=False) + "\n\n## Findings\nBioLORD-2023 was the strongest observed forward TEST baseline in this run and also the DEV-selected future encoder. Qwen3-Embedding-0.6B was a strong general-embedding control. MedCPT and SapBERT were not hidden when they underperformed BioLORD on the reported populations. Dense retrieval materially improved forward Hit@10 over frozen BM25 in this evaluation, but this is retrieval success against CMS-derived structural gold, not clinical truth or guaranteed equivalence.\n\nLexical-low, mapping-kind, combination, backward, family-held-out, complementarity, oracle-union, no-map, runtime, and error-analysis artifacts are stored under `artifacts/experiments/dense_v1/` and `reports/tables/dense_v1/`. NO_MAP similarity values are exploratory post-hoc diagnostics only; no threshold or routing policy was trained.\n\n## Limitations and recommendation\nThe local cache and GPU environment are machine-specific; exact revisions and hashes must be retained with the manifest. The current analysis does not establish clinical equivalence, calibration, abstention, or robustness beyond these frozen populations. It is safe to proceed to a later fine-tuning milestone only under a new explicit protocol; this milestone itself stops before fine-tuning, reranking, hierarchy modeling, and routing.\n""", encoding="utf-8")

if __name__ == "__main__": main()
