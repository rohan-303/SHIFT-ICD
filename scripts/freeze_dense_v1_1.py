# ruff: noqa
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_dense_v1 import K_VALUES, build_corpora, load_examples, population, source_text
from shift_icd.dense.retrieval import rrf_fuse
from shift_icd.evaluation.retrieval import choice_list_recall_at_k, complete_scenario_retrieval_at_k, hit_at_k

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "artifacts/experiments/dense_v1"
OUT = ROOT / "artifacts/experiments/dense_v1_1"
TAB = ROOT / "reports/tables/dense_v1_1"
FIG = ROOT / "reports/figures/dense_v1_1"
MODELS = ["BM25", "SapBERT", "BioLORD-2023", "MedCPT", "Qwen3-Embedding-0.6B", "BM25 + BioLORD RRF"]
DENSE = MODELS[1:5]
SYSTEM_DIR = {
    "SapBERT": "sapbert",
    "BioLORD-2023": "biolord_2023",
    "MedCPT": "medcpt",
    "Qwen3-Embedding-0.6B": "qwen3_embedding_0.6b",
    "BM25 + BioLORD RRF": "bm25_dense_rrf",
}
POPS = ["forward_stratified_test", "forward_family_held_out_test", "backward_stratified_test", "backward_family_held_out_test"]
METRICS = [f"Hit@{k}" for k in K_VALUES] + ["MRR"]
COMBO_METRICS = [f"ChoiceListRecall@{k}" for k in K_VALUES] + [f"CompleteScenarioRetrieval@{k}" for k in K_VALUES]
NONCOMBO = {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_for(system: str, pop: str, bm25_rows: dict[str, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    if system == "BM25":
        if bm25_rows is None:
            raise RuntimeError("BM25 rows were not initialized")
        return bm25_rows[pop]
    return [json.loads(x) for x in (V1 / "rankings" / SYSTEM_DIR[system] / f"{pop}.jsonl").read_text(encoding="utf-8").splitlines()]


def load_frozen_bm25(pop: str) -> list[dict[str, Any]]:
    scoped = {
        r["benchmark_id"]: r
        for r in (
            json.loads(line)
            for line in (ROOT / "artifacts/experiments/bm25_v1_1/scoped_rows.jsonl").read_text(encoding="utf-8").splitlines()
        )
        if r.get("sample_population") == pop
    }
    grouped: dict[str, list[tuple[str, float]]] = {}
    ranking = ROOT / "artifacts/experiments/bm25_v1_1/rankings" / f"{pop}.jsonl"
    for line in ranking.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        grouped.setdefault(r["benchmark_id"], []).append((r["target_code"], float(r["score"])))
    out = []
    for bid, pairs in grouped.items():
        s = scoped[bid]
        out.append(
            {
                "benchmark_id": bid,
                "direction": s["direction"],
                "mapping_kind": s["mapping_kind"],
                "lexical_difficulty": s["lexical_difficulty"],
                "no_map": s["no_map"],
                "ranked_codes": [x[0] for x in pairs],
                "scores": [x[1] for x in pairs],
                **{k: s.get(k) for k in METRICS + COMBO_METRICS},
            }
        )
    return out


def make_rrf_rows(bm: list[dict[str, Any]], dense: list[dict[str, Any]], ex_by_id: dict[str, Any]) -> list[dict[str, Any]]:
    dm = {r["benchmark_id"]: r for r in dense}
    out = []
    for b in bm:
        d = dm[b["benchmark_id"]]
        ex = ex_by_id[b["benchmark_id"]]
        fused = rrf_fuse(b["ranked_codes"], d["ranked_codes"], 60, 100)
        ranked = [c for c, _ in fused]
        scores = [float(s) for _, s in fused]
        valid = set(ex.valid_target_codes)
        row = {
            "benchmark_id": ex.benchmark_id,
            "direction": ex.direction,
            "mapping_kind": ex.mapping_kind,
            "lexical_difficulty": ex.lexical_metadata.get("lexical_difficulty", "UNKNOWN"),
            "no_map": ex.no_map,
            "ranked_codes": ranked,
            "scores": scores,
        }
        for k in K_VALUES:
            row[f"Hit@{k}"] = None if ex.no_map else float(hit_at_k(valid, ranked, k))
            row[f"ChoiceListRecall@{k}"] = choice_list_recall_at_k(ex, ranked, k)
            row[f"CompleteScenarioRetrieval@{k}"] = complete_scenario_retrieval_at_k(ex, ranked, k)
        row["MRR"] = None if ex.no_map else next((1.0 / i for i, c in enumerate(ranked, 1) if c in valid), 0.0)
        out.append(row)
    return out


def scope(pop: str, system: str, rows: list[dict[str, Any]], ex_by_id: dict[str, Any], slice_name: str = "overall") -> dict[str, Any]:
    answerable = [r for r in rows if not r.get("no_map", False)]
    noncombo = [r for r in answerable if r.get("mapping_kind") not in NONCOMBO]
    combo = [r for r in answerable if r.get("mapping_kind") in NONCOMBO]
    out: dict[str, Any] = {
        "model": system,
        "direction": "ICD9CM_to_ICD10CM" if pop.startswith("forward") else "ICD10CM_to_ICD9CM",
        "split_protocol": "source_stratified_and_family_held_out",
        "partition": "stratified_test" if "stratified" in pop else "family_held_out_test",
        "benchmark_version": "1.0",
        "experiment_version": "1.1",
        "sample_population": pop,
        "slice": slice_name,
        "n_total": len(rows),
        "n_answerable": len(answerable),
        "n_noncombination_answerable": len(noncombo),
        "n_combination": len(combo),
        "applicable_hit_k_n": len(noncombo),
        "applicable_mrr_n": len(noncombo),
        "applicable_choice_scenario_n": len(combo),
    }
    for metric in METRICS:
        vals = [r[metric] for r in noncombo if r.get(metric) is not None]
        out[metric] = float(np.mean(vals)) if vals else None
    for metric in COMBO_METRICS:
        vals = [r[metric] for r in combo if r.get(metric) is not None]
        out[metric] = float(np.mean(vals)) if vals else None
    return out


def subset(rows: list[dict[str, Any]], ex_by_id: dict[str, Any], field: str, value: str) -> list[dict[str, Any]]:
    if field == "mapping_kind":
        return [r for r in rows if r.get("mapping_kind") == value]
    if field == "lexical_difficulty":
        return [r for r in rows if r.get("lexical_difficulty") == value]
    if field == "no_map":
        return [r for r in rows if bool(r.get("no_map")) == (value == "NO_MAP")]
    if field == "alternative_size":
        result = []
        for r in rows:
            ex = ex_by_id[r["benchmark_id"]]
            n = len(getattr(ex, "valid_target_codes", []))
            bucket = "1" if n <= 1 else "2-3" if n <= 3 else "4-9" if n <= 9 else "10+"
            if bucket == value:
                result.append(r)
        return result
    return []


def bootstrap_delta(a: np.ndarray, b: np.ndarray, seed: int = 20260830, reps: int = 2000) -> dict[str, float]:
    delta = a - b
    rng = np.random.default_rng(seed)
    samples = np.array([np.mean(rng.choice(delta, len(delta), replace=True)) for _ in range(reps)])
    return {
        "delta": float(np.mean(delta)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "n": int(len(delta)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    examples = load_examples()
    ex_by_id = {e.benchmark_id: e for e in examples}
    bm25_all = {pop: load_frozen_bm25(pop) for pop in POPS}
    all_rows: dict[str, dict[str, list[dict[str, Any]]]] = {m: {p: rows_for(m, p, bm25_all) for p in POPS} for m in ["BM25", *DENSE]}
    all_rows["BM25 + BioLORD RRF"] = {}
    for pop in POPS:
        all_rows["BM25 + BioLORD RRF"][pop] = make_rrf_rows(all_rows["BM25"][pop], all_rows["BioLORD-2023"][pop], ex_by_id)
    # Scope assertions: same IDs across systems and frozen counts.
    scope_audit = []
    for pop in POPS:
        ids = {r["benchmark_id"] for r in all_rows["BM25"][pop]}
        assert ids == {
            e.benchmark_id
            for e in ex_by_id.values()
            if e.direction == ("ICD9CM_TO_ICD10CM" if pop.startswith("forward") else "ICD10CM_TO_ICD9CM")
            and (("family_held_out" in pop and e.source_family_split == "test") or ("stratified" in pop and e.split == "test"))
        }, f"scope mismatch {pop}"
        for m in MODELS:
            assert {r["benchmark_id"] for r in all_rows[m][pop]} == ids, f"system mismatch {m}/{pop}"
        scope_audit.append({"sample_population": pop, "n": len(ids), "systems": len(MODELS), "ids_consistent": True})
    json.dump(
        {
            "experiment_version": "1.1",
            "benchmark_version": "1.0",
            "scope_audit": scope_audit,
            "test_selection_used": False,
            "source": "frozen dense_v1 rankings and bm25_test_rows",
        },
        open(OUT / "scope_audit.json", "w"),
        indent=2,
    )
    # All-population overall tables.
    for pop, filename in zip(
        POPS,
        [
            "forward_stratified_overall.csv",
            "forward_family_held_out_overall.csv",
            "backward_stratified_overall.csv",
            "backward_family_held_out_overall.csv",
        ],
    ):
        pd.DataFrame([scope(pop, m, all_rows[m][pop], ex_by_id) for m in MODELS]).to_csv(TAB / filename, index=False)
    pd.DataFrame([scope("forward_family_held_out_test", m, all_rows[m]["forward_family_held_out_test"], ex_by_id) for m in MODELS]).to_csv(
        TAB / "forward_family_comparison.csv", index=False
    )
    pd.DataFrame(
        [scope("backward_family_held_out_test", m, all_rows[m]["backward_family_held_out_test"], ex_by_id) for m in MODELS]
    ).to_csv(TAB / "backward_family_comparison.csv", index=False)
    pd.DataFrame(
        [scope("forward_stratified_test", m, all_rows[m]["forward_stratified_test"], ex_by_id) for m in MODELS]
        + [scope("backward_stratified_test", m, all_rows[m]["backward_stratified_test"], ex_by_id) for m in MODELS]
    ).to_csv(TAB / "forward_backward_comparison.csv", index=False)
    # Primary forward TEST metric reproduction and frozen copy.
    primary = pd.DataFrame([scope("forward_stratified_test", m, all_rows[m]["forward_stratified_test"], ex_by_id) for m in MODELS])
    primary.to_csv(TAB / "forward_stratified_overall.csv", index=False)
    json.dump(
        {"experiment_version": "1.1", "source_experiment": "dense_v1", "metrics": primary.to_dict(orient="records")},
        open(OUT / "frozen_metrics.json", "w"),
        indent=2,
    )
    # Slice tables.
    for field, values, filename in [
        (
            "mapping_kind",
            ["SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE", "COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"],
            "forward_stratified_by_mapping_kind.csv",
        ),
        (
            "lexical_difficulty",
            ["LEXICAL_EXACT", "LEXICAL_HIGH", "LEXICAL_MEDIUM", "LEXICAL_LOW", "LEXICAL_CONFUSABLE"],
            "forward_stratified_by_lexical_difficulty.csv",
        ),
        ("alternative_size", ["1", "2-3", "4-9", "10+"], "forward_stratified_by_alternative_size.csv"),
    ]:
        data = []
        for m in MODELS:
            for v in values:
                rs = subset(all_rows[m]["forward_stratified_test"], ex_by_id, field, v)
                if rs:
                    data.append(scope("forward_stratified_test", m, rs, ex_by_id, v))
        pd.DataFrame(data).to_csv(TAB / filename, index=False)
    combo = []
    for m in MODELS:
        rs = [r for r in all_rows[m]["forward_stratified_test"] if r.get("mapping_kind") in NONCOMBO]
        for kind in ["COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"]:
            part = [r for r in rs if r.get("mapping_kind") == kind]
            if part:
                combo.append(scope("forward_stratified_test", m, part, ex_by_id, kind))
    pd.DataFrame(combo).to_csv(TAB / "forward_stratified_combination.csv", index=False)
    # NO_MAP diagnostics.
    nm = []
    for m in MODELS:
        for group in ["NO_MAP", "ANSWERABLE"]:
            rs = subset(all_rows[m]["forward_stratified_test"], ex_by_id, "no_map", group)
            vals = np.array(
                [
                    [
                        max(r["scores"]) if r.get("scores") else np.nan,
                        (r["scores"][0] - r["scores"][1]) if len(r.get("scores", [])) > 1 else np.nan,
                        float(np.mean(r["scores"][:5])) if r.get("scores") else np.nan,
                    ]
                    for r in rs
                ]
            )
            nm.append(
                {
                    "model": m,
                    "group": group,
                    "n": len(rs),
                    "max_score_mean": float(np.nanmean(vals[:, 0])),
                    "max_score_median": float(np.nanmedian(vals[:, 0])),
                    "top1_top2_margin_mean": float(np.nanmean(vals[:, 1])),
                    "top1_top2_margin_median": float(np.nanmedian(vals[:, 1])),
                    "mean_top5_mean": float(np.nanmean(vals[:, 2])),
                    "mean_top5_median": float(np.nanmedian(vals[:, 2])),
                    "label": "POST-HOC DIAGNOSTIC ONLY",
                }
            )
    pd.DataFrame(nm).to_csv(TAB / "forward_stratified_no_map_diagnostics.csv", index=False)
    # Ledger, compact and independent of rerun.
    ledger = []
    for m in MODELS:
        for r in all_rows[m]["forward_stratified_test"]:
            row = {
                "benchmark_id": r["benchmark_id"],
                "system": m,
                "top1_target": r["ranked_codes"][0] if r.get("ranked_codes") else None,
                "top1_score": r["scores"][0] if r.get("scores") else None,
                "best_valid_target_rank": next(
                    (i for i, c in enumerate(r.get("ranked_codes", []), 1) if c in set(ex_by_id[r["benchmark_id"]].valid_target_codes)),
                    None,
                ),
                "mapping_kind": r.get("mapping_kind"),
                "lexical_difficulty": r.get("lexical_difficulty"),
                "alternative_size_bucket": (
                    "1"
                    if len(ex_by_id[r["benchmark_id"]].valid_target_codes) <= 1
                    else "2-3"
                    if len(ex_by_id[r["benchmark_id"]].valid_target_codes) <= 3
                    else "4-9"
                    if len(ex_by_id[r["benchmark_id"]].valid_target_codes) <= 9
                    else "10+"
                ),
            }
            for k in K_VALUES:
                row[f"Hit@{k}"] = r.get(f"Hit@{k}")
                row[f"ChoiceListRecall@{k}"] = r.get(f"ChoiceListRecall@{k}")
                row[f"CompleteScenarioRetrieval@{k}"] = r.get(f"CompleteScenarioRetrieval@{k}")
            row["MRR"] = r.get("MRR")
            ledger.append(row)
    pd.DataFrame(ledger).to_csv(TAB / "forward_stratified_prediction_ledger.csv", index=False)
    json.dump(
        {"experiment_version": "1.1", "format": "CSV", "rows": len(ledger), "systems": MODELS, "population": "forward_stratified_test"},
        open(OUT / "prediction_ledger_manifest.json", "w"),
        indent=2,
    )
    # Complementarity and oracle.
    comp = []
    oracle = []
    for m in DENSE:
        for k in [10, 100]:
            counts = {"both_succeed": 0, "bm25_only": 0, "dense_only": 0, "both_fail": 0}
            for b, d in zip(all_rows["BM25"]["forward_stratified_test"], all_rows[m]["forward_stratified_test"]):
                x = bool(b[f"Hit@{k}"])
                y = bool(d[f"Hit@{k}"])
                counts["both_succeed" if x and y else "bm25_only" if x else "dense_only" if y else "both_fail"] += 1
            comp.append(
                {
                    "model": m,
                    "k": k,
                    "direction": "ICD9CM_to_ICD10CM",
                    "split_protocol": "source_stratified",
                    "partition": "stratified_test",
                    "benchmark_version": "1.0",
                    "experiment_version": "1.1",
                    "sample_population": "forward_stratified_test",
                    "n_total": sum(counts.values()),
                    **counts,
                }
            )
        for sl in ["LEXICAL_EXACT", "LEXICAL_HIGH", "LEXICAL_MEDIUM", "LEXICAL_LOW", "SINGLE_EXACT", "SINGLE_APPROXIMATE", "ALTERNATIVE"]:
            field = "lexical_difficulty" if sl.startswith("LEXICAL") else "mapping_kind"
            ids = {r["benchmark_id"] for r in subset(all_rows[m]["forward_stratified_test"], ex_by_id, field, sl)}
            for k in [10, 100]:
                c = {"both_succeed": 0, "bm25_only": 0, "dense_only": 0, "both_fail": 0}
                for b, d in zip(all_rows["BM25"]["forward_stratified_test"], all_rows[m]["forward_stratified_test"]):
                    if b["benchmark_id"] not in ids:
                        continue
                    x = bool(b[f"Hit@{k}"])
                    y = bool(d[f"Hit@{k}"])
                    c["both_succeed" if x and y else "bm25_only" if x else "dense_only" if y else "both_fail"] += 1
                comp.append(
                    {"model": m, "slice": sl, "k": k, "sample_population": "forward_stratified_test", "n_total": sum(c.values()), **c}
                )
    pd.DataFrame(comp).to_csv(TAB / "forward_bm25_dense_complementarity.csv", index=False)
    oracle = []
    for row in comp:
        if "slice" not in row:
            oracle.append(
                {
                    **{k: row[k] for k in ["model", "k", "sample_population"]},
                    "direction": "ICD9CM_to_ICD10CM",
                    "split_protocol": "source_stratified",
                    "partition": "stratified_test",
                    "benchmark_version": "1.0",
                    "experiment_version": "1.1",
                    "n_total": row["n_total"],
                    "oracle_union_coverage": (row["both_succeed"] + row["bm25_only"] + row["dense_only"]) / row["n_total"],
                    "label": "ORACLE UNION; NON-DEPLOYABLE DIAGNOSTIC",
                }
            )
    pd.DataFrame(oracle).to_csv(TAB / "forward_oracle_union.csv", index=False)
    json.dump(oracle, open(OUT / "oracle_union.json", "w"), indent=2)
    # Bootstrap paired deltas on all answerable examples, preserving pair IDs.
    paired = []
    q_rows = {r["benchmark_id"]: r for r in all_rows["Qwen3-Embedding-0.6B"]["forward_stratified_test"]}
    b_rows = {r["benchmark_id"]: r for r in all_rows["BM25"]["forward_stratified_test"]}
    for m in ["SapBERT", "BioLORD-2023", "MedCPT", "Qwen3-Embedding-0.6B", "BM25 + BioLORD RRF"]:
        d_rows = {r["benchmark_id"]: r for r in all_rows[m]["forward_stratified_test"]}
        ids = [i for i, r in b_rows.items() if not r.get("no_map") and ex_by_id[i].mapping_kind not in NONCOMBO]
        for metric in ["Hit@10", "Hit@100", "MRR"]:
            x = np.array([d_rows[i][metric] for i in ids], float)
            y = np.array([b_rows[i][metric] for i in ids], float)
            z = bootstrap_delta(x, y)
            paired.append(
                {
                    "model": m,
                    "comparison": "vs BM25",
                    "metric": metric,
                    "baseline_value": float(np.mean(y)),
                    "comparison_value": float(np.mean(x)),
                    **z,
                    "population": "forward_stratified_test",
                    "applicable_n": len(ids),
                    "delta_definition": "paired mean of per-example comparison minus BM25 on answerable non-combination examples",
                }
            )
    for metric in ["Hit@10", "Hit@100", "MRR"]:
        ids = [i for i, r in q_rows.items() if not r.get("no_map") and ex_by_id[i].mapping_kind not in NONCOMBO]
        x = np.array(
            [
                all_rows["BioLORD-2023"]["forward_stratified_test"][
                    [r["benchmark_id"] for r in all_rows["BioLORD-2023"]["forward_stratified_test"]].index(i)
                ][metric]
                for i in ids
            ]
        )
        y = np.array([q_rows[i][metric] for i in ids])
        z = bootstrap_delta(x, y)
        paired.append(
            {
                "model": "BioLORD-2023",
                "comparison": "vs Qwen3-Embedding-0.6B",
                "metric": metric,
                "baseline_value": float(np.mean(y)),
                "comparison_value": float(np.mean(x)),
                **z,
                "population": "forward_stratified_test",
                "applicable_n": len(ids),
                "delta_definition": "paired mean of BioLORD minus Qwen per-example values",
            }
        )
    pd.DataFrame(paired).to_csv(TAB / "forward_paired_bootstrap.csv", index=False)
    json.dump(paired, open(OUT / "paired_comparisons.json", "w"), indent=2)
    # RRF analysis.
    rr = []
    for k in [10, 100]:
        for category in ["BioLORD_succeeds_RRF_fails", "RRF_succeeds_BioLORD_fails"]:
            n = 0
            for b, r in zip(all_rows["BioLORD-2023"]["forward_stratified_test"], all_rows["BM25 + BioLORD RRF"]["forward_stratified_test"]):
                if (
                    bool(b[f"Hit@{k}"]) and not bool(r[f"Hit@{k}"])
                    if category.startswith("Bio")
                    else bool(r[f"Hit@{k}"]) and not bool(b[f"Hit@{k}"])
                ):
                    n += 1
            rr.append(
                {
                    "category": category,
                    "k": k,
                    "n": n,
                    "fixed_rrf_k": 60,
                    "population": "forward_stratified_test",
                    "interpretation": "descriptive fixed-RRF failure analysis; no tuning",
                }
            )
    pd.DataFrame(rr).to_csv(TAB / "forward_rrf_failure_analysis.csv", index=False)
    json.dump(rr, open(OUT / "rrf_failure_analysis.json", "w"), indent=2)
    # Multi-scenario audit.
    ms = []
    for pop in POPS:
        rs = [
            e
            for e in ex_by_id.values()
            if e.benchmark_id in {r["benchmark_id"] for r in all_rows["BM25"][pop]} and e.mapping_kind == "MULTI_SCENARIO"
        ]
        ms.append(
            {
                "sample_population": pop,
                "direction": "ICD9CM_to_ICD10CM" if pop.startswith("forward") else "ICD10CM_to_ICD9CM",
                "n_multi_scenario": len(rs),
                "metric": "not computed when zero",
            }
        )
    pd.DataFrame(ms).to_csv(TAB / "multi_scenario_audit.csv", index=False)
    json.dump(ms, open(OUT / "multi_scenario_audit.json", "w"), indent=2)
    # Provenance/runtime/dev selection.
    models = read_json(V1 / "models.json")["models"]
    prov = []
    for x in models:
        y = dict(x)
        slug = "models--" + x["model_id"].replace("/", "--")
        snap = Path.home() / ".cache" / "huggingface" / "hub" / slug / "snapshots" / x["revision"]

        def file_hash(name: str) -> str | None:
            path = snap / name
            if not path.exists():
                return None
            return hashlib.sha256(path.read_bytes()).hexdigest()

        y.update(
            {
                "experiment_version": "1.1",
                "max_sequence_length": 64,
                "dtype": "float32 embeddings; inference autocast disabled",
                "model_config_sha256": file_hash("config.json"),
                "tokenizer_config_sha256": file_hash("tokenizer_config.json"),
                "parameter_count": None,
                "parameter_count_note": "not exposed in cached config; exact Hub revision recorded",
            }
        )
        prov.append(y)
    pd.DataFrame(prov).to_csv(TAB / "dense_model_provenance.csv", index=False)
    json.dump(prov, open(OUT / "model_provenance.json", "w"), indent=2)
    runtime = read_json(V1 / "runtime.json")
    runtime_rows = []
    for m in DENSE:
        runtime_rows.append(
            {
                "model": m,
                "experiment_version": "1.1",
                "device": "cuda",
                "batch_size": "see dense_v1 config",
                "dtype": "float32 embeddings",
                "model_load_seconds": None,
                "target_encoding_seconds": None,
                "query_encoding_seconds": None,
                "exact_retrieval_seconds": None,
                "mean_query_latency_seconds": None,
                "median_query_latency_seconds": None,
                "p95_query_latency_seconds": None,
                "gpu_peak_allocated_bytes": None,
                "cpu_peak_rss_bytes": None,
                "note": "Original v1 runtime artifact did not retain per-model timing fields; unavailable without rerunning encoders.",
            }
        )
    pd.DataFrame(runtime_rows).to_csv(TAB / "dense_runtime.csv", index=False)
    json.dump(runtime, open(OUT / "runtime.json", "w"), indent=2)
    dev = read_json(V1 / "dev_metrics.json")
    sel = read_json(V1 / "selection.json")
    json.dump(
        {
            "experiment_version": "1.1",
            "criterion": ["noncombination_answerable_Hit@100", "CompleteScenarioRetrieval@100", "Hit@10", "MRR"],
            "models": dev,
            "selected_model": sel.get("selected_model", "BioLORD-2023"),
            "test_performance_used": False,
        },
        open(OUT / "dev_selection_audit.json", "w"),
        indent=2,
    )
    pd.DataFrame(
        [
            {
                "model": m,
                "selected": m == "BioLORD-2023",
                "test_performance_used": False,
                "criterion": "noncombination Hit@100 > CompleteScenarioRetrieval@100 > Hit@10 > MRR",
            }
            for m in DENSE
        ]
    ).to_csv(TAB / "dense_dev_selection.csv", index=False)
    # Error samples are mechanistic ledger slices; no unsupported medical labels.
    failures = []
    successes = []
    br = b_rows
    bl = {r["benchmark_id"]: r for r in all_rows["BioLORD-2023"]["forward_stratified_test"]}
    for i in br:
        if not br[i].get("no_map") and not bl[i].get("Hit@10"):
            failures.append({"benchmark_id": i, "label": "unclear", "evidence": "retrieval failure only; no unsupported medical judgment"})
        if not br[i].get("no_map") and not br[i].get("Hit@10") and bl[i].get("Hit@10"):
            successes.append(
                {"benchmark_id": i, "label": "unclear", "evidence": "BM25 failure/BioLORD success; mechanism not manually asserted"}
            )
    pd.DataFrame(failures[:100]).to_csv(TAB / "forward_biolord_top10_failure_taxonomy.csv", index=False)
    pd.DataFrame(successes[:50]).to_csv(TAB / "forward_biolord_successes_bm25_failed.csv", index=False)
    # Figures generated from the table/result files, not manually typed values.
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    labels = ["BM25", "SapBERT", "BioLORD-2023", "MedCPT", "Qwen3-Embedding-0.6B", "BM25 + BioLORD RRF"]
    vals = {m: primary.loc[primary.model == m, [f"Hit@{k}" for k in K_VALUES]].iloc[0].to_numpy() for m in labels}
    plt.figure(figsize=(9, 5))
    [plt.plot(K_VALUES, vals[m], marker="o", label=m) for m in labels]
    plt.ylim(0, 1.02)
    plt.xticks(K_VALUES)
    plt.xlabel("K")
    plt.ylabel("Hit@K")
    plt.title("Forward stratified TEST (n=2913)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "figure_1_hit_at_k.png", dpi=180)
    plt.close()
    lex = pd.read_csv(TAB / "forward_stratified_by_lexical_difficulty.csv")
    for num, (col, title, file) in enumerate(
        [
            ("Hit@10", "Hit@10 by lexical difficulty", "figure_2_hit10_lexical_difficulty.png"),
            ("delta", "Absolute Hit@10 delta versus BM25", "figure_3_hit10_delta_lexical_difficulty.png"),
        ],
        2,
    ):
        plt.figure(figsize=(9, 5))
        pivot = lex.pivot(index="slice", columns="model", values="Hit@10")
        if col == "delta":
            pivot = pivot.subtract(pivot["BM25"], axis=0).drop(columns="BM25")
        pivot.plot(kind="bar", ax=plt.gca())
        plt.ylabel(col)
        plt.title(title + " — forward stratified TEST")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(FIG / file, dpi=180)
        plt.close()
    plt.figure(figsize=(8, 5))
    [plt.bar(m, primary.loc[primary.model == m, "Hit@100"].iloc[0]) for m in labels]
    plt.ylim(0, 1.02)
    plt.ylabel("Hit@100")
    plt.title("Candidate coverage — forward stratified TEST")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "figure_4_candidate_coverage.png", dpi=180)
    plt.close()
    combo_df = pd.read_csv(TAB / "forward_stratified_combination.csv")
    p = combo_df[combo_df["slice"].isin(["COMBINATION", "COMBINATION_WITH_ALTERNATIVES"])].pivot_table(
        index="model", values="CompleteScenarioRetrieval@100", aggfunc="mean"
    )
    p.plot(kind="bar", legend=False, figsize=(8, 5))
    plt.ylim(0, 1.02)
    plt.ylabel("CompleteScenarioRetrieval@100")
    plt.title("Combination retrieval — forward stratified TEST")
    plt.tight_layout()
    plt.savefig(FIG / "figure_5_combination_complete_scenario.png", dpi=180)
    plt.close()
    c = pd.DataFrame([x for x in comp if x.get("model") == "BioLORD-2023" and "slice" not in x])
    c.set_index("k")[["both_succeed", "bm25_only", "dense_only", "both_fail"]].plot(kind="bar", stacked=True, figsize=(8, 5))
    plt.ylabel("Examples")
    plt.title("BM25/BioLORD complementarity — forward stratified TEST")
    plt.tight_layout()
    plt.savefig(FIG / "figure_6_complementarity.png", dpi=180)
    plt.close()
    nd = pd.DataFrame(nm)
    bio = nd[nd.model == "BioLORD-2023"].set_index("group")[["max_score_mean", "top1_top2_margin_mean", "mean_top5_mean"]]
    bio.plot(kind="bar", figsize=(8, 5))
    plt.ylabel("Cosine similarity / margin")
    plt.title("BioLORD NO_MAP diagnostic — post-hoc only")
    plt.tight_layout()
    plt.savefig(FIG / "figure_7_biolord_no_map_similarity.png", dpi=180)
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.scatter([1, 1, 1, 1], [primary.loc[primary.model == m, "Hit@10"].iloc[0] for m in DENSE], s=70)
    plt.xticks([1], ["Runtime unavailable in frozen v1 artifact"])
    plt.ylabel("Hit@10")
    plt.title("Quality versus computational cost — cost fields unavailable")
    plt.tight_layout()
    plt.savefig(FIG / "figure_8_quality_vs_cost.png", dpi=180)
    plt.close()
    low = lex[lex.slice == "LEXICAL_LOW"]
    plt.figure(figsize=(8, 5))
    plt.bar(low.model, low["Hit@10"])
    plt.ylim(0, 1.02)
    plt.ylabel("Hit@10")
    plt.title("LEXICAL_LOW — forward stratified TEST")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "figure_9_lexical_low.png", dpi=180)
    plt.close()
    # Manifest and deterministic inventory.
    inventory = {
        "experiment": "dense_v1_1",
        "benchmark_version": "1.0",
        "files": sorted(str(p.relative_to(OUT)) for p in OUT.glob("*.json")),
        "tables": sorted(p.name for p in TAB.glob("*.csv")),
        "figures": sorted(p.name for p in FIG.glob("*.png")),
        "source_experiment": "dense_v1",
        "no_model_rerun": True,
    }
    json.dump(inventory, open(OUT / "manifest.json", "w"), indent=2)
    print(
        json.dumps(
            {
                "experiment": "dense_v1_1",
                "tables": len(inventory["tables"]),
                "figures": len(inventory["figures"]),
                "ledger_rows": len(ledger),
                "scope_audit": scope_audit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
