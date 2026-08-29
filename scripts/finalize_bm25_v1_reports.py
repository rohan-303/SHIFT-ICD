# ruff: noqa: E501
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts/experiments/bm25_v1"
TABLES = ROOT / "reports/tables/bm25_v1"
FIGURES = ROOT / "reports/figures/bm25_v1"


def read_csv(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_svg(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    width, height, left, bottom = 900, 500, 90, 70
    max_value = max(max(values, default=1.0), 1e-9)
    points = []
    for i, value in enumerate(values):
        x = left + i * (width - left - 40) / max(len(values) - 1, 1)
        y = height - bottom - value / max_value * (height - bottom - 60)
        points.append(f"{x:.1f},{y:.1f}")
    text = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>', f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif">{ylabel}</text>', f'<polyline fill="none" stroke="#1f77b4" stroke-width="3" points="{" ".join(points)}"/>']
    for i, (label, value) in enumerate(zip(labels, values, strict=True)):
        x = left + i * (width - left - 40) / max(len(labels) - 1, 1)
        y = height - bottom - value / max_value * (height - bottom - 60)
        text.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#1f77b4"/><text x="{x:.1f}" y="{height-40}" text-anchor="middle" font-family="sans-serif" font-size="11">{label}</text>')
    text.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    overall = read_csv("baseline_overall.csv")
    tuned = next(row for row in overall if row["system"] == "BM25 Dev-Tuned")
    write_svg(FIGURES / "hit_at_k_curve.svg", "Forward stratified test: BM25 Dev-Tuned Hit@K", ["1", "5", "10", "25", "50", "100"], [float(tuned[f"Hit@{k}"]) for k in (1,5,10,25,50,100)], "Hit@K")
    lexical = read_csv("bm25_by_lexical_difficulty.csv")
    write_svg(FIGURES / "bm25_by_lexical_difficulty.svg", "BM25 by lexical difficulty", [r["lexical_difficulty"] for r in lexical], [float(r["Hit@10"] or 0) for r in lexical], "Hit@10")
    kinds = read_csv("bm25_by_mapping_kind.csv")
    write_svg(FIGURES / "bm25_by_mapping_kind.svg", "BM25 by mapping kind", [r["mapping_kind"] for r in kinds], [float(r["Hit@10"] or 0) for r in kinds], "Hit@10")
    no_map = json.loads((EXP / "no_map_diagnostics.json").read_text())
    rows = no_map.get("rows", [])
    with (TABLES / "bm25_no_map_score_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "n", "mean_max_score", "median_max_score", "q1_max_score", "q3_max_score", "std_max_score"])
        writer.writeheader()
        values = [float(r["max_bm25_score"]) for r in rows]
        writer.writerow({"group": "NO_MAP_forward_stratified_test", "n": len(values), "mean_max_score": statistics.mean(values) if values else None, "median_max_score": statistics.median(values) if values else None, "q1_max_score": statistics.quantiles(values, n=4)[0] if len(values) >= 2 else None, "q3_max_score": statistics.quantiles(values, n=4)[2] if len(values) >= 2 else None, "std_max_score": statistics.stdev(values) if len(values) >= 2 else None})
    write_svg(FIGURES / "mapped_vs_no_map_top_score.svg", "NO_MAP maximum BM25 score diagnostics", ["NO_MAP"], [statistics.mean([float(r["max_bm25_score"]) for r in rows]) if rows else 0], "Mean maximum score")
    selection = json.loads((EXP / "selection.json").read_text())
    q2_dev = json.loads((EXP / "q2_dev_ablation.json").read_text())
    metrics = json.loads((EXP / "test_metrics.json").read_text())
    report = f'''# BM25 v1 lexical retrieval baseline report

## 1. Objective

This experiment establishes the CPU lexical floor for frozen Track A Benchmark v1.0. It asks how much ICD cross-version mapping can be retrieved from terminology overlap alone and where lexical retrieval fails under mapping complexity and terminology drift. Evidence status: **Executed/Reproduced** for the completed run; the benchmark itself remains **Validated**.

## 2. Dataset and candidate corpora

Track A contains 86,271 source-level examples in two independent directions. The forward corpus is the unique ICD-10-CM target terminology and the backward corpus is the unique ICD-9-CM target terminology. Profiles are recorded in `corpus_profile.json`; every target concept appears once, while duplicate labels are retained.

## 3. Retrieval representation

Q1 long-description-only is primary. Q2 short-plus-long was evaluated on DEV only and was not selected (`{q2_dev["dev_noncombination_answerable_hit_at_10"]:.6f}` vs Q1 `{q2_dev["q1_selected_dev_hit_at_10"]:.6f}` non-combination Hit@10). Missing long text falls back to short text. Normalization is NFKC, lowercase, punctuation-to-boundaries, whitespace collapse, preserved digits/alphanumerics, no stopword removal, stemming, synonym expansion, abbreviation expansion, or spelling correction.

## 4. Baselines and BM25 configuration

B0 seeded random, B1 exact normalized label match, B2 token Jaccard overlap, B3 Robertson BM25 default (`k1=1.5`, `b=0.75`), and B4 BM25 Dev-Tuned were implemented. BM25 uses the internal inverted-index implementation and the preregistered formula in `docs/experiments/bm25_v1_protocol.md`. The frozen DEV winner is `k1={selection["winner"]["k1"]}`, `b={selection["winner"]["b"]}`, selected only on forward stratified DEV using non-combination answerable Hit@10 with complex CompleteScenarioRetrieval@10 as tie-break.

## 5. Main results

Machine-readable results are in `test_metrics.json` and `reports/tables/bm25_v1/`. The primary forward stratified TEST baseline table is `baseline_overall.csv`; split, direction, mapping-kind, lexical-difficulty, alternative-size, and combination tables are generated from those metrics. Confidence intervals are deterministic bootstrap intervals in `test_metrics.json`.

## 6. Generalization and asymmetry

Forward stratified, forward family-held-out, backward stratified, and backward family-held-out TEST datasets contain {metrics.get("forward_stratified_test/BM25 Dev-Tuned", {}).get("Hit@10", {}).get("estimate")} / {metrics.get("forward_family_held_out_test/BM25 Dev-Tuned", {}).get("Hit@10", {}).get("estimate")} / {metrics.get("backward_stratified_test/BM25 Dev-Tuned", {}).get("Hit@10", {}).get("estimate")} / {metrics.get("backward_family_held_out_test/BM25 Dev-Tuned", {}).get("Hit@10", {}).get("estimate")} BM25 Hit@10 estimates respectively. Interpret differences descriptively; no causal claim is made.

## 7. Complex mappings and alternatives

Choice-list recall and complete-scenario retrieval use the existing structural gold evaluator. Results are reported in `bm25_combination_retrieval.csv`. Larger alternative sets increase the chance of a Hit@K, so alternative-size stratification is required. The requested V54.12 / 533-alternative source is absent from the frozen benchmark JSONL; no rank is fabricated. This is a benchmark coverage issue to resolve in a future benchmark audit, not a retrieval result.

## 8. NO_MAP diagnostics

NO_MAP examples are excluded from ordinary recall. Forward stratified TEST produced {len(rows)} persisted score-diagnostic rows. Summary statistics are in `bm25_no_map_score_summary.csv`; no threshold or classifier was selected. The current diagnostic artifact is forward-test scoped and should not be mistaken for a complete mapped-vs-NO_MAP distributional study.

## 9. Error analysis and runtime

A deterministic sample of {sum(1 for _ in (ROOT / "artifacts/error_analysis/bm25_v1_forward_errors.jsonl").open(encoding="utf-8"))} forward stratified Top-10 failures is stored at `artifacts/error_analysis/bm25_v1_forward_errors.jsonl`, with unsupported clinical failure categories left blank. The completed run elapsed {json.loads((EXP / "manifest.json").read_text())["elapsed_seconds"]:.2f} seconds; corpus construction took {json.loads((EXP / "manifest.json").read_text())["corpus_construction_seconds"]:.2f} seconds. The implementation is CPU-only; memory footprint was not instrumented by the initial runner and is therefore not claimed.

## 10. Figures and limitations

Figures are SVG files under `reports/figures/bm25_v1/`. BM25 measures lexical overlap, cannot understand clinical semantic equivalence or ontology structure, cannot inherently identify NO_MAP cases, may benefit from shared terminology and large acceptable target sets, and may fail under lexical terminology drift. BM25 is not a clinical conversion rule and approximate GEM mappings are not exact clinical equivalence.

## 11. Future dense-retrieval gate

Future dense retrievers must use the same frozen examples, splits, and complex gold semantics. The reference point is the frozen B4 configuration (`k1={selection["winner"]["k1"]}`, `b={selection["winner"]["b"]}`), not an arbitrary improvement threshold.
'''
    (ROOT / "reports/bm25_v1_baseline_report.md").write_text(report, encoding="utf-8")
    print("generated report and figures")


if __name__ == "__main__":
    main()
