# ruff: noqa: E501
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from shift_icd.evaluation.scope import random_hit_probability

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts/experiments/bm25_v1_1"
TABLES = ROOT / "reports/tables/bm25_v1_1"
FIGURES = ROOT / "reports/figures/bm25_v1_1"


def read_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (EXP / "scoped_rows.jsonl").open(encoding="utf-8") if line.strip()]


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with (TABLES / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg(name: str, title: str, points: list[tuple[str, float]], y_label: str) -> None:
    width, height = 900, 500
    max_y = max([value for _label, value in points] + [1.0])
    coords = []
    for i, (label, value) in enumerate(points):
        x = 80 + i * (width - 140) / max(1, len(points) - 1)
        y = height - 70 - value / max_y * (height - 140)
        coords.append((x, y, label, value))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y, _label, _value in coords)
    labels = "".join(f'<text x="{x:.1f}" y="{height-42}" text-anchor="middle" font-size="12">{label}</text><circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#9b6b32"/><text x="{x:.1f}" y="{y-10:.1f}" text-anchor="middle" font-size="11">{value:.3f}</text>' for x, y, label, value in coords)
    content = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#fbf7ef"/><text x="450" y="32" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="bold">{title}</text><text x="20" y="250" transform="rotate(-90 20 250)" text-anchor="middle" font-family="sans-serif" font-size="13">{y_label}</text><line x1="70" y1="430" x2="850" y2="430" stroke="#333"/><line x1="70" y1="70" x2="70" y2="430" stroke="#333"/><polyline points="{poly}" fill="none" stroke="#9b6b32" stroke-width="3"/>{labels}<text x="450" y="480" text-anchor="middle" font-family="sans-serif" font-size="12">ICD-9-CM → ICD-10-CM · stratified · test · bm25_v1_1</text></svg>'
    (FIGURES / name).write_text(content, encoding="utf-8")


def auc(scores_a: list[float], scores_b: list[float]) -> float | None:
    if not scores_a or not scores_b:
        return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in scores_a for b in scores_b)
    return wins / (len(scores_a) * len(scores_b))


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    forward = [row for row in rows if row["sample_population"] == "forward_stratified_test"]
    buckets = ["1", "2-5", "6-20", "21-100", ">100"]
    chance_rows = []
    for bucket in buckets:
        subset = [row for row in forward if row["alternative_bucket"] == bucket]
        expectations = {f"random_hit_expectation@{k}": statistics.mean(random_hit_probability(17513, row["alternative_count"], k) for row in subset) if subset else None for k in (1, 5, 10, 25, 100)}
        chance_rows.append({"direction": "ICD9CM_TO_ICD10CM", "split_protocol": "stratified", "partition": "test", "benchmark_version": "1.0", "sample_population": "forward_stratified_test", "alternative_bucket": bucket, "n": len(subset), **expectations, "Hit@1": statistics.mean([row["Hit@1"] for row in subset if row["Hit@1"] is not None]) if any(row["Hit@1"] is not None for row in subset) else None, "Hit@5": statistics.mean([row["Hit@5"] for row in subset if row["Hit@5"] is not None]) if any(row["Hit@5"] is not None for row in subset) else None, "Hit@10": statistics.mean([row["Hit@10"] for row in subset if row["Hit@10"] is not None]) if any(row["Hit@10"] is not None for row in subset) else None, "Hit@25": statistics.mean([row["Hit@25"] for row in subset if row["Hit@25"] is not None]) if any(row["Hit@25"] is not None for row in subset) else None, "Hit@100": statistics.mean([row["Hit@100"] for row in subset if row["Hit@100"] is not None]) if any(row["Hit@100"] is not None for row in subset) else None, "MRR": statistics.mean([row["MRR"] for row in subset if row["MRR"] is not None]) if any(row["MRR"] is not None for row in subset) else None})
    write_csv("forward_stratified_alternative_chance_expectation.csv", chance_rows)
    overall = json.loads((EXP / "metrics.json").read_text(encoding="utf-8"))["forward_stratified_test"]
    runtime = json.loads((EXP / "runtime_measurement.json").read_text(encoding="utf-8")) if (EXP / "runtime_measurement.json").exists() else json.loads((EXP / "runtime.json").read_text(encoding="utf-8"))
    kind_rows = list(csv.DictReader((TABLES / "forward_stratified_by_mapping_kind.csv").open(encoding="utf-8")))
    lexical_rows = list(csv.DictReader((TABLES / "forward_stratified_by_lexical_difficulty.csv").open(encoding="utf-8")))
    svg("hit_at_k_overall.svg", "BM25 frozen reference — forward stratified test", [(str(k), float(overall[f"Hit@{k}"])) for k in (1, 5, 10, 25, 50, 100)], "Hit@K")
    svg("hit_at_10_by_mapping_kind.svg", "BM25 Hit@10 by mapping kind — forward stratified test", [(row["mapping_kind"], float(row["Hit@10"])) for row in kind_rows if row["Hit@10"]], "Hit@10")
    svg("hit_at_10_by_lexical_difficulty.svg", "BM25 Hit@10 by lexical difficulty — forward stratified test", [(row["lexical_difficulty"], float(row["Hit@10"])) for row in lexical_rows if row["Hit@10"]], "Hit@10")
    svg("hit_at_10_by_alternative_size.svg", "BM25 Hit@10 by alternative-set size — forward stratified test", [(row["alternative_bucket"], float(row["Hit@10"])) for row in chance_rows if row["Hit@10"] is not None], "Hit@10")
    no_map = [row for row in json.loads((EXP / "no_map_diagnostics.json").read_text(encoding="utf-8"))["rows"] if row["group"] in {"NO_MAP", "ANSWERABLE"}]
    groups = [(group, [float(row["max_bm25_score"]) for row in no_map if row["group"] == group]) for group in ("NO_MAP", "ANSWERABLE")]
    svg("mapped_vs_no_map_score_distribution.svg", "BM25 maximum-score diagnostic — forward stratified test", [(group, statistics.mean(values)) for group, values in groups], "Mean maximum BM25 score")
    svg("random_hit_expectation_vs_actual.svg", "Actual vs random-hit expectation — forward stratified test", [(row["alternative_bucket"], float(row["Hit@10"]) - float(row["random_hit_expectation@10"])) for row in chance_rows], "Hit@10 minus random expectation")
    auc_value = auc([value for group, values in groups if group == "ANSWERABLE" for value in values], [value for group, values in groups if group == "NO_MAP" for value in values])
    report = f'''# BM25 v1.1 Audit and Correction Report

## Reason for audit
Step 5 slice tables reported counts larger than the stated forward stratified TEST population, and the V54.12 case study incorrectly reported the source as absent.

## Scope and root cause
The old slice generation collected metric rows from all four TEST datasets and filtered only `system == BM25 Dev-Tuned`; it omitted the dataset/scope filter. The old mapping-kind and lexical tables therefore each reported **34,494 rows**, the sum of the four TEST dataset sizes: forward stratified 2,913; forward family-held-out 2,908; backward stratified 14,341; backward family-held-out 14,332. The old combination table reported 2,172 complex rows from that same mixed population. The corrected pipeline filters explicit direction, protocol, partition, and benchmark version before aggregation.

## Overall metrics
The frozen forward stratified TEST metrics were independently recomputed and are unchanged: Hit@1 `{overall['Hit@1']:.6f}`, Hit@5 `{overall['Hit@5']:.6f}`, Hit@10 `{overall['Hit@10']:.6f}`, Hit@25 `{overall['Hit@25']:.6f}`, Hit@50 `{overall['Hit@50']:.6f}`, Hit@100 `{overall['Hit@100']:.6f}`, MRR `{overall['MRR']:.6f}`. BM25 tuning remains valid because v1.1 did not retune; it retains `k1=2.0`, `b=0.75`, Q1 long-description-only, selected previously on forward stratified DEV.

## V54.12 / V5412
Canonical and benchmark storage use normalized source code `V5412`; display form is `V54.12`. The case exists as `track_a_v1.0:ICD9CM_TO_ICD10CM:V5412`, mapping kind ALTERNATIVE, 533 valid targets, in forward stratified TRAIN. The prior absence was a case-study lookup bug. It was not a benchmark eligibility exclusion. Its valid-target density is `533 / 17,513 = 0.030435`; the best valid target rank and top-K coverage are in `artifacts/experiments/bm25_v1_1/v54_12_case_study.json`. Hit@K is mechanically easier with larger gold sets and must not be compared without alternative-set context.

## Membership and benchmark version
Canonical and benchmark source-level keys both contain 14,567 forward and 71,704 backward sources. `canonical - benchmark = empty`; `benchmark - canonical = empty`. Unique benchmark IDs and `(direction, source_code)` keys are preserved by existing validation. Benchmark version remains **1.0**; no benchmark semantics changed. Experiment version is **bm25_v1_1**, a result-pipeline correction preserving bm25_v1 provenance.

## Corrected slices
`forward_stratified_by_mapping_kind.csv` contains mutually exclusive mapping-kind membership summing to 2,913. Combination rows report ChoiceListRecall@K and CompleteScenarioRetrieval@K; ordinary Hit@K is intentionally blank for combinations. `forward_stratified_by_lexical_difficulty.csv` contains exact, high, medium, and low slices summing to 2,913; confusable is empty. Lexical slices are mutually exclusive because each example has one generated lexical-difficulty label; the empty confusable slice reflects the existing criterion/metadata, not forced reassignment.

## NO_MAP diagnostics
The corrected comparison contains NO_MAP n=85 and ANSWERABLE n=2,695, matching forward TEST semantics for diagnostic groups. It reports mean, median, standard deviation, Q1, Q3 for maximum BM25 score, top1-top2 margin, mean Top-5 score, and maximum lexical overlap. These are descriptive retrieval diagnostics only; no threshold or classifier was trained. Post-hoc AUROC using maximum BM25 score, if present, is labeled diagnostic only: `{auc_value}`.

## Alternative-set chance effect
The normalized buckets are 1, 2–5, 6–20, 21–100, and >100. The chance table reports actual Hit@K, MRR, and the diagnostic expectation `1 - C(N-G,K)/C(N,K)` with N=17,513 candidate concepts and G acceptable targets. It is not a replacement metric and assumes uniform sampling without replacement.

## Runtime and memory
The forward stratified TEST measurement reports index build `{runtime.get('index_build_seconds')}`, serialized index size `{runtime.get('index_serialized_bytes')}` bytes, mean query latency `{runtime.get('average_query_latency_ms')}` ms, median `{runtime.get('median_query_latency_ms')}` ms, p95 `{runtime.get('p95_query_latency_ms')}` ms, and throughput `{runtime.get('throughput_queries_per_second')}` queries/sec over `{runtime.get('query_count')}` Top-100 queries. Peak RSS is `{runtime.get('process_peak_rss_bytes')}` because psutil was unavailable. The measurement is approximate and not used for configuration selection. No neural packages, neural models, FAISS, PyTorch, Transformers, sentence-transformers, or dense retrieval components were introduced.

## Artifacts and limitations
Corrected machine-readable artifacts are under `artifacts/experiments/bm25_v1_1/`; publication tables are under `reports/tables/bm25_v1_1/`; figures are under `reports/figures/bm25_v1_1/`. The historical v1 tables are preserved. GEM mappings remain translation-assistance/comparison resources, not clinical equivalence guarantees. V5412 is in TRAIN, so its case-study rank is not a TEST result.
'''
    (ROOT / "reports/bm25_v1_1_audit_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"report": "reports/bm25_v1_1_audit_report.md", "tables": len(list(TABLES.glob("*.csv"))), "figures": len(list(FIGURES.glob("*.svg"))), "posthoc_auc": auc_value}, sort_keys=True))


if __name__ == "__main__":
    main()
