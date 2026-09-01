# ruff: noqa
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts/experiments/shift_map_v1_4"
TAB = ROOT / "reports/tables/shift_map_v1_4"
FIG = ROOT / "reports/figures/shift_map_v1_4"

def j(name): return json.loads((EXP / name).read_text(encoding="utf8"))
def write(name, rows):
    fields = sorted({k for r in rows for k in r}) or ["status"]
    with (TAB / name).open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
def save(name, title, labels, values):
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(labels, values); ax.set_title(title); ax.tick_params(axis="x", rotation=30); fig.tight_layout(); fig.savefig(FIG / name, dpi=120); plt.close(fig)

def main():
    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    write("representation_drift.csv", [{"status": j("representation_drift.json").get("status", "pending"), "mean": j("representation_drift.json").get("overall", {}).get("mean")}])
    write("source_drift.csv", [{"split": r.get("split"), **{k: v for k, v in r.items() if k != "split"}} for r in j("source_drift.json").get("rows", [])] or [{"status": "pending"}])
    write("inference_runtime.csv", j("runtime.json").get("inference_measurements", []) or [{"status": "pending", "training_peak_memory": "NOT_AVAILABLE"}])
    write("retrieval_error_floor.csv", [j("retrieval_error_floor.json")])
    write("final_retriever_spec.csv", [j("retriever_freeze.json")])
    write("scientific_position.csv", [{"claim": k, "classification": v} for k, v in j("scientific_position.json").items()])
    write("v2_gate.csv", [j("v2_gate.json")])
    write("family_held_out.csv", j("family_held_out.json").get("rows", []))
    write("backward_transfer.csv", [{"population": p, **r} for p, values in j("backward_transfer.json").get("populations", {}).items() for r in values])
    k = j("candidate_k_analysis.json").get("rows", []); write("candidate_k_tradeoff.csv", k)
    paired = j("l1_l2_paired.json").get("paired", []); write("l1_vs_l2_paired.csv", paired)
    save("01_l1_l2_paired.png", "L1 versus L2 paired delta", [r["metric"] for r in paired], [r["delta"] for r in paired])
    card = j("positive_cardinality.json").get("groups", []); l2card = [r for r in card if r.get("model") == "l2_seed17"]; save("02_positive_cardinality.png", "L2 Hit@10 by positive cardinality", [r["group"] for r in l2card], [r.get("Hit@10") or 0 for r in l2card])
    hub = j("hubness.json"); save("03_target_hubness.png", "Top-1 target uniqueness", list(hub), [hub[m]["unique_top1_targets"] for m in hub])
    geom = j("score_geometry.json"); save("04_gold_margin.png", "Top-1/top-2 margin summary", [r["model"] for r in geom], [r.get("mean") or 0 for r in geom])
    save("05_family_held_out.png", "Forward family-held-out Hit@10", [r["model"] for r in j("family_held_out.json")["rows"]], [r["Hit@10"] or 0 for r in j("family_held_out.json")["rows"]])
    back = j("backward_transfer.json")["populations"]["backward_stratified"]; save("06_backward_transfer.png", "Backward stratified Hit@10", [r["model"] for r in back], [r["Hit@10"] or 0 for r in back])
    save("07_candidate_k_coverage.png", "Candidate K versus ordinary coverage", [str(r["k"]) for r in k], [r["coverage"] for r in k])
    no = j("no_map_diagnostics.json")["groups"]; save("08_no_map_similarity.png", "NO_MAP versus answerable top similarity", list(no["l2_seed17"]), [no["l2_seed17"][g]["max_similarity"].get("mean", 0) for g in no["l2_seed17"]])
    save("09_l1_l2_success_regression.png", "Paired success/regression at K=100", ["L2-only", "L1-only", "Both", "Neither"], [j("l1_l2_paired.json")["success_regression"]["K=100"][x] for x in ["l2_success", "l1_success", "both_success", "both_fail"]])
    drift = j("representation_drift.json"); save("10_representation_drift_status.png", "Representation drift status", [drift.get("status", "pending")], [1])
    manifest = {"experiment_version": "1.4", "training_occurred": False, "generated_artifacts": sorted(p.name for p in EXP.glob("*.json") if p.name != "manifest.json"), "tables": sorted(p.name for p in TAB.glob("*.csv")), "figures": sorted(p.name for p in FIG.glob("*.png")), "test_exposure": True}
    (EXP / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(f"tables={len(list(TAB.glob('*.csv')))} figures={len(list(FIG.glob('*.png')))}")
if __name__ == "__main__": main()
