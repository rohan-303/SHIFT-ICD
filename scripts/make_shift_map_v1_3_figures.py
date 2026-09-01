# ruff: noqa: E501, E701, E702
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts/experiments/shift_map_v1_3"
OUT = ROOT / "reports/figures/shift_map_v1_3"
OUT.mkdir(parents=True, exist_ok=True)

def csv_plot(filename: str, title: str, metric: str) -> None:
    rows = [r for r in csv.DictReader((EXP / filename).open()) if r.get(metric)]
    fig, ax = plt.subplots()
    ax.bar([r["group"] for r in rows], [float(r[metric]) for r in rows])
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUT / filename.replace(".csv", ".png"), dpi=120)
    plt.close(fig)

csv_plot("mapping_kind.csv", "Canonical L2 mapping kind Hit@10", "Hit@10")
csv_plot("alternative_analysis.csv", "Canonical L2 positive-set cardinality Hit@10", "Hit@10")
csv_plot("family_held_out.csv", "Canonical L2 family split Hit@10", "Hit@10")

rows = json.loads((EXP / "test_seed17.json").read_text())["populations"]["ICD9CM_TO_ICD10CM"]["rows"]
for filename, title, selector in [
    ("04_combination_transfer.png", "Combination choice-list recall", lambda r: r["mapping_kind"] == "COMBINATION"),
    ("05_backward_transfer.png", "Forward-trained to backward transfer", None),
]:
    selected = [r for r in rows if selector(r)] if selector else json.loads((EXP / "test_seed17.json").read_text())["populations"]["ICD10CM_TO_ICD9CM"]["rows"]
    selected = [r for r in selected if r.get("valid_target_codes")]
    fig, ax = plt.subplots()
    if selected:
        values = [sum(float(r.get("ChoiceListRecall@10", r.get("Hit@10", 0))) for r in selected) / len(selected)]
        ax.bar(["canonical"], values)
    ax.set_ylim(0, 1); ax.set_title(title); fig.tight_layout(); fig.savefig(OUT / filename, dpi=120); plt.close(fig)

fig, ax = plt.subplots(); hub = json.loads((EXP / "hubness.json").read_text())["top20"]
ax.bar(range(min(20, len(hub))), [v for _, v in hub[:20]]); ax.set_title("Canonical L2 top-1 target hubness"); fig.tight_layout(); fig.savefig(OUT / "06_target_hubness.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(); dev = json.loads((EXP / "seed_dev_metrics.json").read_text())
for seed, history in dev["seeds"].items(): ax.plot([h["epoch"] for h in history], [h["Hit@10"] for h in history], marker="o", label=seed)
ax.set_title("Corrected DEV training curves: Hit@10"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "07_training_curves.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots()
bridge = json.loads((EXP / "bridge_ablation.json").read_text())["negative_strategy"]
ax.bar(list(bridge), [bridge[k]["Hit@100"] for k in bridge])
ax.set_ylim(.9, 1); ax.set_title("L2 negative-strategy bridge"); fig.tight_layout(); fig.savefig(OUT / "08_l2_negative_bridge.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots()
lr = json.loads((EXP / "bridge_ablation.json").read_text())["learning_rate"]
ax.bar(list(lr), [lr[k]["Hit@100"] for k in lr])
ax.set_ylim(.9, 1); ax.set_title("L2 learning-rate bridge"); fig.tight_layout(); fig.savefig(OUT / "09_l2_learning_rate_bridge.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(); ax.plot([1, 10, 100], [.719852, .929128, .990353], marker="o", label="L2 seed 17"); ax.plot([1, 10, 100], [.717996, .921336, .984045], marker="o", label="BioLORD"); ax.set_ylim(.6, 1); ax.set_title("Corrected L2 vs zero-shot Hit@K"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "10_hit_at_k.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(); ax.axis("off"); ax.text(.05, .5, "Representation drift: not computed\nRuntime peak memory: unavailable", fontsize=14); fig.savefig(OUT / "11_unavailable_analyses.png", dpi=120); plt.close(fig)
fig, ax = plt.subplots(); ax.axis("off"); ax.text(.05, .5, "L1 paired TEST ledger: unavailable\nComparison retained as summary-only", fontsize=14); fig.savefig(OUT / "12_l1_limitation.png", dpi=120); plt.close(fig)
print("figures_written", len(list(OUT.glob("*.png"))))
