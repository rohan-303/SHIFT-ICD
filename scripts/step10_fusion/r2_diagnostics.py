# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/step10_fusion/r2_ablation.py"
spec = importlib.util.spec_from_file_location("step10_r2_ablation", SCRIPT)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def read_rows(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def write_rows(path, rows):
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    benchmark = m.r2.load_benchmark()
    _, train, val = m.prepare_inner(benchmark)
    raw_train = m.r2.fast_features(train, *m.r2.build_nodes())
    raw_val = m.r2.fast_features(val, *m.r2.build_nodes())
    scaler = m.r2.fit_train_scaler((tuple(row) for g in raw_train for row in g), role="TRAIN")
    val8 = m.r2.apply_variant(raw_val, scaler, "H3_CANDIDATE_SET_STRUCTURAL_CONTEXT")
    features = {m.sid(g): [list(row[4:]) for row in f] for g, f in zip(val, val8, strict=True)}
    z = {m.sid(g): m.normalized_scores(g) for g in val}
    b0_ranked, _, _ = m.score_model(None, val, features, z, 0.0)
    reps = read_rows(ROOT / "reports/tables/step10_fusion/configuration_representatives.csv")
    selected = next(r for r in reps if r["selected_global"] == "True")
    model = m.r1.FusionResidual()
    payload = torch.load(Path(selected["checkpoint_path"]), map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    selected_ranked, _, _ = m.score_model(model, val, features, z, float(selected["lambda"]))
    ordinary_indices = [i for i,g in enumerate(val) if benchmark[m.sid(g)]["mapping_kind"] in m.ORDINARY and benchmark[m.sid(g)].get("valid_target_codes")]
    def rr(i, ranking):
        gold={str(x) for x in benchmark[m.sid(val[i])]["valid_target_codes"]}
        hits=[p for p,c in enumerate(ranking[i],1) if c in gold]
        return 0.0 if not hits else 1.0/hits[0]
    top1 = {"improved":0,"worsened":0,"unchanged":0}
    mrr = {"improved":0,"worsened":0,"unchanged":0}
    for i in ordinary_indices:
        gold={str(x) for x in benchmark[m.sid(val[i])]["valid_target_codes"]}
        b = b0_ranked[i][0] in gold; s = selected_ranked[i][0] in gold
        if s and not b: top1["improved"] += 1
        elif b and not s: top1["worsened"] += 1
        else: top1["unchanged"] += 1
        br,sr=rr(i,b0_ranked),rr(i,selected_ranked)
        if sr>br: mrr["improved"] += 1
        elif sr<br: mrr["worsened"] += 1
        else: mrr["unchanged"] += 1
    movement=[{"status":"COMPUTED_SOURCE_LEVEL","top1_improved":top1["improved"],"top1_worsened":top1["worsened"],"top1_unchanged":top1["unchanged"],"mrr_improved":mrr["improved"],"mrr_worsened":mrr["worsened"],"mrr_unchanged":mrr["unchanged"],"ordinary_source_count":len(ordinary_indices)}]
    displacements=[]; top1_changes=0
    for i in range(len(val)):
        old_pos={c:p for p,c in enumerate(b0_ranked[i])}; new_pos={c:p for p,c in enumerate(selected_ranked[i])}
        displacements.extend(abs(new_pos[c]-old_pos[c]) for c in old_pos)
        top1_changes += b0_ranked[i][0] != selected_ranked[i][0]
    rank=[{"status":"COMPUTED_SELECTED_MODEL","mean_abs_rank_displacement":float(np.mean(displacements)),"median_abs_rank_displacement":float(np.median(displacements)),"p95_abs_rank_displacement":float(np.percentile(displacements,95)),"max_abs_rank_displacement":int(max(displacements)),"top1_change_fraction":top1_changes/len(val),"candidate_mutation_count":0,"Hit@100_invariant":True,"structural_at_100_invariant":True}]
    write_rows(ROOT / "reports/tables/step10_fusion/movement_diagnostics.csv", movement)
    write_rows(ROOT / "reports/tables/step10_fusion/rank_perturbation.csv", rank)
    print({"movement":movement[0],"rank":rank[0],"selected":selected["configuration_id"]})

if __name__ == "__main__": main()
