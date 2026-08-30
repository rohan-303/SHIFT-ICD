# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/experiments/shift_map_v1"
MOD = ROOT / "artifacts/models/shift_map_v1"
TAB = ROOT / "reports/tables/shift_map_v1"
FIG = ROOT / "reports/figures/shift_map_v1"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def runs(prefix):
    rows = []
    for d in MOD.glob(prefix + "*"):
        for p in d.glob("*_dev.json"):
            payload = json.loads(p.read_text())
            r = payload.get("summary", payload) if isinstance(payload, dict) else payload[-1]
            r["run_id"] = d.name
            r["epoch_file"] = p.name
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for name, prefix in [
        ("positive_policy_ablation", "dev_p"),
        ("negative_strategy_ablation", "dev_p2_l1_"),
        ("learning_rate_ablation", "dev_p2_l1_random_lr"),
    ]:
        d = runs(prefix)
        if len(d):
            d.to_csv(TAB / f"dev_{name}.csv", index=False)
            (OUT / f"{name}.json").write_text(d.to_json(orient="records", indent=2))
    seeds = []
    for seed in [17, 42, 2026]:
        d = MOD / f"final_seed{seed}"
        payload = json.loads((d / "epoch_3_dev.json").read_text())
        dev = payload.get("summary", payload) if isinstance(payload, dict) else payload[-1]
        seeds.append(
            {
                "seed": seed,
                "run_id": d.name,
                "dev": dev[-1] if isinstance(dev, list) else dev,
                "checkpoint": "epoch_3",
                "checkpoint_sha256": sha(d / "epoch_3" / "model.safetensors") if (d / "epoch_3" / "model.safetensors").exists() else None,
            }
        )
    (OUT / "seed_results.json").write_text(json.dumps(seeds, indent=2))
    ck = []
    for d in MOD.glob("final_seed*"):
        for e in d.glob("epoch_*"):
            files = [p for p in e.rglob("*") if p.is_file()]
            ck.append(
                {
                    "run_id": d.name,
                    "epoch": e.name,
                    "bytes": sum(p.stat().st_size for p in files),
                    "files": len(files),
                    "sha256": sha(next((p for p in files if p.name.endswith(".safetensors")), files[0])) if files else None,
                }
            )
    (OUT / "checkpoint_manifest.json").write_text(json.dumps(ck, indent=2))
    (OUT / "runtime.json").write_text(
        json.dumps(
            {
                "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU",
                "torch": "2.7.1+cu126",
                "precision": "float32",
                "micro_batch_size": 8,
                "gradient_accumulation": 4,
                "effective_source_batch_size": 32,
                "checkpoint_sizes": ck,
            },
            indent=2,
        )
    )
    names = [
        "dev_positive_policy_ablation",
        "dev_negative_strategy_ablation",
        "training_curves",
        "zero_shot_vs_shift_map",
        "lexical_difficulty",
        "family_held_out",
        "mapping_kind",
        "combination_transfer",
        "representation_drift",
        "complementarity",
    ]
    for _i, n in enumerate(names):
        plt.figure(figsize=(7, 4))
        plt.title("SHIFT-MAP v1 " + n.replace("_", " "))
        plt.text(0.5, 0.5, "Machine-readable source tables\nsee reports/tables/shift_map_v1", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIG / (n + ".png"), dpi=150)
        plt.close()
    report = ROOT / "reports/shift_map_v1_report.md"
    report.write_text(
        """# SHIFT-MAP v1 Report\n\n## 1. Research objective\nBioLORD-initialized supervised cross-version retrieval adaptation under a strict TRAIN/DEV/TEST boundary.\n\n## 2. Frozen baselines and boundary\nBM25 `bm25_v1_1` and zero-shot `dense_v1_1` remained unchanged. Gradient updates used only forward stratified TRAIN. DEV selected policy, objective, negative strategy, learning rate, epoch, and canonical seed. TEST was locked before evaluation.\n\n## 3. Training design\nP2 (SINGLE_EXACT + SINGLE_APPROXIMATE + ALTERNATIVE) with source-balanced sampling was selected. Combinations and NO_MAP were excluded from pairwise contrastive training. L1 masked single-positive InfoNCE was selected; no symmetric target-to-source loss was used. Full valid-positive masking prevents alternatives becoming negatives.\n\n## 4. Final configuration\nAdamW, LR `2e-5`, weight decay `0.01`, temperature `0.05`, 3 epochs, linear warmup/decay, max length 64, effective source batch 32, seed set 17/42/2026. DEV-selected canonical seed: 42.\n\n## 5. Test results\nSee `reports/tables/shift_map_v1/final_three_seed_results.csv`, `final_forward_overall.csv`, and `final_paired_comparisons.csv`. The final model is not clinically validated. On this run, SHIFT-MAP has strong Hit@100 but substantially lower Hit@1/Hit@10 than zero-shot BioLORD; this is a negative adaptation result at the most useful early-candidate ranks and is reported without post-hoc redesign.\n\n## 6. Slices and transfer\nLexical difficulty, mapping kind, alternative-size, family-held-out, backward transfer, combination transfer, NO_MAP diagnostics, and complementarity are serialized in the Step 7 tables. Reverse-direction results are transfer only: the model was trained forward, not backward.\n\n## 7. Safety and limitations\nNO_MAP values are post-hoc similarity diagnostics only; no abstention, threshold, calibration, or routing was implemented. CMS GEM retrieval is not clinical equivalence. Combination mappings were never reduced to pairwise positives. Fine-tuned weights remain local and Git-ignored; no redistribution occurred.\n\n## 8. Recommendation\n**BLOCKED for SHIFT-MAP v2 cross-encoder reranking.** First investigate the early-rank degradation, score/tie behavior, structural transfer, and representation drift under a separately approved protocol. Do not proceed as if the trained encoder improved retrieval.\n"""
    )


if __name__ == "__main__":
    main()
