# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.evaluation.retrieval import choice_list_recall_at_k, complete_scenario_retrieval_at_k

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"
CANON = ROOT / "data/processed/cms/2018_gem/normalized_rows.parquet"


def metrics(
    examples: list[BenchmarkExample], model: SentenceTransformer, target_texts: dict[str, str], device: str
) -> list[dict[str, object]]:
    codes = sorted(target_texts)
    target = model.encode(
        [target_texts[c] for c in codes],
        batch_size=128,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
        max_length=64,
    ).astype(np.float32)
    query = model.encode(
        [x.source_label or "" for x in examples],
        batch_size=128,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
        max_length=64,
    ).astype(np.float32)
    out = []
    for x, q in zip(examples, query, strict=True):
        scores = target @ q
        take = min(100, len(codes))
        idx = np.argpartition(-scores, take - 1)[:take]
        ranked = [codes[i] for i in sorted(idx.tolist(), key=lambda i: (-float(scores[i]), codes[i]))]
        valid = set(x.valid_target_codes)
        complex_kind = x.mapping_kind in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}
        row = {
            "benchmark_id": x.benchmark_id,
            "direction": x.direction,
            "split": x.split,
            "source_family_split": x.source_family_split,
            "mapping_kind": x.mapping_kind,
            "lexical_difficulty": x.lexical_metadata.get("lexical_difficulty"),
            "valid_target_codes": sorted(valid),
            "n": len(valid),
            "top1_score": float(scores[idx].max()),
            "top2_score": float(sorted(scores[idx], reverse=True)[1]) if len(idx) > 1 else float(scores[idx].max()),
            "mean_top5_similarity": float(np.mean(sorted(scores[idx], reverse=True)[:5])),
        }
        for k in (1, 5, 10, 25, 50, 100):
            top = ranked[:k]
            row[f"Hit@{k}"] = int(bool(valid.intersection(top))) if valid and not complex_kind else None
            row[f"ChoiceListRecall@{k}"] = choice_list_recall_at_k(x, ranked, k)
            row[f"CompleteScenarioRetrieval@{k}"] = complete_scenario_retrieval_at_k(x, ranked, k)
        rr = next((1 / (i + 1) for i, c in enumerate(ranked) if c in valid), 0.0) if valid and not complex_kind else None
        row["MRR"] = rr
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    df = pd.read_parquet(CANON)
    target_texts = {str(c): str(g) for c, g in zip(df.target_code, df.target_label, strict=True) if pd.notna(c) and pd.notna(g)}
    examples = [BenchmarkExample.model_validate_json(line) for line in BENCH.open(encoding="utf8")]
    model = SentenceTransformer(args.model_dir, device=device)
    rows = metrics([x for x in examples if x.direction == "ICD9CM_TO_ICD10CM" and x.split == "test"], model, target_texts, device)
    rows += metrics(
        [x for x in examples if x.direction == "ICD10CM_TO_ICD9CM" and x.split == "test"],
        model,
        {
            str(c): str(g)
            for c, g in zip(
                df.loc[df.direction == "ICD10CM_TO_ICD9CM", "target_code"],
                df.loc[df.direction == "ICD10CM_TO_ICD9CM", "target_label"],
                strict=True,
            )
            if pd.notna(c) and pd.notna(g)
        },
        device,
    )
    Path(args.output).write_text(
        json.dumps(
            {"experiment": "shift_map_v1", "seed": args.seed, "model_dir": args.model_dir, "test_data_used": True, "rows": rows}, indent=2
        )
        + "\n",
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
