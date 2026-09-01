# ruff: noqa
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.dense.corpus import BACKWARD, FORWARD, build_target_corpus
from shift_icd.evaluation.retrieval import choice_list_recall_at_k, complete_scenario_retrieval_at_k

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"
CANON = ROOT / "data/processed/cms/2018_gem/normalized_rows.parquet"
K_VALUES = (1, 5, 10, 25, 50, 100)


def rank_rows(examples: list[BenchmarkExample], model: SentenceTransformer, target_texts: dict[str, str], device: str) -> list[dict[str, Any]]:
    codes = sorted(target_texts)
    target = model.encode([target_texts[c] for c in codes], batch_size=128, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True, device=device, max_length=64).astype(np.float32)
    query = model.encode([x.source_label or "" for x in examples], batch_size=128, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True, device=device, max_length=64).astype(np.float32)
    rows: list[dict[str, Any]] = []
    for example, vector in zip(examples, query, strict=True):
        scores = target @ vector
        take = min(100, len(codes))
        idx = np.argpartition(-scores, take - 1)[:take]
        order = sorted(idx.tolist(), key=lambda i: (-float(scores[i]), codes[i]))
        ranked = [codes[i] for i in order]
        ranked_scores = [float(scores[i]) for i in order]
        valid = set(example.valid_target_codes)
        complex_kind = example.mapping_kind in {"COMBINATION", "COMBINATION_WITH_ALTERNATIVES", "MULTI_SCENARIO"}
        row: dict[str, Any] = {"benchmark_id": example.benchmark_id, "direction": example.direction, "split": example.split, "source_family_split": example.source_family_split, "mapping_kind": example.mapping_kind, "lexical_difficulty": example.lexical_metadata.get("lexical_difficulty"), "valid_target_codes": sorted(valid), "n": len(valid), "ranked_codes": ranked, "ranked_scores": ranked_scores, "top1_target": ranked[0] if ranked else None, "top1_score": ranked_scores[0] if ranked_scores else None, "top2_score": ranked_scores[1] if len(ranked_scores) > 1 else None, "mean_top5_similarity": float(np.mean(ranked_scores[:5])) if ranked_scores else None}
        valid_rank = next((i + 1 for i, code in enumerate(ranked) if code in valid), None)
        row["best_valid_target_rank"] = valid_rank
        row["reciprocal_rank"] = (1.0 / valid_rank) if valid_rank and not complex_kind else (None if complex_kind else 0.0)
        for k in K_VALUES:
            top = ranked[:k]
            row[f"Hit@{k}"] = (int(bool(valid.intersection(top))) if valid and not complex_kind else None)
            row[f"ChoiceListRecall@{k}"] = choice_list_recall_at_k(example, ranked, k)
            row[f"CompleteScenarioRetrieval@{k}"] = complete_scenario_retrieval_at_k(example, ranked, k)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/experiments/shift_map_v1_4/ledgers")
    args = parser.parse_args()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame = pd.read_parquet(CANON)
    corpora = {direction: build_target_corpus(frame, direction) for direction in (FORWARD, BACKWARD)}
    examples = [BenchmarkExample.model_validate_json(line) for line in BENCH.open(encoding="utf8") if line.strip()]
    model_specs = {"zero_shot": {"model": "FremyCompany/BioLORD-2023", "revision": "167aab527b238a50ca65224e6319215d2ff4fc9f"}, "l1_seed42": {"model": str(ROOT / "artifacts/models/shift_map_v1/final_seed42/epoch_3")}, "l2_seed17": {"model": str(ROOT / "artifacts/models/shift_map_v1/v1_3_final_l2_n1_seed17/epoch_3")}}
    metadata: dict[str, Any] = {"experiment_version": "1.4", "evaluator_version": "2.0", "test_exposure": True, "device": device, "models": {}}
    for name, spec in model_specs.items():
        kwargs: dict[str, Any] = {"device": device, "trust_remote_code": True}
        if spec.get("revision"):
            kwargs["revision"] = spec["revision"]
        model = SentenceTransformer(spec["model"], **kwargs)
        metadata["models"][name] = spec
        for split in ("train", "dev", "test"):
            for direction, corpus in corpora.items():
                selected = [x for x in examples if x.direction == direction and x.split == split]
                rows = rank_rows(selected, model, corpus.as_dict(), device)
                (out / f"{name}_{'forward' if direction == FORWARD else 'backward'}_{split}.json").write_text(json.dumps({"experiment_version": "1.4", "evaluator_version": "2.0", "model": spec["model"], "direction": direction, "split": split, "target_corpus": {"count": len(corpus.codes), "hash": corpus.corpus_hash, "terminology_version": corpus.terminology_version}, "rows": rows}, indent=2) + "\n", encoding="utf8")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
    print(f"ledger_models={len(model_specs)} device={device}")


if __name__ == "__main__":
    main()
