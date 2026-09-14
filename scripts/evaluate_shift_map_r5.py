from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.dense.text import clean_dense_text
from shift_icd.evaluation.metric_contract import classify_population, metric_values
from shift_icd.evaluation.retrieval import choice_list_recall_at_k, complete_scenario_retrieval_at_k

K = (1, 5, 10, 25, 50, 100)
ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"
TERM = ROOT / "artifacts/terminology_universe_v2"
OUT = ROOT / "artifacts/experiments/shift_map_full_universe/r5_evaluations"
MODEL_ID = "FremyCompany/BioLORD-2023"
REV = "167aab527b238a50ca65224e6319215d2ff4fc9f"
LOCK_HASH = "5f97b347a811fd648e3d99a15b93e1c1ae583ca48c6d74868535c213e0081deb"


def load_examples():
    return [BenchmarkExample.model_validate_json(x) for x in BENCH.read_text(encoding="utf8").splitlines() if x.strip()]


def docs(filename, terminology):
    out = {}
    for line in (TERM / filename).read_text(encoding="utf8").splitlines():
        x = json.loads(line)
        if x["terminology"] == terminology:
            out[x["canonical_code"]] = clean_dense_text(x.get("long_description") or x.get("short_description") or "")
    return out


def ranked(scores, codes):
    idx = np.argpartition(-scores, 99)[:100]
    order = sorted(idx.tolist(), key=lambda i: (-float(scores[i]), codes[i]))
    return [codes[i] for i in order], [float(scores[i]) for i in order]


def evaluate(examples, model, corpus):
    targets = {}
    codes = {}
    for direction, d in corpus.items():
        codes[direction] = sorted(d)
        targets[direction] = model.encode(
            [d[c] for c in codes[direction]],
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device="cuda",
            max_length=64,
        ).astype("float32")
    result = {}
    for name, selected in {
        "forward_stratified_test": [x for x in examples if x.direction == "ICD9CM_TO_ICD10CM" and x.split == "test"],
        "forward_family_held_out": [x for x in examples if x.direction == "ICD9CM_TO_ICD10CM" and x.source_family_split == "test"],
        "backward_stratified_test": [x for x in examples if x.direction == "ICD10CM_TO_ICD9CM" and x.split == "test"],
        "backward_family_held_out": [x for x in examples if x.direction == "ICD10CM_TO_ICD9CM" and x.source_family_split == "test"],
    }.items():
        q = model.encode(
            [clean_dense_text(x.source_label or "") for x in selected],
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device="cuda",
            max_length=64,
        ).astype("float32")
        rows = []
        for x, v in zip(selected, q, strict=True):
            rc, sc = ranked(targets[x.direction] @ v, codes[x.direction])
            valid = sorted(x.valid_target_codes)
            pop = classify_population(x)
            row = {
                "benchmark_id": x.benchmark_id,
                "direction": x.direction,
                "source_code": x.source_code,
                "source_description": x.source_label,
                "mapping_kind": x.mapping_kind,
                "population": pop,
                "lexical_difficulty": x.lexical_metadata.get("lexical_difficulty", "UNKNOWN"),
                "no_map": x.no_map,
                "valid_target_codes": valid,
                "ranked_codes": rc,
                "scores": sc,
            }
            row.update(metric_values(set(valid), rc, no_map=x.no_map, k_values=K))
            for k in K:
                row[f"ChoiceListRecall@{k}"] = choice_list_recall_at_k(x, rc, k)
                row[f"CompleteScenarioRetrieval@{k}"] = complete_scenario_retrieval_at_k(x, rc, k)
            rows.append(row)
        result[name] = {"n": len(rows), "rows": rows, "summary_by_population": summaries(rows)}
    return result


def summaries(rows):
    out = {}
    pops = ["P_ORDINARY_ANSWERABLE", "P_COMBINATION", "P_COMBINATION_WITH_ALTERNATIVES", "P_COMPLEX", "P_NO_MAP"]
    for pop in pops:
        rr = [
            r
            for r in rows
            if r["population"] == pop
            or (pop == "P_COMPLEX" and r["population"] in {"P_COMBINATION", "P_COMBINATION_WITH_ALTERNATIVES", "P_COMPLEX"})
        ]
        d = {"n": len(rr)}
        for k in K:
            for prefix in ["Hit", "ChoiceListRecall", "CompleteScenarioRetrieval"]:
                vals = [r[f"{prefix}@{k}"] for r in rr if r[f"{prefix}@{k}"] is not None]
                if vals:
                    d[f"{prefix}@{k}"] = statistics.mean(vals)
        vals = [r["MRR"] for r in rr if r["MRR"] is not None]
        if vals:
            d["MRR"] = statistics.mean(vals)
        out[pop] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--checkpoint")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    examples = load_examples()
    corpus = {
        "ICD9CM_TO_ICD10CM": docs("icd10cm_diagnosis.jsonl", "ICD-10-CM"),
        "ICD10CM_TO_ICD9CM": docs("icd9cm_diagnosis.jsonl", "ICD-9-CM"),
    }
    assert len(corpus["ICD9CM_TO_ICD10CM"]) == 71704 and len(corpus["ICD10CM_TO_ICD9CM"]) == 14567
    started = time.time()
    model = SentenceTransformer(
        args.checkpoint if args.checkpoint else MODEL_ID, revision=None if args.checkpoint else REV, device="cuda", trust_remote_code=False
    )
    model.max_seq_length = 64
    res = evaluate(examples, model, corpus)
    model.to("cpu")
    del model
    torch.cuda.empty_cache()
    payload = {
        "schema_version": "shift_map_full_universe_r5_evaluation_v1",
        "system": args.system,
        "checkpoint": args.checkpoint or "zero_shot_biolord",
        "base_model": MODEL_ID,
        "base_revision": REV,
        "test_lock_sha256": LOCK_HASH,
        "target_counts": {"forward": 71704, "backward": 14567},
        "evaluator": "retrieval_evaluator_v3",
        "elapsed_seconds": time.time() - started,
        "datasets": {k: {"n": v["n"], "summary_by_population": v["summary_by_population"]} for k, v in res.items()},
        "test_data_used": True,
    }
    sd = OUT / args.system
    sd.mkdir(parents=True, exist_ok=True)
    for k, v in res.items():
        (sd / f"{k}_rows.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in v["rows"]), encoding="utf8")
    (sd / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
