from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from shift_icd.benchmark.schemas import BenchmarkExample
from shift_icd.dense.text import clean_dense_text

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/benchmarks/cms_track_a/v1.0/all_examples.jsonl"
TERM = ROOT / "artifacts/terminology_universe_v2"
OUT = ROOT / "artifacts/candidates/shift_map_full_universe_v2"
MODEL_ID = "FremyCompany/BioLORD-2023"
REV = "167aab527b238a50ca65224e6319215d2ff4fc9f"
CK_HASH = "9e9739e45ad041a4027cdfd611fb83e398debb50658ede710482660e16efe302"
LOCK_HASH = "5f97b347a811fd648e3d99a15b93e1c1ae583ca48c6d74868535c213e0081deb"


def docs():
    d = {}
    for line in (TERM / "icd10cm_diagnosis.jsonl").read_text(encoding="utf8").splitlines():
        x = json.loads(line)
        if x["terminology"] == "ICD-10-CM":
            d[x["canonical_code"]] = clean_dense_text(x.get("long_description") or x.get("short_description") or "")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    examples = [BenchmarkExample.model_validate_json(x) for x in BENCH.read_text(encoding="utf8").splitlines() if x.strip()]
    target = docs()
    codes = sorted(target)
    assert len(codes) == 71704
    model = SentenceTransformer(args.checkpoint, device="cuda", trust_remote_code=False)
    model.max_seq_length = 64
    start = time.time()
    tv = model.encode(
        [target[c] for c in codes],
        batch_size=256,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cuda",
        max_length=64,
    ).astype("float32")
    files = []
    for split in ["train", "dev", "test"]:
        src = [x for x in examples if x.direction == "ICD9CM_TO_ICD10CM" and x.split == split]
        q = model.encode(
            [clean_dense_text(x.source_label or "") for x in src],
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device="cuda",
            max_length=64,
        ).astype("float32")
        path = OUT / f"forward_{split}_k100.jsonl.gz"
        n = 0
        with gzip.open(path, "wt", encoding="utf8") as f:
            for x, v in zip(src, q, strict=True):
                scores = tv @ v
                idx = np.argpartition(-scores, 99)[:100]
                order = sorted(idx.tolist(), key=lambda i: (-float(scores[i]), codes[i]))
                for rank, i in enumerate(order, 1):
                    f.write(
                        json.dumps(
                            {
                                "schema_version": "shift_map_full_universe_candidate_v1",
                                "split": split,
                                "source_id": x.benchmark_id,
                                "source_code": x.source_code,
                                "source_description": x.source_label,
                                "candidate_rank": rank,
                                "target_code": codes[i],
                                "target_description": target[codes[i]],
                                "retriever_score": float(scores[i]),
                                "candidate_is_gold": codes[i] in set(x.valid_target_codes),
                                "candidate_is_gold_semantics": "EVALUATION_ONLY",
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    n += 1
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"split": split, "file": path.name, "source_count": len(src), "row_count": n, "sha256": h, "candidate_k": 100})
    model.to("cpu")
    del model
    manifest = {
        "schema_version": "shift_map_full_universe_candidate_freeze_manifest_v1",
        "status": "CANDIDATES_GENERATED_PENDING_FREEZE",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "canonical_seed": 17,
        "canonical_checkpoint_sha256": CK_HASH,
        "test_lock_sha256": LOCK_HASH,
        "base_model": MODEL_ID,
        "base_revision": REV,
        "target_count": 71704,
        "target_order_hash": "8ca0d0cf0213581645fe64d9c1e599552a2fb207eb740f32ca09ffeaaf9f8e26",
        "target_corpus_hash": "32f572abeb2ff4cb003145216fa83bfd9984fab96579205f432993aa9c550464",
        "elapsed_seconds": time.time() - start,
        "files": files,
    }
    (OUT / "generation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
