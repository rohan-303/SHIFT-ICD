from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "artifacts/candidates/shift_map_v2/forward_stratified_dev_k100.jsonl.gz"
MODEL_REVISION = "71caf65d4927987813984f54c284405a13fcca49"


def groups() -> list[dict[str, object]]:
    by_id: dict[str, dict[str, object]] = {}
    with gzip.open(CANDIDATE, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            by_id.setdefault(row["benchmark_id"], {"benchmark_id": row["benchmark_id"], "rows": []})["rows"].append(row)
    result = list(by_id.values())
    for group in result:
        group["rows"].sort(key=lambda row: int(row["retriever_rank"]))
        if len(group["rows"]) != 100:
            raise ValueError("candidate membership invariant failed: expected 100 rows/source")
    return result


def score(model_path: str, output_root: Path, limit_sources: int | None = None, batch_size: int = 32) -> dict[str, object]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_EXTERNAL_SCORING")
    gs = groups()[:limit_sources] if limit_sources else groups()
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=MODEL_REVISION, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, revision=MODEL_REVISION, local_files_only=True
    ).to("cuda:0").eval()
    run_dir = output_root / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(gs), 50):
        selected = gs[start:start + 50]
        pairs = [(r["source_description"], r["target_description"]) for g in selected for r in g["rows"]]
        values: list[float] = []
        for offset in range(0, len(pairs), batch_size):
            encoded = tokenizer(pairs[offset:offset + batch_size], padding="longest", truncation=True, max_length=96, return_tensors="pt")
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            with torch.inference_mode():
                values.extend(model(**encoded).logits[:, 0].float().cpu().tolist())
        if len(values) != len(pairs) or not all(torch.isfinite(torch.tensor(values))):
            raise RuntimeError("NONFINITE_OR_MISSING_LOGITS")
        position = 0
        chunk_rows = []
        for group in selected:
            source_rows = group["rows"]
            logits = values[position:position + len(source_rows)]
            position += len(source_rows)
            ranked = sorted(
                zip(source_rows, logits, strict=True),
                key=lambda item: (-item[1], int(item[0]["retriever_rank"])),
            )
            chunk_rows.append({
                "benchmark_id": group["benchmark_id"],
                "ranked_codes": [r["target_code"] for r, _ in ranked],
                "scores": [float(v) for _, v in ranked],
            })
        payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in chunk_rows).encode()
        chunk = chunks_dir / f"sources-{start:06d}-{start + len(selected):06d}.jsonl"
        chunk.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        manifest = run_dir / "manifest.json"
        current = (
            json.loads(manifest.read_text())
            if manifest.exists()
            else {"model_revision": MODEL_REVISION, "precision": "FP32", "chunks": []}
        )
        current["chunks"].append({
            "chunk_id": chunk.stem,
            "source_start": start,
            "source_end": start + len(selected),
            "pair_count": len(pairs),
            "sha256": digest,
        })
        manifest.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "COMPLETE",
        "sources": len(gs),
        "pairs": len(gs) * 100,
        "run_dir": str(run_dir),
        "model_revision": MODEL_REVISION,
        "precision": "FP32",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit-sources", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(score(args.model_path, args.output_root, args.limit_sources, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
