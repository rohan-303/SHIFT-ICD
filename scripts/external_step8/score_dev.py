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


def evaluate_rankings(groups: list[dict[str, object]], ranked_codes: list[list[str]]) -> dict[str, object]:
    """Compute compact DEV-only ranking metrics from immutable candidate rows."""
    if len(groups) != len(ranked_codes):
        raise ValueError("ranking/source population mismatch")
    hit_ks = (1, 3, 5, 10, 25, 50, 100)
    hits = {k: 0 for k in hit_ks}
    reciprocal_ranks: list[float] = []
    ndcg = {5: [], 10: []}
    membership_valid = True
    for group, ranked in zip(groups, ranked_codes, strict=True):
        rows = list(group["rows"])
        candidates = [str(row["target_code"]) for row in rows]
        membership_valid = membership_valid and len(ranked) == len(candidates) and set(ranked) == set(candidates)
        gold = {str(row["target_code"]) for row in rows if bool(row["candidate_is_gold"])}
        positions = [index + 1 for index, code in enumerate(ranked) if code in gold]
        first = min(positions) if positions else None
        for k in hit_ks:
            hits[k] += int(first is not None and first <= k)
        reciprocal_ranks.append(0.0 if first is None else 1.0 / first)
        for k in ndcg:
            gains = [1.0 if code in gold else 0.0 for code in ranked[:k]]
            dcg = sum(gain / __import__("math").log2(index + 2) for index, gain in enumerate(gains))
            ideal = sum(1.0 / __import__("math").log2(index + 2) for index in range(min(len(gold), k)))
            ndcg[k].append(0.0 if ideal == 0.0 else dcg / ideal)
    count = len(groups)
    result: dict[str, object] = {"sources": count, "candidate_membership_valid": membership_valid}
    for k in hit_ks:
        result[f"hit_at_{k}"] = hits[k] / count if count else 0.0
    result["mrr"] = sum(reciprocal_ranks) / count if count else 0.0
    for k in ndcg:
        result[f"ndcg_at_{k}"] = sum(ndcg[k]) / count if count else 0.0
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

    all_ranked_codes: list[list[str]] = []
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
            codes = [r["target_code"] for r, _ in ranked]
            all_ranked_codes.append(codes)
            chunk_rows.append({
                "benchmark_id": group["benchmark_id"],
                "ranked_codes": codes,
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
    metrics = evaluate_rankings(gs, all_ranked_codes)
    metrics.update({"status": "COMPLETE", "pairs": len(gs) * 100, "model_revision": MODEL_REVISION, "precision": "FP32"})
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {**metrics, "run_dir": str(run_dir)}


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
