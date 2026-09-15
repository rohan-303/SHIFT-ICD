# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path
from statistics import quantiles

from scripts.step8_full_universe.runner import benchmark_rows, code_descriptions, load_candidate_groups

ROOT = Path(__file__).resolve().parents[2]

def stats(values: list[int]) -> dict[str, float | int]:
    q = quantiles(values, n=100, method="inclusive")
    return {"p50": q[49], "p95": q[94], "p99": q[98], "max": max(values), "n": len(values)}

def main() -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    benchmark = benchmark_rows()
    descriptions = code_descriptions()
    output: dict[str, object] = {}
    for split in ("train", "dev"):
        path = ROOT / f"artifacts/candidates/shift_map_full_universe_v2/forward_{split}_k100.jsonl.gz"
        groups = load_candidate_groups(path)
        sources = [str(benchmark[str(g[0]["source_id"])]["source_label"]) for g in groups]
        targets = [descriptions[str(r["target_code"])] for g in groups for r in g]
        pairs = [(sources[i], targets[i * 100 + j]) for i in range(len(groups)) for j in range(100)]
        def lengths(values: list[str] | list[tuple[str, str]]) -> list[int]:
            result: list[int] = []
            for start in range(0, len(values), 4096):
                batch = values[start:start + 4096]
                encoded = tokenizer(batch, truncation=False, padding=False, return_length=True)
                result.extend(int(x) for x in encoded["length"])
            return result
        source_l = lengths(sources)
        target_l = lengths(targets)
        pair_l = lengths(pairs)
        output[split] = {"source": stats(source_l), "target": stats(target_l), "paired": stats(pair_l), "truncation_fraction_at_96": sum(x > 96 for x in pair_l) / len(pair_l)}
    (ROOT / "token_lengths.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf8")

if __name__ == "__main__":
    main()
