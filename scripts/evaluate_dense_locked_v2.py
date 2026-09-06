# ruff: noqa: E501, I001
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from run_dense_full_universe_v2 import (
    BACKWARD,
    FORWARD,
    SPECS,
    evaluate_rows,
    load_examples,
    summarize,
)
from shift_icd.dense.models import load_encoder

ROOT = Path(__file__).resolve().parents[1]
TEST_OUT = Path(os.environ.get("DENSE_TEST_OUTPUT", str(ROOT / "artifacts/experiments/dense_full_universe_v2/test")))
TARGET_ROOT = Path(os.environ.get("DENSE_TARGET_ROOT", ""))
LOCK_HASH = os.environ.get("DENSE_TEST_LOCK_HASH", "")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def select_examples(rows: list[Any], direction: str, split: str) -> list[Any]:
    if split == "family_held_out":
        return [row for row in rows if row.direction == direction and row.source_family_split == "test"]
    return [row for row in rows if row.direction == direction and row.split == split]


def main() -> None:
    if not LOCK_HASH or not TARGET_ROOT.is_dir():
        raise RuntimeError("DENSE_TEST_LOCK_HASH and DENSE_TARGET_ROOT are required")
    rows = load_examples()

    datasets = {
        "forward_stratified_test": (FORWARD, "test"),
        "forward_family_held_out": (FORWARD, "family_held_out"),
        "backward_stratified_test": (BACKWARD, "test"),
        "backward_family_held_out": (BACKWARD, "family_held_out"),
    }
    all_results: dict[str, Any] = {}
    for spec in SPECS:
        slug = spec.name.lower().replace("-", "_")
        model_root = TARGET_ROOT / "dense_full_universe_v2" / slug
        if not model_root.is_dir():
            raise FileNotFoundError(model_root)
        encoder = load_encoder(spec, "cuda")
        matrices: dict[str, np.ndarray] = {}
        code_lists: dict[str, list[str]] = {}
        for direction in (FORWARD, BACKWARD):
            target_direction = "icd9cm_to_icd10cm" if direction == FORWARD else "icd10cm_to_icd9cm"
            metadata_path = model_root / "embeddings" / f"{slug}_{target_direction}_targets.json"
            matrix_path = metadata_path.with_suffix(".npy")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_count = 71704 if direction == FORWARD else 14567
            expected_hash = "8ca0d0cf0213581645fe64d9c1e599552a2fb207eb740f32ca09ffeaaf9f8e26" if direction == FORWARD else "204be6e29572332f3141a18347a28e95f2b258e5bacec8cf98046974adbc2eb5"
            if metadata["model_id"] != spec.model_id or metadata["revision"] != spec.revision or metadata["target_count"] != expected_count or metadata["code_hash"] != expected_hash or metadata["terminology_version"] != "terminology_universe_v2":
                raise RuntimeError(f"cache provenance mismatch for {spec.name}/{direction}")
            matrices[direction] = np.load(matrix_path, mmap_mode="r")
            code_lists[direction] = metadata["codes"]
            if matrices[direction].shape != (expected_count, spec.embedding_dim):
                raise RuntimeError(f"cache shape mismatch for {spec.name}/{direction}")
        result: dict[str, Any] = {"model_id": spec.model_id, "revision": spec.revision, "test_lock_hash": LOCK_HASH, "datasets": {}}
        for dataset_name, (direction, split) in datasets.items():
            examples = select_examples(rows, direction, split)
            evaluated, query_meta = evaluate_rows(examples, encoder, matrices, code_lists, 64, 32 if spec.kind != "transformers_cls" else 64)
            result["datasets"][dataset_name] = {"summary": summarize(evaluated), "n": len(evaluated), "query": query_meta}
            out = TEST_OUT / slug / f"{dataset_name}_rows.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in evaluated), encoding="utf-8")
        write_json(TEST_OUT / slug / "metrics.json", result)
        all_results[spec.name] = result
        encoder.close()
    write_json(TEST_OUT / "all_metrics.json", all_results)
    print(json.dumps({name: value["datasets"] for name, value in all_results.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
