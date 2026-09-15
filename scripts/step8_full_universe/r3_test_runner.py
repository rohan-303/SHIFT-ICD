# ruff: noqa: E501
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from scripts.step8_full_universe import r2_runner as r2

LOCK_SHA256 = "3ebbad472841743aeb023e3eebab59bcbf9e5c34aa692c91437f7fb0bb3aa543"
TEST_CANDIDATE_SHA256 = "6ef38443a201b1d630148b30717d845b0785b1a4ed433eeab4b03d893b6bf1ce"
BENCHMARK_SHA256 = "a177f2ba4d95889837c19a4906f6d36b1978c6b332e064a7b3ae3c002df6fe9f"
SEEDS = (17, 42, 2026)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_lock(lock: Path, candidate: Path, benchmark: Path, checkpoints: dict[int, Path]) -> dict[str, Any]:
    if sha256(lock) != LOCK_SHA256:
        raise RuntimeError("TEST_LOCK_HASH_MISMATCH")
    data = json.loads(lock.read_text(encoding="utf-8"))
    if data.get("status") != "TEST_LOCKED_BEFORE_SCORING":
        raise RuntimeError("TEST_LOCK_STATUS_INVALID")
    if data.get("test_scoring_count_before_lock") != 0 or data.get("test_training_count_before_lock") != 0:
        raise RuntimeError("TEST_ACCESS_BEFORE_LOCK")
    if sha256(candidate) != TEST_CANDIDATE_SHA256:
        raise RuntimeError("TEST_CANDIDATE_HASH_MISMATCH")
    if sha256(benchmark) != BENCHMARK_SHA256:
        raise RuntimeError("BENCHMARK_HASH_MISMATCH")
    actual = {str(seed): sha256(path) for seed, path in checkpoints.items()}
    if actual != data["checkpoint_sha256"]:
        raise RuntimeError(f"CHECKPOINT_HASH_MISMATCH: {actual}")
    return {"lock_sha256": LOCK_SHA256, "candidate_test_sha256": TEST_CANDIDATE_SHA256, "benchmark_sha256": BENCHMARK_SHA256, "checkpoint_sha256": actual}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = {seed: args.checkpoint_root / f"seed_{seed}/selected_checkpoints/final_seed_set_positive_listwise_random_within_candidate_3e-05.pt" for seed in SEEDS}
    provenance = verify_lock(args.lock, args.candidate, args.benchmark, checkpoints)
    benchmark = r2.benchmark_rows()
    test_groups = r2.prepare_groups(args.candidate, benchmark)
    baseline_ranked = [[str(row["target_code"]) for row in group] for group in test_groups]
    output: dict[str, Any] = {"schema": "step8_r3_corrected_test_scores_v1", "status": "TEST_SCORING_COMPLETE", "provenance": provenance, "source_count": len(test_groups), "candidate_mutation_count": 0, "test_training_count": 0, "test_scoring_count": 0, "scores": {}}
    output["scores"]["SHIFT_MAP"] = {"metrics": r2.metrics(test_groups, baseline_ranked, benchmark), "ranked": baseline_ranked}
    tokenizer = None
    model = None
    device = torch.device("cuda:0")
    for seed in SEEDS:
        if tokenizer is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=r2.MODEL_REVISION, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(args.model_path, revision=r2.MODEL_REVISION, local_files_only=True, trust_remote_code=False).float().to(device)
        state = torch.load(checkpoints[seed], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        ranked = r2.score_groups(model, tokenizer, test_groups, benchmark, device)
        output["scores"][f"MEDCPT_SEED_{seed}"] = {"checkpoint_sha256": provenance["checkpoint_sha256"][str(seed)], "metrics": r2.metrics(test_groups, ranked, benchmark), "ranked": ranked}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "source_count": len(test_groups), "models_scored": list(output["scores"]), "test_training_count": 0}, indent=2))


if __name__ == "__main__":
    main()
