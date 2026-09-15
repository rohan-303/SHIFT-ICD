# ruff: noqa: E501
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/experiments/step8_full_universe"
CANDIDATE_ROOT = ROOT / "artifacts/candidates/shift_map_full_universe_v2"
FREEZE = ROOT / "artifacts/experiments/shift_map_full_universe/candidate_freeze_manifest.json"
BENCHMARK = ROOT / "data/benchmarks/cms_track_a/v1.0/forward/all.jsonl"
MAX_LENGTH = 96
MODEL_ID = "ncbi/MedCPT-Cross-Encoder"
MODEL_REVISION = "71caf65d4927987813984f54c284405a13fcca49"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with BENCHMARK.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows[str(row["benchmark_id"])] = row
    return rows


def code_descriptions() -> dict[str, str]:
    from shift_icd.terminology.universe import build_icd9_universe, build_icd10_universe

    corpora = [
        build_icd10_universe(ROOT / "data/raw/cms/2018_gem/2018-icd-10-code-descriptions.zip"),
        build_icd9_universe(ROOT / "data/raw/cms/2018_gem/icd-9-cm-v32-master-descriptions.zip"),
    ]
    return {
        record.canonical_code: (record.long_description or record.short_description or "")
        for corpus in corpora
        for record in corpus.records
    }


def rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def load_candidate_groups(path: Path) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows(path):
        groups.setdefault(str(row["source_id"]), []).append(row)
    return [sorted(groups[source_id], key=lambda row: int(row["candidate_rank"])) for source_id in sorted(groups)]


def audit_candidates() -> dict[str, Any]:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    benchmark = benchmark_rows()
    descriptions = code_descriptions()
    universe = {code for code in descriptions if code and code[0].isalnum()}
    result: dict[str, Any] = {"status": "PASS", "files": [], "description_join": {"missing_source": 0, "missing_target": 0}}
    expected = {item["split"]: item for item in freeze["files"]}
    for split in ("train", "dev", "test"):
        path = CANDIDATE_ROOT / f"forward_{split}_k100.jsonl.gz"
        data = rows(path)
        by_source: dict[str, list[dict[str, Any]]] = {}
        invalid_target = duplicate = wrong_rank = split_mismatch = invalid_source = 0
        seen_pairs: set[tuple[str, str]] = set()
        for row in data:
            source_id = str(row["source_id"])
            target = str(row["target_code"])
            pair = (source_id, target)
            if pair in seen_pairs:
                duplicate += 1
            seen_pairs.add(pair)
            by_source.setdefault(source_id, []).append(row)
            invalid_target += int(target not in universe)
            invalid_source += int(source_id not in benchmark)
            split_mismatch += int(row.get("split") != split)
            if source_id in benchmark:
                result["description_join"]["missing_source"] += int(not benchmark[source_id].get("source_label"))
            result["description_join"]["missing_target"] += int(target not in descriptions)
        for _source_id, group in by_source.items():
            ranks = sorted(int(row["candidate_rank"]) for row in group)
            wrong_rank += int(ranks != list(range(1, 101)))
        expected_file = expected[split]
        checks = {
            "split": split,
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "expected_sha256": expected_file["sha256"],
            "sha_match": sha256(path) == expected_file["sha256"],
            "source_count": len(by_source),
            "expected_source_count": expected_file["source_count"],
            "row_count": len(data),
            "expected_row_count": expected_file["row_count"],
            "exact_100_per_source": all(len(group) == 100 for group in by_source.values()),
            "duplicate_candidate_count": duplicate,
            "invalid_target_count": invalid_target,
            "missing_source_count": invalid_source,
            "wrong_rank_count": wrong_rank,
            "split_mismatch_count": split_mismatch,
            "deterministic_order": data == sorted(data, key=lambda r: (str(r["source_id"]), int(r["candidate_rank"]))),
        }
        result["files"].append(checks)
        if (
            not checks["sha_match"]
            or checks["source_count"] != checks["expected_source_count"]
            or checks["row_count"] != checks["expected_row_count"]
            or not checks["exact_100_per_source"]
            or any(
                checks[key]
                for key in (
                    "duplicate_candidate_count",
                    "invalid_target_count",
                    "missing_source_count",
                    "wrong_rank_count",
                    "split_mismatch_count",
                )
            )
            or not checks["deterministic_order"]
        ):
            result["status"] = "FAIL"
    if any(result["description_join"].values()):
        result["status"] = "FAIL"
    return result


def load_model(model_path: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    config = AutoConfig.from_pretrained(model_path, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=MODEL_REVISION, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False
    )
    params = sum(parameter.numel() for parameter in model.parameters())
    files = [
        {"path": str(item.relative_to(model_path)), "bytes": item.stat().st_size, "sha256": sha256(item)}
        for item in sorted(model_path.rglob("*"))
        if item.is_file()
    ]
    manifest = {
        "status": "PASS",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "config_sha256": sha256(model_path / "config.json"),
        "tokenizer_revision": MODEL_REVISION,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "model_class": model.__class__.__name__,
        "architecture": getattr(config, "architectures", None),
        "num_labels": int(config.num_labels),
        "parameter_count": params,
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "special_tokens": {
            name: getattr(tokenizer, name, None) for name in ("bos_token", "eos_token", "sep_token", "cls_token", "pad_token", "unk_token")
        },
        "torch": torch.__version__,
        "files": files,
    }
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT / "model_snapshot_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return manifest


def require_no_test() -> None:
    raise SystemExit("STEP8_TEST_ACCESS_FORBIDDEN: no corrected Step 8 TEST lock is supplied")


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN_REMOTE_NAMESPACE"


def emit_manifest(output: Path, *, objective: str, seed: int, batch_size: int, checkpoint_hash: str | None = None) -> dict[str, Any]:
    """Emit the compact provenance contract for every corrected runner invocation."""
    manifest = {
        "schema": "step8_run_manifest_v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "candidate_hashes": {
            split: next(item["sha256"] for item in json.loads(FREEZE.read_text(encoding="utf-8"))["files"] if item["split"] == split)
            for split in ("train", "dev", "test")
        },
        "candidate_contract_hash": sha256(ROOT / "docs/interfaces/step8_corrected_candidate_contract.md"),
        "training_population_hash": sha256(ARTIFACT / "training_population_manifest.json"),
        "objective": objective,
        "negative_list_strategy": "MIXED_RANK_WITHIN_FROZEN_TOP100",
        "learning_rate": 1e-5,
        "batch_size": batch_size,
        "effective_batch_size": batch_size,
        "max_length": MAX_LENGTH,
        "epochs": 1,
        "seed": seed,
        "git_head": _git_head(),
        "environment": {"python": sys.version, "platform": platform.platform()},
        "checkpoint_hash": checkpoint_hash,
        "dev_evaluator_version": "retrieval_evaluator_v3",
        "test_access_allowed": False,
        "test_scoring_count": 0,
        "test_training_count": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_dev_score(model_path: Path, output: Path, limit: int, batch_size: int) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from scripts.step8_full_universe.smoke import authoritative_pairs, score_pairs

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=MODEL_REVISION, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            model_path, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False
        )
        .float()
        .to("cuda:0")
        .eval()
    )
    groups, pairs = authoritative_pairs(CANDIDATE_ROOT / "forward_dev_k100.jsonl.gz", limit)
    scores = score_pairs(model, tokenizer, pairs, batch_size)
    result = {"split": "dev", "sources": len(groups), "pairs": len(pairs), "scores": len(scores), "precision": "FP32", "test_access": False}
    (output / "dev_score_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (output / "dev_score_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    emit_manifest(output, objective="DEV_SCORE_ONLY", seed=17, batch_size=batch_size)
    del model
    torch.cuda.empty_cache()
    return result


def run_training_smoke(model_path: Path, output: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from scripts.step8_full_universe.smoke import training_probe

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=MODEL_REVISION, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            model_path, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False
        )
        .float()
        .to("cuda:0")
    )
    result = training_probe(model, tokenizer, CANDIDATE_ROOT / "forward_train_k100.jsonl.gz", output)
    checkpoint = Path(result["checkpoint"])
    manifest = emit_manifest(
        output, objective="BCE_AND_SET_POSITIVE_LISTWISE_SMOKE", seed=17, batch_size=4, checkpoint_hash=sha256(checkpoint)
    )
    result["manifest"] = manifest
    (output / "training_smoke_summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected Step 8-R1 preflight runner")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit-candidates")
    model = sub.add_parser("model-snapshot")
    model.add_argument("--model-path", type=Path, required=True)
    score = sub.add_parser("score-dev")
    score.add_argument("--model-path", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--limit", type=int, default=100)
    score.add_argument("--batch-size", type=int, default=32)
    train = sub.add_parser("training-smoke")
    train.add_argument("--model-path", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    sub.add_parser("test")
    sub.add_parser("environment")
    args = parser.parse_args()
    if args.command == "audit-candidates":
        print(json.dumps(audit_candidates(), indent=2))
    elif args.command == "model-snapshot":
        print(json.dumps(load_model(args.model_path), indent=2, default=str))
    elif args.command == "test":
        require_no_test()
    elif args.command == "score-dev":
        print(json.dumps(run_dev_score(args.model_path, args.output, args.limit, args.batch_size), indent=2))
    elif args.command == "training-smoke":
        print(json.dumps(run_training_smoke(args.model_path, args.output), indent=2, default=str))
    elif args.command == "environment":
        print(json.dumps({"python": sys.version, "platform": platform.platform()}, indent=2))


if __name__ == "__main__":
    main()
