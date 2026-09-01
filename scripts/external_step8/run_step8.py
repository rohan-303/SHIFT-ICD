from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "artifacts" / "experiments" / "shift_map_v2_external"
CONFIG = ROOT / "configs" / "shift_map_v2_external.yaml"
DATA_MANIFEST = HANDOFF / "data_manifest.json"
RUN_ROOT = ROOT / "artifacts" / "experiments" / "shift_map_v2_external_runs"
MODEL_ID = "ncbi/MedCPT-Cross-Encoder"
MODEL_REVISION = "71caf65d4927987813984f54c284405a13fcca49"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data_manifest() -> dict[str, object]:
    return json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))


def verify_data() -> dict[str, object]:
    manifest = load_data_manifest()
    checked = []
    failures = []
    for row in manifest["files"]:
        item = dict(row)
        path = ROOT / str(item["relative_path"])
        exists = path.is_file()
        actual = sha256(path) if exists else None
        ok = exists and actual == item["sha256"]
        checked.append({"logical_name": item["logical_name"], "exists": exists, "sha256": actual, "ok": ok})
        if not ok:
            failures.append(str(item["relative_path"]))
    return {"status": "PASS" if not failures else "FAIL", "checked": checked, "failures": failures}


def preflight(cpu_check: bool) -> dict[str, object]:
    data = verify_data()
    reasons: list[str] = []
    if data["status"] != "PASS":
        reasons.append("DATA_HASH_OR_FILE_FAILURE")
    result: dict[str, object] = {
        "status": "PASS" if not reasons else "FAIL",
        "cpu_check": cpu_check,
        "python": sys.version,
        "os": platform.platform(),
        "data": data,
        "artifact_directory_writable": HANDOFF.is_dir() and HANDOFF.exists(),
    }
    if not cpu_check:
        try:
            import torch
            result.update({"cuda_available": bool(torch.cuda.is_available()), "gpu_count": torch.cuda.device_count()})
            if not torch.cuda.is_available():
                reasons.append("CUDA_UNAVAILABLE")
            elif torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3:
                reasons.append("GPU_VRAM_BELOW_OPERATIONAL_MINIMUM_8GB")
        except ImportError:
            reasons.append("TORCH_IMPORT_FAILED")
    if reasons:
        result["status"] = "FAIL"
        result["reasons"] = reasons
    return result


def locked_test() -> None:
    lock = HANDOFF / "test_lock.json"
    if not lock.is_file():
        raise SystemExit("TEST_LOCK_REQUIRED: create the approved external test lock before TEST execution")
    raise SystemExit("TEST_LOCK_PRESENT_BUT_TEST_EXECUTION_IS_NOT_IMPLEMENTED_IN_HANDOFF")


def status() -> dict[str, object]:
    return {
        "handoff_version": "8.3",
        "scientific_version": "2.0",
        "artifact_root": str(RUN_ROOT),
        "test_lock": (HANDOFF / "test_lock.json").is_file(),
        "data": verify_data(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Portable frozen Step 8 external-compute handoff")
    subs = parser.add_subparsers(dest="command", required=True)
    pre = subs.add_parser("preflight", help="validate runtime and data; use --cpu-check for local validation")
    pre.add_argument("--cpu-check", action="store_true")
    subs.add_parser("verify-data", help="verify all portable file hashes")
    subs.add_parser("download-model", help="download the exact Hugging Face revision on external compute")
    smoke = subs.add_parser("smoke", help="score the deterministic 10-source DEV smoke subset")
    smoke.add_argument("--model-path")
    smoke.add_argument("--output-root", type=Path, default=RUN_ROOT)
    dev = subs.add_parser("zero-shot-dev", help="score all DEV from source 0 in a new external run namespace")
    dev.add_argument("--model-path")
    dev.add_argument("--output-root", type=Path, default=RUN_ROOT)
    subs.add_parser("training-probe", help="run no-optimizer-step BCE/listwise memory probes")
    for name in ("objective-run", "negative-run", "lr-run", "final-seed"):
        subs.add_parser(name, help="reserved scientific stage; requires prior gates")
    subs.add_parser("test", help="protected TEST stage")
    subs.add_parser("status", help="show handoff and artifact status")
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(args.cpu_check), indent=2))
    elif args.command == "verify-data":
        result = verify_data()
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["status"] == "PASS" else 1)
    elif args.command == "test":
        locked_test()
    elif args.command == "status":
        print(json.dumps(status(), indent=2))
    elif args.command in {"smoke", "zero-shot-dev"}:
        if not args.model_path:
            raise SystemExit("MODEL_PATH_REQUIRED: pass the local path of the exact downloaded snapshot")
        command = [
            sys.executable,
            str(Path(__file__).with_name("score_dev.py")),
            "--model-path", args.model_path,
            "--output-root", str(args.output_root),
        ]
        if args.command == "smoke":
            command += ["--limit-sources", "10"]
        result = subprocess.run(command, cwd=ROOT, text=True)
        raise SystemExit(result.returncode)
    elif args.command in {"download-model", "training-probe", "objective-run", "negative-run", "lr-run", "final-seed"}:
        payload = {
            "status": "EXTERNAL_STAGE_REQUIRED",
            "command": args.command,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "artifact_root": str(RUN_ROOT),
        }
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
