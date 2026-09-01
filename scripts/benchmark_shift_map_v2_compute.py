# ruff: noqa
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import gzip
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shift_icd.reranking.inference import ThermalGuard, ThermalState

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/experiments/shift_map_v2_compute"
MODEL_REVISION = "71caf65d4927987813984f54c284405a13fcca49"


def gpu_temperature() -> float:
    result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return float(result.stdout.strip().splitlines()[0])


def pairs(n: int) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    with gzip.open(ROOT / "artifacts/candidates/shift_map_v2/forward_stratified_dev_k100.jsonl.gz", "rt", encoding="utf8") as handle:
        for line in handle:
            row = json.loads(line)
            result.append((row["source_description"], row["target_description"]))
            if len(result) >= n:
                break
    return result


def score(batch_pairs: list[tuple[str, str]], tokenizer: Any, model: Any, device: str, batch_size: int, max_length: int, fp16: bool = False, guard: ThermalGuard | None = None) -> tuple[np.ndarray, float, list[int]]:
    values: list[float] = []
    lengths: list[int] = []
    start = time.perf_counter()
    for offset in range(0, len(batch_pairs), batch_size):
        if guard is not None:
            state = guard.check()
            while state == ThermalState.COOLDOWN:
                time.sleep(2)
                state = guard.check()
            if state == ThermalState.HARD_STOP:
                raise RuntimeError("THERMAL_HARD_STOP")
        batch = batch_pairs[offset : offset + batch_size]
        encoded = tokenizer(batch, padding="longest", truncation=True, max_length=max_length, return_tensors="pt", return_length=True)
        lengths.extend(int(x) for x in encoded.pop("length"))
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            if fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(**encoded).logits[:, 0]
            else:
                logits = model(**encoded).logits[:, 0]
        values.extend(logits.float().cpu().tolist())
    return np.asarray(values, dtype=np.float64), time.perf_counter() - start, lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=96)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    data = pairs(args.pairs)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(args.snapshot, local_files_only=True, revision=MODEL_REVISION).to(device).eval()
    guard = ThermalGuard(gpu_temperature, soft_pause=80, hard_stop=83, resume=72)
    results: list[dict[str, Any]] = []
    for batch_size in (1, 2, 4, 8, 16, 32):
        if guard.check() == ThermalState.HARD_STOP:
            break
        torch.cuda.reset_peak_memory_stats()
        start_temp = guard.max_temperature
        try:
            values, elapsed, lengths = score(data, tokenizer, model, device, batch_size, args.max_length)
            end_state = guard.check()
            results.append({"batch_size": batch_size, "pairs": len(data), "pairs_per_second": len(data) / elapsed, "wall_seconds": elapsed, "mean_batch_tokens": float(np.mean(lengths)), "max_batch_tokens": max(lengths), "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "start_temperature": start_temp, "maximum_temperature": guard.max_temperature, "thermal_state": str(end_state)})
            if end_state == ThermalState.HARD_STOP:
                break
        except RuntimeError as exc:
            results.append({"batch_size": batch_size, "status": "FAILED", "error": str(exc), "start_temperature": start_temp, "maximum_temperature": guard.max_temperature})
            break
    fp32, fp32_time, _ = score(data[: min(1000, len(data))], tokenizer, model, device, 8, args.max_length, False)
    fp16 = None
    fp16_time = None
    if guard.check() != ThermalState.HARD_STOP:
        fp16, fp16_time, _ = score(data[: min(1000, len(data))], tokenizer, model, device, 8, args.max_length, True)
    parity = {"n": len(fp32), "fp32_seconds": fp32_time, "fp16_seconds": fp16_time, "mean_abs_logit_difference": float(np.mean(np.abs(fp32 - fp16))) if fp16 is not None else None, "max_abs_logit_difference": float(np.max(np.abs(fp32 - fp16))) if fp16 is not None else None, "spearman": float(np.corrcoef(np.argsort(np.argsort(fp32)), np.argsort(np.argsort(fp16)))[0, 1]) if fp16 is not None else None, "top1_pair_agreement": None, "status": "MEASURED" if fp16 is not None else "THERMAL_HARD_STOP"}
    output = {"model_id": "ncbi/MedCPT-Cross-Encoder", "model_revision": MODEL_REVISION, "device": device, "pairs": len(data), "max_length": args.max_length, "thermal_policy": {"soft_pause": 80, "hard_stop": 83, "resume": 72}, "batch_results": results, "fp32_fp16_parity": parity, "maximum_temperature": guard.max_temperature, "pause_count": guard.pause_count, "cooldown_seconds": guard.cooldown_seconds, "hard_stop": guard.hard_stop_triggered, "status": "THERMAL_HARD_STOP" if guard.hard_stop_triggered else "COMPLETE"}
    (OUT / "batch_benchmark.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf8")
    (OUT / "fp16_parity.json").write_text(json.dumps(parity, indent=2) + "\n", encoding="utf8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
