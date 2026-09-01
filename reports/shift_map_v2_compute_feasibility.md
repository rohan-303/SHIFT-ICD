# SHIFT-ICD Step 8.2 Compute Feasibility Report

**Compute experiment version:** `8.2`
**Scientific Step 8 version:** `2.0`
**Status:** `CF4 — still not established because AC power was unreliable`

## 1. Step 8.1 prior interruption

Step 8.1 established a safe 1,000-pair GPU reference benchmark, reaching 66°C at batch 32. The initial full DEV run stopped after AC power disconnected at 100/1,457 sources and 10,000/145,700 pairs. This was not a thermal, OOM, model, candidate, or correctness failure.

## 2. Stable-power preflight

A four-sample, 15-second preflight passed immediately before the resume attempt: all samples had `power_plugged=True`, battery 61%, and GPU temperature 50–51°C. During the resumed run, AC disconnected again at 64% battery. The runner emitted `POWER_LOSS_STOP` and exited without recomputing completed chunks or continuing on battery.

## 3. Resume integrity

The two existing chunks were revalidated before execution. Both SHA-256 hashes matched. The resumed runner skipped both chunks. No new chunk was persisted because AC disconnected before another atomic chunk completed. Existing state remains 100 sources and 10,000 pairs; remaining work is 1,357 sources and 135,700 pairs.

## 4. Complete zero-shot DEV

Full DEV completion: **INCOMPLETE**. Only the existing 100-source/10,000-pair state is validly completed. The partial metrics are not reported as scientific results.

## 5. Hit@100 invariant

The complete DEV Hit@100 invariant was not evaluated because full aggregation did not complete. No interpretation of partial metrics is permitted.

## 6. Full DEV scientific metrics

Hit@1, Hit@3, Hit@5, Hit@10, Hit@25, Hit@50, Hit@100, MRR, NDCG@5, and NDCG@10: **NOT COMPUTED**.

## 7. Full DEV runtime

Complete active GPU runtime, complete wall-clock runtime, effective throughput, complete peak memory, and complete cooldown time: **NOT COMPUTED**. Resume attempt wall time was 8.97 seconds before power-loss stop; this is not a full-run runtime.

## 8. Thermal behavior

Thermal policy remains unchanged: soft pause 80°C, hard stop 83°C, resume 72°C. The resume attempt reached 57°C in the last recorded sample. Thermal pause count was zero. No process termination occurred due to temperature; the only stop was the required AC power-loss stop.

## 9. Tokenization profile

Separate tokenizer, host/device transfer, and forward-pass timing shares remain **NOT COMPUTED**. This was deferred because the required stable power window failed during resumed execution.

## 10. Pretokenization decision

**NOT COMPUTED.** No cache-vs-on-the-fly controlled comparison was run.

## 11–13. BCE, listwise, and memory feasibility

BCE backward probe: **NOT RUN**. Listwise backward probe: **NOT RUN**. Training memory behavior and FP16 training numerics: **NOT RUN**. No optimizer updates occurred and no checkpoint was created.

## 14. Weight immutability

No model weights were written, no checkpoint was saved, and no optimizer step occurred. The runner was inference-only. A byte-level before/after model-cache comparison was not executed in this interruption-only milestone.

## 15. Training cost projection

**NOT COMPUTED**, because source-level forward/backward throughput was not measured.

## 16. Resumable scientific staging plan

The planned order remains: zero-shot DEV resume, BCE probe, listwise probe, negative-strategy runs, learning-rate runs, and final seeds 17, 42, and 2026. No scientific training was executed.

## 17. Revised classification

**CF4 — STILL NOT ESTABLISHED / UNSUITABLE POWER ENVIRONMENT.** The limiting factor is repeated AC disconnection, not GPU temperature or model compute. The short reference benchmark remains safe, but full DEV and training feasibility cannot be established without stable AC.

## 18. Recommended next milestone

Pause local GPU execution until the laptop has a verified stable AC connection or use suitable external compute. Resume from the existing hash-verified 10,000-pair manifest boundary. Do not replace MedCPT, change candidate K, alter max length or hypotheses, perform optimizer updates, or access TEST.

## Artifact paths

- Protocol: `C:\Users\rohan\SHIFT-ICD\docs\experiments\shift_map_v2_compute_protocol.md`
- Compute artifacts: `C:\Users\rohan\SHIFT-ICD\artifacts\experiments\shift_map_v2_compute\`
- Chunk manifest: `C:\Users\rohan\SHIFT-ICD\artifacts\experiments\shift_map_v2_compute\zero_shot_dev_chunks\manifest.json`
- Tables: `C:\Users\rohan\SHIFT-ICD\reports\tables\shift_map_v2_compute\`
