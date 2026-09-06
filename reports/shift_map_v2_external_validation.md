# Step 8.4 external compute validation — blocked preflight

**Run ID:** `external_validation_20260901_rohan_pc_preflight_blocked`  
**Scientific execution:** `false`

## Handoff and data integrity

- Repository: `C:\Users\rohan\SHIFT-ICD`
- Current commit: `54cf0b613450a2c3c0094bcb881fbb28eea0159c`
- The user-supplied handoff commit `ce7d16ad04b9424137a655f4cbf2acb06b35a96b` is present in history; the current commit is a later metadata amendment of the same handoff milestone.
- `git diff --check`: passed.
- Candidate/data verification: passed, including transferred TEST-file hash verification only. TEST was not scored or otherwise accessed for scientific evaluation.

## Actual runtime gate

The available host is not an external GPU environment:

```text
host: ROHAN-PC
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
VRAM: 6,144 MiB
minimum operational external-GPU guidance: 8,192 MiB
preferred guidance: 12,288 MiB
```

The repository runner returned `GPU_VRAM_BELOW_OPERATIONAL_MINIMUM_8GB`.

## Decision

**EF classification: NOT COMPUTABLE; external validation preflight blocked.**

No MedCPT model was loaded. No smoke, FP32 reference, FP16 parity, batch benchmark, full DEV scoring, tokenization profiling, pretokenization benchmark, BCE/listwise backward probe, checkpoint hash, cost estimate, or scientific training was run. All such values are `NOT RUN`.

This is an infrastructure gate, not a MedCPT, candidate, scientific-design, or inference-correctness failure. The frozen Step 8 contract remains unchanged. The laptop's historical 10,000-pair partial logits remain separate provenance evidence and were not read or merged.

## Required next action

Transfer or connect this repository to a real stable CUDA GPU machine with at least 8 GB VRAM (12 GB preferred). Then rerun `verify-data`, `preflight`, exact-revision download/provenance verification, smoke, FP32/FP16 parity, bounded batch benchmark, and full zero-shot DEV from source 0.

**STEP 8.5 is not authorized.**
