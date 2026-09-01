# SHIFT-ICD Step 8.1 Compute Feasibility Report

**Compute experiment version:** `8.1`
**Scientific Step 8 version:** `2.0`
**Status:** `CF4 — full MedCPT Step 8 not established as practical on current power window`

## 1. Why Step 8.1 was required

The prior GPU attempt reached 85°C and the CPU fallback exceeded the bounded window. Step 8.1 implemented guarded, resumable inference instead of repeating an opaque full run.

## 2. Frozen scientific contract

MedCPT Cross-Encoder revision `71caf65d4927987813984f54c284405a13fcca49`; max length 96; frozen SHIFT-MAP v1.3 top-100 candidates; source and target descriptions only; no TEST access; no optimizer updates.

## 3. Hardware and power

RTX 3060 Laptop GPU, 6,144 MiB VRAM. Pre-run: 80% battery, AC connected, 49°C. During DEV execution AC disconnected at 100/1,457 sources and 62°C; the process was terminated immediately.

## 4. Thermal safety policy

Soft pause 80°C; hard stop 83°C; resume 72°C. The reference benchmark reached 66°C with zero thermal pauses and zero thermal hard stops. The DEV run stopped for power loss, not thermal excess.

## 5. Resumable inference design

`src/shift_icd/reranking/inference.py` provides deterministic ordering, chunk SHA-256, duplicate rejection, missing-range detection, persisted manifests, and thermal state handling. The DEV runner uses 50-source/5,000-pair chunks, batch 32, dynamic longest padding, and resume support. Two chunks (100 sources, 10,000 pairs) were persisted and hash-verified.

## 6–8. Dynamic padding, FP32 reference, and FP16 parity

A deterministic 1,000-pair GPU reference used tokenizer-native `padding="longest"`, truncation, and max length 96. FP32 was the reference. FP16 mean absolute logit difference was `0.007343999667093158`, maximum difference `0.5193080902099609`, and Spearman `0.999994215994216`. Top-1 agreement was not recorded, so FP16 was not automatically promoted; FP32 is retained.

## 9. Batch benchmark

Batch 1/2/4/8/16/32 achieved respectively 77.28/134.89/253.79/502.37/762.35/1028.04 pairs/s, with peak temperatures 53/54/58/64/64/66°C. Peak allocated VRAM ranged from 429.0 MiB to 471.0 MiB. Batch 32 was the largest measured safe reference batch under AC power.

## 10. Tokenization and pretokenization

Separate tokenization, host-transfer, and forward timers were not instrumented. Pretokenized-cache comparison was not executed. Dynamic padding itself was exercised.

## 11–18. Full DEV and training probes

Full DEV metrics are `NOT COMPUTED`: only 100/1,457 sources and 10,000/145,700 pairs completed. Hit@1, Hit@5, Hit@10, Hit@100, MRR, NDCG@10, full membership aggregation, and the Hit@100 invariant are therefore unavailable. BCE/listwise backward probes and FP16 training numerics were not run. No optimizer step occurred and no checkpoint was written. Training cost estimates are not computed.

## 19–21. Limitations, classification, continuation

**CF4.** The short reference benchmark demonstrates safe inference under AC power, but complete DEV feasibility and training feasibility were not established because AC disconnected. This is an engineering/resource block, not a scientific/model failure. The main limiting factor is safe sustained execution on the laptop power envelope. The next milestone should resume from the validated 10,000-pair manifest only during an explicitly stable AC window or on suitable external compute. Do not replace MedCPT, change K, alter hypotheses, train, or access TEST.

## Artifact paths

- Protocol: `C:\Users\rohan\SHIFT-ICD\docs\experiments\shift_map_v2_compute_protocol.md`
- Compute artifacts: `C:\Users\rohan\SHIFT-ICD\artifacts\experiments\shift_map_v2_compute\`
- Tables: `C:\Users\rohan\SHIFT-ICD\reports\tables\shift_map_v2_compute\`
