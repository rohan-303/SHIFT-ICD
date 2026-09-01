# SHIFT-MAP v2 Step 8.1 — Compute-Feasibility Protocol

## Purpose

Step 8.1 is an engineering milestone only. It makes the existing MedCPT reranking pipeline resumable, dynamically padded, thermally guarded, and measurable. It does not alter the Step 8 scientific contract, train weights, select objectives, evaluate TEST, or implement scientific ablations.

## Frozen scientific contract

- Model: `ncbi/MedCPT-Cross-Encoder`
- Revision: `71caf65d4927987813984f54c284405a13fcca49`
- Input: source description plus target description only
- Max length: 96
- Candidate membership: exact frozen SHIFT-MAP v1.3 L2 top-100
- Candidate order: may be reordered after scoring
- Candidate additions/removals: forbidden
- TEST access: forbidden
- Optimizer updates: forbidden

## Thermal policy

GPU execution uses one process, CUDA device 0, and two CPU threads. The default thresholds are:

- `SOFT_PAUSE_TEMP_C = 80`
- `HARD_STOP_TEMP_C = 83`
- `RESUME_TEMP_C = 72`

At soft pause, the current completed chunk is persisted and no new GPU batch is submitted until cooldown reaches 72°C. At hard stop, progress is persisted and the run is marked `THERMAL_HARD_STOP`. The process must never intentionally continue to 85°C.

## Compute measurements

The reference workload is a deterministic approximately 1,000-pair subset from forward DEV covering sequence lengths, ranks, positives/negatives, and ordinary mapping kinds. FP32 is the reference. FP16 is accepted only if Spearman correlation is at least 0.999, source Top-1 agreement is at least 99%, and no material metric difference appears. Batch sizes 1, 2, 4, 8, 16, and 32 are measured only on the reference workload and stop on OOM or thermal hard stop. Dynamic tokenizer padding uses `padding="longest"`, `truncation=True`, and `max_length=96`.

Tokenization, host-to-device transfer where measurable, and forward time are recorded separately. Pretokenization is compared on a controlled subset and remains an implementation optimization only.

## Resumability

Every chunk is written before its manifest is advanced. Manifests include model revision, candidate-file hash, expected/completed pair counts, chunk ranges and SHA-256 hashes, precision, batch size, max length, device, timestamps, temperature maxima, pause/cooldown data, and completion status. Resume validates existing chunk hashes and rejects duplicates, gaps, and candidate-order changes.

## Training feasibility

No optimizer step is permitted. Future source-balanced BCE and set-positive listwise forward/backward probes may use deterministic TRAIN groups, but model state must be hashed before and after and remain unchanged. These are compute probes, not training or scientific selection.

## Scope boundary

Step 8.1 may use only TRAIN/DEV. It must not run forward TEST, family-held-out TEST, backward TEST, scientific objective/negative/LR ablations, final seeds, or test lock creation. A full DEV run is attempted only after the compute configuration is frozen; an incomplete run is reported as incomplete rather than converted into a result.
