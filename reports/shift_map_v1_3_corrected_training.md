# SHIFT-MAP v1.3 Corrected Controlled Training

## 1. Scope and validity

Step 7.3 retrained SHIFT-MAP v1 after evaluator repair. The evaluator is direction-scoped (`2.0`); the benchmark and canonical schema remain `1.0`. No SHIFT-MAP v2, reranking, calibration, routing, or other downstream method was implemented.

The Track A TEST partition was previously observed under the defective Step 7 evaluator and through corrected diagnostic L1 replay in Step 7.2. Therefore the TEST results below are historical/replayed, not pristine first-look holdout results. Configuration selection used forward stratified DEV only.

## 2. Provenance and composability

The historical N1/N2/N3/N5 negative-strategy runs and 5e-6/1e-5/2e-5 learning-rate runs were L1 checkpoints, so the sequential selections were not L2-composable. Required DEV-only bridges were run.

The L2 negative bridge selected N1_RANDOM. Corrected DEV epoch-3 results were: N1 Hit@100 0.985905, N2 0.985905 but lower CompleteScenarioRetrieval@100, N3 0.970326, and N5 0.985163. The L2 learning-rate bridge selected 2e-5: 5e-6 and 1e-5 reached 0.985163 Hit@100, while 2e-5 reached 0.986647 and CompleteScenarioRetrieval@100 0.820896.

## 3. Frozen final configuration

`P2 + L2 + N1_RANDOM + 2e-5`, AdamW, weight decay 0.01, temperature 0.05, 3 epochs, no scheduler, max sequence length 64, FP32, micro-batch 1, gradient accumulation 32, effective source batch 32. Base model: BioLORD-2023 revision `167aab527b238a50ca65224e6319215d2ff4fc9f`. Target corpus hash: `a394fe6a0055b13e2df4eaeb4ac4b739e7d8d54277adb02f9bdecff67ad80a3a`.

## 4. L2 implementation and cardinality audit

L2 uses a numerically stable log-mean-exp numerator over the complete valid positive set represented in the candidate batch, minus the all-candidate logsumexp denominator. Temperature is applied to normalized embedding similarities. Known valid targets are not treated as negatives; only source-to-target loss is used. The positive-set numerator is count-normalized, so positive-set cardinality does not receive a direct log-cardinality advantage. Synthetic full-positive and negative-masking tests pass.

## 5. DEV selection

Seed 17 selected epoch 3: Hit@1 0.718843, Hit@10 0.925074, Hit@100 0.987389, MRR 0.795917, CompleteScenarioRetrieval@100 0.835821.

Seed 42 selected epoch 3: Hit@1 0.722552, Hit@10 0.924332, Hit@100 0.986647, MRR 0.798212, CompleteScenarioRetrieval@100 0.820896.

Seed 2026 selected epoch 2 by the frozen lexicographic DEV criterion. Canonical seed: 17. No TEST result was used for either decision.

## 6. Forward TEST results

The locked ordinary answerable non-combination population is `n=2,695` per seed. Results:

| seed | Hit@1 | Hit@10 | Hit@100 | MRR |
|---:|---:|---:|---:|---:|
| 17 | 0.719852 | 0.929128 | 0.990353 | 0.795232 |
| 42 | 0.727273 | 0.929499 | 0.989610 | 0.800620 |
| 2026 | 0.726531 | 0.926902 | 0.989610 | 0.798974 |
| mean | 0.724552 | 0.928510 | 0.989858 | 0.798275 |
| SD | 0.004087 | 0.001405 | 0.000428 | 0.002762 |

## 7. Zero-shot comparison

Frozen BioLORD is Hit@1 0.717996, Hit@10 0.921336, Hit@100 0.984045, MRR 0.792142. Canonical L2 paired bootstrap, 5,000 samples, seed 2026: Hit@1 +0.001855 (95% CI -0.005566,+0.009276), Hit@10 +0.007792 (+0.002968,+0.012616), Hit@100 +0.006308 (+0.003340,+0.009647), MRR +0.003089 (-0.001250,+0.007428).

This is a modest and selective improvement, strongest at Hit@10/100; Hit@1 and MRR intervals include zero.

## 8. Baselines, slices, and limitations

Frozen BM25 primary values are Hit@10 0.781447, Hit@100 0.892022, MRR 0.651011. Frozen Qwen values are Hit@1 0.660111, Hit@10 0.900186, Hit@100 0.975881, MRR 0.744939. Canonical corrected slice tables are in `artifacts/experiments/shift_map_v1_3/` and `reports/tables/shift_map_v1_3/`.

The retained L1 TEST artifact is summary-only, so a paired per-query L1/L2 TEST CI cannot be computed without inventing data. Runtime peak GPU/CPU memory was not persisted by the historical training script and is marked unavailable rather than reconstructed. These are explicit limitations. TEST exposure also prevents pristine-holdout claims.

## 9. Outcome and v2 gate

Outcome classification: **C2 — MODEST BUT CONSISTENT IMPROVEMENT**, qualified by the nonzero Hit@10/Hit@100 paired intervals, low three-seed variation, and zero-containing Hit@1/MRR intervals. The model does not establish broad dominance or clinical equivalence.

Recommendation: **PATH A — use corrected L2 SHIFT-MAP v1 as candidate generator**, subject to the TEST-exposure limitation and follow-on validation. SHIFT-MAP v2 is not implemented and should begin only as a separate authorized milestone after review of this report.
