# SHIFT-MAP v2 Step 8.3 External Compute Handoff Report

**Scope:** `C:\Users\rohan\SHIFT-ICD` only. Engineering handoff; no local long GPU execution.

1. **Repository state before handoff:** `main`, clean; latest pre-handoff state included the Step 8.2 power-loss record.
2. **Compute-feasibility checkpoint commit:** `2c63a3628f0a364d0f7014b6b0a66db559e4fae3` — `chore: preserve SHIFT-MAP v2 compute feasibility work`.
3. **Step 8.3 handoff version:** `8.3`.
4. **Current software environment:** Windows 10 build `10.0.26200`; Python `3.11.15`; package `shift_icd 0.1.0`; PyTorch `2.7.1+cu126`; CUDA build `12.6`; Transformers `4.52.4`; tokenizers `0.21.4`; safetensors `0.5.3`; huggingface-hub `0.30.2`; NumPy `2.4.6`; pandas `2.3.3`; SciPy `1.17.1`; scikit-learn `1.9.0`; psutil `7.0.0`. Full freeze: `artifacts/experiments/shift_map_v2_external/python_environment_lock.txt`.
5. **Minimal external dependencies:** `requirements-step8.txt` (`torch==2.7.1`, Transformers/tokenizers/safetensors/HF Hub, NumPy/pandas/SciPy/scikit-learn/psutil/PyYAML). GPU drivers/CUDA runtime remain host-managed.
6. **Frozen MedCPT model/revision:** `ncbi/MedCPT-Cross-Encoder`, revision `71caf65d4927987813984f54c284405a13fcca49`; measured parameters `109,483,009`; weights excluded from Git and handoff.
7. **Frozen candidate K:** `100`.
8. **Max length:** `96`.
9. **Required portable files:** candidate manifest; frozen TRAIN/DEV/TEST candidate gzip files; benchmark manifest; forward stratified TRAIN/DEV/TEST description JSONL; external config; requirements; runner scripts; environment evidence; execution guide.
10. **TRAIN candidate file hash:** `f632854cd22ff9875390d9ff7280b4033b571c0c41ac54561730bf6f77536a35`.
11. **DEV candidate file hash:** `ea7d94aa7f8e2d73d9d07dfff6d21a1b8e9a4a3db7837dd5dfa9f76feb92eafa`.
12. **TEST candidate file hash:** `adcdcc2348165723c29f76687e5f487c2e5275dac5ff2389352ff4570d8f961b`.
13. **Portable data manifest hash:** `09f7fc5ac1a9fd353c398845db972ab056b55ba54f4a6d16034d136fd3999045` for `artifacts/experiments/shift_map_v2_external/data_manifest.json`.
14. **External config path/hash:** `configs/shift_map_v2_external.yaml`; SHA-256 `494cf8936878b8022e9316273f8bf4638759c2946de9898a05d67514f8ae44fa`.
15. **External runner path:** `scripts/external_step8/run_step8.py`; companion scorer `scripts/external_step8/score_dev.py`.
16. **Supported runner commands:** `preflight`, `verify-data`, `download-model`, `smoke`, `zero-shot-dev`, `training-probe`, `objective-run`, `negative-run`, `lr-run`, `final-seed`, `test`, `status`.
17. **Preflight behavior:** verifies Python/platform, data files and hashes, writable artifact namespace, and—unless `--cpu-check`—CUDA, GPU count, and conservative 8-GB minimum. Local `preflight --cpu-check` passed.
18. **Smoke-test behavior:** deterministic first 10 DEV sources / 1,000 pairs; exact revision, 100 candidates/source, finite scalar logits, deterministic ordering, and membership invariants are enforced by the scorer. **Not run locally** because no local model execution is authorized in this milestone.
19. **Resume behavior:** external runs begin from source 0 in a new run namespace; rerunning with the same output root preserves completed hashed chunks and never merges local 10,000-pair chunks.
20. **Chunking behavior:** 50 sources / 5,000 pairs initially; atomic JSONL chunk plus manifest entry with source range, count, SHA-256, revision, FP32 precision, and run/device identity.
21. **FP32 initial policy:** mandatory initial zero-shot external precision; FP16 is not inherited and may be considered only after an external FP32 parity reference.
22. **External batch-size benchmark policy:** bounded engineering benchmark at 8, 16, 32, 64 and optionally 128 if safe; select only by throughput, VRAM, and stability—not retrieval performance. **Not run locally.**
23. **Training-probe behavior:** eventual external BCE and listwise forward/loss/backward probes, list size 8, micro-batches 1/2/4; no optimizer updates. **Not run.**
24. **optimizer.step protection:** handoff policy forbids optimizer updates in feasibility probes; no optimizer update occurred in Step 8.3.
25. **Checkpoint immutability protection:** future probes must hash parameters before/after and assert equality; optimizer step count must remain zero. **External probe not run.**
26. **TEST lock protection:** `test` refuses with `TEST_LOCK_REQUIRED` unless `artifacts/experiments/shift_map_v2_external/test_lock.json` exists. Local protection test passed; final lock was not created.
27. **External artifact-root policy:** `artifacts/experiments/shift_map_v2_external_runs/<run_id>/`; never overwrite local Step 8.1/8.2 artifacts.
28. **Sync-back mechanism:** `scripts/external_step8/export_results.py`; exports compact manifests, metrics, configs, curves, CSVs, and hashes only.
29. **Candidate giant-file transfer requirement:** ER2 manual copy required; candidate gzip files and ignored benchmark JSONL must be copied separately and hash-verified.
30. **Model-weight download policy:** do not package weights; external host downloads the exact Hugging Face revision through normal loading, verifies resolved revision and config/tokenizer/weight hashes, and keeps cache external.
31. **Model checkpoint policy:** trained checkpoints remain ignored/external; compact manifests record checkpoint path, bytes, SHA-256, base revision, config hash, seed, and epoch. No checkpoint is uploaded automatically.
32. **Execution guide path:** `docs/experiments/shift_map_v2_external_execution.md`.
33. **Portable archive status:** no archive created; the handoff is a compact tracked metadata/code bundle plus a large-file copy checklist.
34. **Handoff manifest hash:** `artifacts/experiments/shift_map_v2_external/handoff_manifest.json`; SHA-256 `1a934b2db8e7621f496db4782165c3668ac676496662d2a9a3e595d325ed1bd4`.
35. **Tests added:** `tests/unit/test_external_step8_runner.py` — CLI command inventory, CPU preflight, and TEST lock refusal.
36. **pytest:** `87 passed in 5.41s` before the final report-only amendment; post-amend verification is recorded in the final response.
37. **ruff:** passed after final script formatting.
38. **mypy:** `Success: no issues found in 31 source files`.
39. **Integrity verification:** frozen candidate hashes and benchmark files passed `verify-data`; `git diff --check` passed before final report generation; local partial chunks remain preserved and separate.
40. **Git commit hash/message:** `ce7d16ad04b9424137a655f4cbf2acb06b35a96b` — `chore: prepare SHIFT-MAP v2 external compute handoff`.
41. **Exact Git status:** `## main`; clean working tree after commit.
42. **External readiness classification:** **ER2 — EXTERNAL READY WITH MANUAL LARGE-FILE COPY**.
43. **Remaining manual transfer steps:** clone/copy repository; create compatible `.venv`; install host-compatible PyTorch plus `requirements-step8.txt`; copy the eight listed large/ignored data files; run `verify-data`, `preflight`, model download/revision verification, smoke, then full zero-shot DEV.
44. **Local long-running GPU execution:** explicitly paused. Do not resume full DEV, training probes, scientific ablations, or TEST on this laptop in Step 8.3.
45. **Next milestone:** **STEP 8.4 — EXTERNAL COMPUTE VALIDATION AND ZERO-SHOT DEV COMPLETION**.

## Scientific boundary

Step 8 scientific version remains `2.0`. The frozen candidate membership, K=100, source/target-only model input, max length 96, hypotheses, evaluator, and benchmark are unchanged. The prior local 10,000-pair partial run remains provenance/resumability evidence only; it is not merged into a fresh external scientific score set. Full DEV metrics, training feasibility, objective selection, ablations, final seeds, and TEST are `NOT COMPUTED` in Step 8.3.
