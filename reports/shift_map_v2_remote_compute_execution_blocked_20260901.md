# SHIFT-MAP v2 Step 8.4 remote-compute execution record — BLOCKED

## Status

**BLOCKED_PREEXECUTION_IMPLEMENTATION** — a healthy remote GPU environment and frozen-data transfer were established, but the committed Step 8.3 external runner does not implement the scientific Step 8.4 stages required by the current protocol. No MedCPT model snapshot was downloaded, no DEV source was scored, no TEST evaluation was performed, and no optimizer step occurred.

## Remote identity and workspace

- SSH host: `10.80.34.245` (authenticated account `gra_rohan`, hostname `ubuntu22`)
- Remote OS: Ubuntu 22.04.5 LTS
- Workspace: `/home/gra_rohan/shift_icd_temporary_compute/shift_icd_step8_20260901_142359`
- Workspace mode: `700`
- Scratch policy: `REMOTE_BACKUP_POLICY_UNKNOWN`; `/scratch` was unavailable, so a private user-home workspace was used.
- GPU 0: NVIDIA GeForce RTX 4090, 24,564 MiB, compute capability 8.9, driver 565.57.01, idle at inspection; selected GPU 0.
- GPU 1: NVIDIA GeForce RTX 4090, 24,564 MiB, compute capability 8.9, driver 565.57.01, idle at inspection.
- CUDA runtime selected by PyTorch: 12.6
- Remote Python: 3.12.12 (`/home/gra_rohan/ENTER/bin/python`)
- Remote PyTorch: `2.7.1+cu126`; `torch.cuda.is_available() == True`.

## Authoritative local / transferred state

- Branch: `main`
- Remote scientific-code HEAD: `54cf0b613450a2c3c0094bcb881fbb28eea0159c`
- The historical `ce7d16a...` object exists locally but is not an ancestor of current HEAD after the metadata amend.
- The repository was transferred as a shallow exact-HEAD clone, then all manifest-declared candidate/description files were transferred separately.
- Archive SHA-256: `83af6627a69de3f4a132783b4ad97f767fe8bcd0d61a4e857b0152192de10b80`

## Data integrity

All required candidate and description data hashes passed remotely:

- TRAIN candidates: `f632854cd22ff9875390d9ff7280b4033b571c0c41ac54561730bf6f77536a35`
- DEV candidates: `ea7d94aa7f8e2d73d9d07dfff6d21a1b8e9a4a3db7837dd5dfa9f76feb92eafa`
- TEST candidates (hash-only; never scored): `adcdcc2348165723c29f76687e5f487c2e5275dac5ff2389352ff4570d8f961b`
- Candidate manifest and all TRAIN/DEV/TEST description files: PASS.

The benchmark manifest required an explicit separate transfer because the repository archive contained the committed tracked version while the authoritative local working tree contained the manifest hash frozen by `data_manifest.json`. After transfer, remote `verify-data`/CPU preflight passed with the required manifest SHA-256 `5464b14e2857affd3bff3801e6497d4342449976a006325f6189271bfcabfd47`. This left only that data manifest modified in the remote worktree; scientific code remained exact HEAD.

## Quality-gate evidence

- Local before transfer: pytest 87 passed; Ruff passed; mypy passed; package import passed; git diff --check passed; local `verify-data` passed.
- Remote full suite: 83 passed, 4 failed. Three failures require intentionally excluded unrelated historical/raw benchmark artifacts; one required optional `pyarrow`.
- After installing `pyarrow`, targeted Step 8 suite: 19 passed, 1 failed. The remaining failure was the runner CPU preflight before the separately-required benchmark manifest was copied. Direct remote CPU preflight then passed after that transfer.

## Scientific blocker

`scripts/external_step8/run_step8.py` lines 136–144 return `EXTERNAL_STAGE_REQUIRED` for:

- `download-model`
- `training-probe`
- all later scientific-stage commands

Therefore the command did **not** download or verify `ncbi/MedCPT-Cross-Encoder` at revision `71caf65d4927987813984f54c284405a13fcca49`.

`scripts/external_step8/score_dev.py` can write reranked raw-score chunks when given an existing local model path, but it does not implement the required metrics, FP16 parity, batch benchmark, BCE/listwise feasibility probes, checkpoint immutability evidence, runtime projection, or EF1–EF5 classification.

## Result

- MedCPT resolved revision: NOT DOWNLOADED / NOT VERIFIED
- Smoke: NOT RUN
- FP32 / FP16 parity: NOT COMPUTED
- Full zero-shot DEV: NOT STARTED; 0 sources / 0 pairs
- DEV Hit@1/3/5/10/25/50/100, MRR, NDCG@5/10: NOT COMPUTED
- Candidate membership / Hit@100 invariant: NOT SCIENTIFICALLY EVALUATED
- BCE/listwise/training-precision probes: NOT RUN
- Model immutability: NOT APPLICABLE; model not loaded
- `optimizer.step()` count: 0
- TEST scoring: NOT PERFORMED
- EF classification: NOT COMPUTED
- Step 8.5 authorization: NOT AUTHORIZED
- Export package / local sync-back: NOT APPLICABLE; no scientific output exists
- Checkpoints returned: none

## Remote preservation / cleanup

`LOCAL_SYNC_VERIFIED` is **FALSE** because there is no valid Step 8.4 result package to sync. The temporary remote workspace is intentionally preserved for recovery and implementation completion. No filesystem-level cleanup was performed. No claim of secure erasure is made.

The run-scoped SSH public-key entry and local private key are also intentionally retained only while the remote workspace must be preserved; they must be removed as part of verified final cleanup after a valid local result package exists.
