# Reproducibility contract

Every experiment must be reconstructable from a clean checkout, documented environment, immutable raw inputs, versioned scripts, and stored configuration.

## Required controls

- Use deterministic random seeds where possible and record all seeds.
- Drive experiments from configuration files rather than source edits.
- Never use hard-coded absolute machine paths.
- Keep raw data immutable.
- Generate interim and processed datasets through scripts.
- Record dataset versions, hashes, and checksums where available.
- Store the experiment configuration with its results.
- Record operating system, Python, package, and relevant system versions.
- Record exact model names and revisions once models are introduced.
- Record the command required to reproduce each experiment.
- Perform train/dev/test leakage checks.
- Never use the test set for model selection or threshold tuning.
- Implement benchmark split logic programmatically.
- Add unit tests for parsers, canonical mapping reconstruction, and future split generation.

## Evidence levels

Results will be labeled as idea, implemented, executed, reproduced, validated, or externally supported. A README claim or a single notebook output is not equivalent to a clean reproduced run.

## Experiment record

Each future run should retain: configuration, source and processed-data identifiers, split identifier, seed, software environment, model/revision, command, metrics, logs, warnings, and artifact paths.

## Environment lock

`requirements.lock` records the exact lightweight foundation dependency set verified in the local Python 3.11 environment. Recreate with `python -m pip install -r requirements.lock` followed by `python -m pip install -e .`. On the Hermes host, clear the injected `PYTHONPATH` during isolation checks with `env -u PYTHONPATH ...`. Heavy ML dependencies remain intentionally deferred.

## Notebooks

Notebooks are for exploration and visualization. Reusable transformations and evaluation logic must move into tested package modules or scripts. A notebook must not be the sole source of a reported result.

## Statistical reporting

Report uncertainty where appropriate, repeated seeds when feasible, calibration, subgroup/temporal performance, error analysis, and compute/latency if operational claims are made. Negative and failed experiments remain part of the research record.
