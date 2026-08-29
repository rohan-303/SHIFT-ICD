# SHIFT-ICD

Publication-grade research foundation for trustworthy clinical terminology mapping under ICD version and ontology shift.

## Build and test

- Install development dependencies: `python -m pip install -e ".[dev]"`
- Run tests: `pytest`
- Run lint: `ruff check .`
- Check package import: `python -c "import shift_icd; print(shift_icd.__version__)"`

## Research conventions

- Build one research stage at a time; do not implement later model stages prematurely.
- Treat raw authoritative data as immutable and record provenance for every resource.
- Never force ambiguous, one-to-many, information-loss, or no-map cases into single labels.
- Keep test data out of model selection and threshold tuning.
- Every proposed feature requires an ablation and reproducible evaluation.
- Distinguish hypotheses, executed results, reproduced results, and validated claims.
- This repository is research software, not a clinically validated decision system.

## Code conventions

- Python 3.11–3.12, src layout, type-aware code, tested transformations.
- Prefer small modules and configuration-driven scripts.
- Match existing style and run tests/lint before declaring a milestone complete.
