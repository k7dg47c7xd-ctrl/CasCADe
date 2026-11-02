# CasCADe repair-stage ablation package

This folder evaluates error-aware CAD-JSON repair on the configured data splits.

Core files:

- `generate.py`: initial generation, repair orchestration, full final evaluation.
- `repair.py`: iterative `(T, y_t, E_t)` repair loop.
- `validation.py`: Syntax -> Schema -> Geometry barrier.
- `schema_check.py`: reference-free CAD-JSON schema checker.
- `geometry_check.py`: OpenCascade reconstruction/topology/Boolean-effect checks.
- `scoring.py`: CAD reference score (reference is evaluation-only).
- `run_ablation.py`: single variant or grouped repair ablations.
- `analyze.py`: validity, repair, and CAD-reference metrics.

Current configured difficulties are `simple` and `complex`; `--difficulty all` reads these dynamically from `config.yaml`.

Before running, verify the paths in `config.yaml`, especially `test_splits`, `merged_model`, and `output_dir`.

For the repair experiments, see `README_REPAIR.md`.
