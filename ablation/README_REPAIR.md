# Error-aware CAD-JSON repair ablation

The runtime loop is:

`LLM -> JSON Syntax -> Schema -> Geometry -> Pass/Reject -> E_t -> Repair -> Validation`

## Important experiment design

Repair triggering and final evaluation are separated:

- `syntax_only_repair`: only syntax errors trigger repair.
- `syntax_schema_repair`: syntax/schema errors trigger repair.
- `full_repair`: syntax/schema/geometry errors trigger repair.
- **Every variant is finally evaluated with full Syntax + Schema + Geometry**, so JSON/Schema/Geometry/Overall validity rates are directly comparable.

Ground-truth JSON is used only for the CAD reference score and is never passed into the repair prompt.

`reuse_initial_outputs: true` stores one initial output per sample in `outputs/_initial_cache/` and reuses it across variants, ensuring the ablation starts from the same `y0`.

## Three paper ablation groups

```bash
python run_ablation.py --group repair --difficulty all
python run_ablation.py --group trigger --difficulty all
python run_ablation.py --group feedback --difficulty all
```

With the current config, `all` means exactly the keys under `test_splits` (simple and complex).

Quick smoke test:

```bash
python run_ablation.py --group all --difficulty all --limit 2
```

## Required environment

Geometry validation requires OpenCascade / pythonocc-core. If OCC is unavailable, full geometry validation intentionally fails with `occ_unavailable` instead of silently passing.

Conda is usually the most reliable installation route:

```bash
conda install -c conda-forge pythonocc-core
```

## Output

Each variant writes:

- `rows.jsonl`
- `summary.json`

Per-difficulty CSV summaries are written to `outputs/summary_simple.csv` and `outputs/summary_complex.csv`; `summarize_all.py` writes `outputs/ablation_summary.csv`.
