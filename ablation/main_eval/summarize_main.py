# -*- coding: utf-8 -*-
"""Aggregate the per-variant rows.jsonl files produced by run_main.py into main_results.csv.

The metrics mirror ablation/analyze.py. All paths are resolved relative to this
file, so the script can be run from any working directory.
"""
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "outputs" / "main_results"
OUT_CSV = SCRIPT_DIR / "main_results.csv"


def rate(rows, key):
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return None if not vals else round(sum(bool(v) for v in vals) / len(vals), 4)


def main():
    csv_rows = []
    for variant_dir in sorted(RESULTS_ROOT.iterdir()):
        if not variant_dir.is_dir():
            continue
        jsonl = variant_dir / "rows.jsonl"
        if not jsonl.exists():
            print(f"skip: {variant_dir.name}, no rows.jsonl")
            continue
        rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            print(f"skip: {variant_dir.name}, rows.jsonl is empty")
            continue
        csv_rows.append({
            "method": variant_dir.name,
            "overall": rate(rows, "validation_valid_final"),
            "geometry": rate(rows, "geometry_valid_final"),
            "score": round(sum(r.get("optimized_score", 0) for r in rows) / len(rows), 2),
        })

    df = pd.DataFrame(csv_rows)
    df.to_csv(OUT_CSV, index=False)
    print(df)
    print("CSV written to:", OUT_CSV)


if __name__ == "__main__":
    main()
