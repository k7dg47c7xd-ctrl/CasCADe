# -*- coding: utf-8 -*-
"""Aggregate the summary.json of every difficulty/variant under outputs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "outputs"
OUT = OUT_ROOT / "ablation_summary.csv"

rows = []
for summary_path in sorted(OUT_ROOT.glob("*/*/summary.json")):
    difficulty = summary_path.parent.parent.name
    d = json.loads(summary_path.read_text(encoding="utf-8"))
    d["difficulty"] = difficulty
    rows.append(d)

if rows:
    keys = ["variant", "difficulty"]
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, "rows", len(rows))
else:
    print("no summary.json found under", OUT_ROOT)
