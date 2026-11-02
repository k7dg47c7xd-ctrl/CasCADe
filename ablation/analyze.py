# -*- coding: utf-8 -*-
"""Aggregate rows.jsonl with repair-stage metrics."""
import argparse
import json
from pathlib import Path


def _rate(rows, key):
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return None if not vals else round(sum(bool(v) for v in vals) / len(vals), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out_summary", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in (Path(a.input_dir) / "rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)
    d = max(n, 1)
    triggered = [r for r in rows if r.get("repair_triggered", False)]
    trigger_resolved = [r for r in triggered if r.get("trigger_valid_final", False)]
    full_repaired = [r for r in triggered if r.get("validation_valid_final", False)]

    agg = {
        "variant": a.variant,
        "n": n,
        "raw_score": round(sum(r["raw_score"] for r in rows) / d, 2),
        "optimized_score": round(sum(r["optimized_score"] for r in rows) / d, 2),
        "json_valid_raw_rate": _rate(rows, "json_valid_raw"),
        "json_valid_final_rate": _rate(rows, "json_valid_final"),
        "schema_valid_raw_rate": _rate(rows, "schema_valid_raw"),
        "schema_valid_final_rate": _rate(rows, "schema_valid_final"),
        "geometry_valid_raw_rate": _rate(rows, "geometry_valid_raw"),
        "geometry_valid_final_rate": _rate(rows, "geometry_valid_final"),
        "overall_valid_raw_rate": _rate(rows, "validation_valid_raw"),
        "overall_valid_final_rate": _rate(rows, "validation_valid_final"),
        "repair_trigger_rate": round(len(triggered) / d, 4),
        "trigger_resolution_rate": None if not triggered else round(len(trigger_resolved) / len(triggered), 4),
        "repair_success_rate": None if not triggered else round(len(full_repaired) / len(triggered), 4),
        "avg_repair_rounds": round(sum(r.get("repair_rounds", 0) for r in rows) / d, 2),
    }
    for s in [
        "type_count_score",
        "preorder_type_score",
        "hierarchy_score",
        "coordinate_score",
        "parameter_score",
        "boolean_operation_score",
    ]:
        agg[s] = round(sum((r["detail_final"].get(s, 0) or 0) for r in rows) / d, 2)

    Path(a.out_summary).write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
