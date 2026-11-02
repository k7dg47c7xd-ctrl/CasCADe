# -*- coding: utf-8 -*-
"""Run repair-stage ablations from config.yaml.

Supports only the difficulties declared in config.yaml and can run the three
paper ablation groups directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

P = Path(__file__).resolve().parent
REPO_ROOT = P.parent


def _resolve(p):
    """Absolute paths pass through; relative paths resolve against the repo root."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)

GROUPS = {
    "repair": ["no_repair", "full_repair"],
    "trigger": ["syntax_only_repair", "syntax_schema_repair", "full_repair"],
    "feedback": ["full_no_error_feedback", "full_generic_feedback", "full_repair"],
}


def _write_summary_csv(out_root: Path, difficulty: str):
    summaries = sorted(out_root.glob(f"{difficulty}/*/summary.json"))
    if not summaries:
        return
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in summaries]
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    out_csv = out_root / f"summary_{difficulty}.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote", out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(P / "config.yaml"))
    ap.add_argument("--difficulty", default="simple", help="simple, complex, or all; must exist in config.test_splits")
    ap.add_argument("--variant", default="", help="Run one named variant")
    ap.add_argument("--group", choices=["repair", "trigger", "feedback", "all"], default="", help="Run one paper ablation group")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    difficulties = list(cfg.get("test_splits", {}).keys())
    if not difficulties:
        raise SystemExit("config.yaml contains no test_splits")
    if args.difficulty == "all":
        selected_difficulties = difficulties
    else:
        if args.difficulty not in difficulties:
            raise SystemExit(f"difficulty={args.difficulty!r} does not exist; choose one of: {difficulties} or all")
        selected_difficulties = [args.difficulty]

    if args.variant and args.group:
        raise SystemExit("--variant and --group are mutually exclusive")
    if args.variant:
        wanted = [args.variant]
    elif args.group:
        if args.group == "all":
            wanted = []
            for g in ["repair", "trigger", "feedback"]:
                for v in GROUPS[g]:
                    if v not in wanted:
                        wanted.append(v)
        else:
            wanted = GROUPS[args.group]
    else:
        wanted = list(cfg["variants"].keys())

    missing = [v for v in wanted if v not in cfg["variants"]]
    if missing:
        raise SystemExit(f"config.yaml is missing variants: {missing}")

    out_root = _resolve(cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    reuse_initial = bool(cfg.get("reuse_initial_outputs", True))

    for difficulty in selected_difficulties:
        dataset = _resolve(cfg["test_splits"][difficulty]) / f"{difficulty}.jsonl"
        if not dataset.exists():
            raise SystemExit(f"Dataset not found: {dataset}")

        cache_path = out_root / "_initial_cache" / f"{difficulty}.jsonl"

        for key in wanted:
            v = cfg["variants"][key]
            out_dir = out_root / difficulty / key
            out_dir.mkdir(parents=True, exist_ok=True)
            model = v.get("model_override") or cfg["merged_model"]
            if model == "base":
                model = cfg["base_model"]
            rounds = v.get("max_repair_rounds", cfg["generation"]["max_repair_rounds"])

            api_base = cfg.get("api_base_url") or ""
            api_model = cfg.get("api_model") or Path(str(model)).name
            cmd = [
                sys.executable,
                str(P / "generate.py"),
                "--dataset", str(dataset),
                "--output_dir", str(out_dir),
                "--variant_name", key,
                "--model_id", api_model,
                "--max_new_tokens", str(cfg["generation"]["max_new_tokens"]),
                "--temperature", str(cfg["generation"]["temperature"]),
                "--top_p", str(cfg["generation"]["top_p"]),
                "--max_repair_rounds", str(rounds),
                "--repair_trigger", v.get("repair_trigger", "full"),
                "--feedback_mode", v.get("feedback_mode", "structured"),
            ]
            if v.get("repair"):
                cmd += ["--repair"]
            if v.get("allow_no_effect_boolean"):
                cmd += ["--allow_no_effect_boolean"]
            if reuse_initial:
                cmd += ["--initial_cache", str(cache_path)]
            if api_base:
                cmd += ["--api_base_url", api_base, "--api_key", cfg.get("api_key", "EMPTY")]
            else:
                cmd += ["--local_model", str(_resolve(model))]
                if v.get("adapter"):
                    cmd += ["--adapter", v["adapter"]]
            if args.limit:
                cmd += ["--limit", str(args.limit)]

            print("RUN:", " ".join(cmd))
            subprocess.run(cmd, check=True)
            subprocess.run([
                sys.executable,
                str(P / "analyze.py"),
                "--input_dir", str(out_dir),
                "--variant", key,
                "--out_summary", str(out_dir / "summary.json"),
            ], check=True)

        _write_summary_csv(out_root, difficulty)


if __name__ == "__main__":
    main()
