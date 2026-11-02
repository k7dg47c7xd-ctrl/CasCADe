# -*- coding: utf-8 -*-
"""Combine description.txt + _lite.json pairs into an evaluation JSONL."""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--difficulty", default="simple", choices=["simple", "medium", "complex"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="0 means all samples")
    args = ap.parse_args()

    root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for desc_path in sorted(root.rglob("*_description.txt")):
        json_path = desc_path.with_name(desc_path.name.replace("_description.txt", "_lite.json"))
        if not json_path.exists():
            continue
        rows.append({"id": desc_path.stem.replace("_description", ""),
                     "description": desc_path.read_text(encoding="utf-8").strip(),
                     "reference_path": str(json_path)})

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.limit > 0:
        rows = rows[: args.limit]

    out = out_dir / f"{args.difficulty}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
