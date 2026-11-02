#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
format_model_json.py

Role: re-format already-extracted JSON into tree-form JSON only.
No normalization checks are performed here; normalization is left to cadjson_quality_check.py.
Normal stdout / --output carries only the tree-form JSON.

Examples:
  python format_model_json.py --input model.json \
    | python cadjson_quality_check.py --output normalized.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence


def read_text(path: str) -> str:
    if not path or path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_json(text: str) -> Any:
    obj = json.loads(text.strip())
    if isinstance(obj, str):
        obj = json.loads(obj)
    return obj


def write_text(text: str, output: str = "") -> None:
    if output:
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Format existing JSON into tree JSON. stdout only contains JSON.")
    parser.add_argument("--input", "-i", default="-", help="JSON file; defaults to stdin")
    parser.add_argument("--output", "-o", default="", help="Tree-form JSON output file; stdout when omitted")
    parser.add_argument("--indent", type=int, default=2, help="Tree-form JSON indentation")
    args = parser.parse_args(argv)

    try:
        obj = load_json(read_text(args.input))
        text = json.dumps(obj, ensure_ascii=False, indent=args.indent) + "\n"
        write_text(text, args.output)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
