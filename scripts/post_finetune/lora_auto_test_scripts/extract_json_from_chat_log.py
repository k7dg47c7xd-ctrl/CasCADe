#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_json_from_chat_log.py

Role: extract JSON from model chat text / raw output only, and emit tree-form JSON.
No normalization checks are performed here; normalization is left to cadjson_quality_check.py.
Normal stdout / --output carries only the tree-form JSON; no [OK]/[FAIL] lines are printed.

Examples:
  python extract_json_from_chat_log.py --input model_output.txt \
    | python cadjson_quality_check.py --output normalized.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def strip_markdown_and_prefix(text: str) -> str:
    text = ANSI_RE.sub("", text or "").strip()
    for prefix in ["Assistant:", "assistant:", "ASSISTANT:", "模型输出：", "模型输出:", "输出：", "输出:"]:  # Chinese prefixes are kept for compatibility with legacy model logs
        if prefix in text:
            text = text.rsplit(prefix, 1)[1].strip()
    text = re.sub(r"```(?:json|JSON|text|plaintext)?", "", text)
    return text.replace("```", "").strip()


def extract_first_json(text: str) -> str:
    text = strip_markdown_and_prefix(text)
    candidates = []
    for i, ch in enumerate(text):
        if ch == "{":
            candidates.append((i, "{", "}"))
        elif ch == "[":
            candidates.append((i, "[", "]"))
    if not candidates:
        raise ValueError("No JSON start marker { or [ found")

    last_error = None
    for start, open_char, close_char in candidates:
        stack: List[str] = []
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_char:
                stack.append(ch)
            elif ch == close_char:
                if not stack:
                    break
                stack.pop()
                if not stack:
                    json_text = text[start:i + 1]
                    try:
                        json.loads(json_text)
                        return json_text
                    except Exception as exc:
                        last_error = exc
                        break
    if last_error:
        raise ValueError(f"A JSON-like fragment was found, but json.loads failed to parse it: {last_error}")
    raise ValueError("JSON start marker found but no complete end marker; the model output may be truncated")


def parse_json_from_text(text: str) -> Any:
    obj = json.loads(extract_first_json(text))
    if isinstance(obj, str):
        obj = json.loads(obj)
    return obj


def read_text(path: str) -> str:
    if not path or path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8", errors="ignore")


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
    parser = argparse.ArgumentParser(description="Extract tree JSON from model raw chat output. stdout only contains JSON.")
    parser.add_argument("--input", "-i", default="-", help="Model raw-output file; defaults to stdin")
    parser.add_argument("--output", "-o", default="", help="Tree-form JSON output file; stdout when omitted")
    parser.add_argument("--indent", type=int, default=2, help="Tree-form JSON indentation")
    args = parser.parse_args(argv)

    try:
        obj = parse_json_from_text(read_text(args.input))
        text = json.dumps(obj, ensure_ascii=False, indent=args.indent) + "\n"
        write_text(text, args.output)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
