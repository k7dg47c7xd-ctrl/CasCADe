#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rating_raw_structure.py

New scoring rule: compare the raw model output directly with the original reference CAD-JSON.

This file no longer treats "raw structural ability" as the final new score; the new score is:
    raw_reference_score = comparison score between the first raw model output (raw_output) and the original reference JSON pointed to by reference_path

Strict requirements:
- Do not call cadjson_template_framework.normalize_cadjson_text()
- Do not repair
- Do not complete templates
- Do not modify the JSON structure of the model output
- The scorer may only extract JSON fragments from the model text for json.loads()

The legacy scoring rules remain in Rating_modified.py and are called by the main program after normalize / repair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from Rating_modified import score_json_output_with_reference


def _read_text(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk")


def score_raw_structured_output(
    output_text: str,
    requirement: str = "",
    expected_types: Optional[Sequence[str]] = None,
    reference_path: str | Path = "",
    reference_text: Optional[str] = None,
    software_open: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    New scoring: compare the raw model output directly with the original reference CAD-JSON.

    Parameters:
        output_text:
            The raw text returned by the model the first time. Do not pass normalized JSON or repaired output.
        reference_path / reference_text:
            The original reference JSON of this sample, e.g. 00000007_lite.json.

    Returns:
        Full details compatible with the legacy scorer, plus:
        - raw_reference_score: final score of raw output vs original reference JSON
        - scoring_mode: description of the scoring mode
    """
    expected_types = expected_types or []

    detail = score_json_output_with_reference(
        output_text=output_text,
        reference_text=reference_text,
        reference_path=reference_path,
        expected_types=expected_types,
        requirement=requirement,
        software_open=software_open,
    )

    raw_score = detail.get("total_score_without_software", 0)
    detail["raw_reference_score"] = raw_score
    detail["scoring_mode"] = "raw_output_vs_original_reference_json_no_normalize_no_repair"
    detail["normalized_or_repaired_before_scoring"] = False
    return detail


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score raw model output directly against the original reference CAD-JSON."
    )
    parser.add_argument("--input", required=True, help="Model raw-output text file, e.g. raw_output.txt")
    parser.add_argument("--reference", default="", help="Corresponding original reference JSON, e.g. 00000007_lite.json")
    parser.add_argument("--reference-text", default=None, help="Pass the reference JSON text directly; usually not needed")
    parser.add_argument("--requirement", default="", help="Natural-language requirement text")
    parser.add_argument("--requirement-file", default="", help="Natural-language requirement file")
    parser.add_argument("--expected-types", default="", help="Comma-separated expected types, optional")
    parser.add_argument("--output", default="", help="Scoring output JSON file; printed to the terminal when omitted")
    args = parser.parse_args()

    output_text = _read_text(args.input)
    requirement = args.requirement
    if args.requirement_file:
        requirement = _read_text(args.requirement_file)

    expected_types = [x.strip() for x in args.expected_types.replace("，", ",").split(",") if x.strip()]

    result = score_raw_structured_output(
        output_text=output_text,
        requirement=requirement,
        expected_types=expected_types,
        reference_path=args.reference,
        reference_text=args.reference_text,
    )

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
