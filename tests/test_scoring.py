# -*- coding: utf-8 -*-
"""Offline basic tests for the CAD reference score (scoring.py, no torch/OCC)."""
import copy
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ablation"))
sys.path.insert(0, str(ROOT / "tests"))

from scoring import extract_json_from_text, score_text_against_reference  # noqa: E402
from test_schema_check import VALID  # noqa: E402


def test_identical_json_scores_full_marks():
    res = score_text_against_reference(json.dumps(VALID, ensure_ascii=False), VALID)
    assert res["json_valid"] is True
    assert res["reference_compare_score"] == 100.0


def test_perturbed_parameter_scores_lower():
    ref = copy.deepcopy(VALID)
    gen = copy.deepcopy(VALID)
    gen["features"][0]["parameters"]["height"] = 5.0
    res = score_text_against_reference(json.dumps(gen, ensure_ascii=False), ref)
    assert res["json_valid"] is True
    assert res["parameter_score"] < 20.0
    assert res["reference_compare_score"] < 100.0


def test_extract_json_from_prose():
    text = "Here is the generated result:\n```json\n" + json.dumps(VALID, ensure_ascii=False) + "\n```"
    parsed = json.loads(extract_json_from_text(text))
    assert parsed["type"] == "Group"
