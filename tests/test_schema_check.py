# -*- coding: utf-8 -*-
"""Offline basic tests for the CAD-JSON schema validator (no torch/OCC)."""
import copy
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ablation"))

from schema_check import validate_cadjson_schema  # noqa: E402

VALID = {
    "type": "Group",
    "group": "operator",
    "coordinates": [[0.0, 0.0, 0.0]],
    "parameters": {},
    "features": [
        {
            "type": "Prism",
            "group": "operator",
            "coordinates": [[0.0, 0.0, 0.0]],
            "parameters": {"height": 0.5},
            "features": [
                {
                    "type": "Face",
                    "group": "operator",
                    "coordinates": [[0.0, 0.0, 0.0]],
                    "parameters": {},
                    "features": [
                        {
                            "type": "Wire",
                            "group": "operator",
                            "coordinates": [[0.0, 0.0, 0.0]],
                            "parameters": {},
                            "features": [
                                {
                                    "type": "Edge",
                                    "group": "creator",
                                    "coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                                    "parameters": {},
                                    "features": [],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    ],
}


def test_valid_cadjson_passes():
    report = validate_cadjson_schema(VALID)
    assert report["valid"] is True, report["errors"]
    assert report["errors"] == []


def test_prism_missing_height_rejected():
    bad = {
        "type": "Group", "group": "operator", "coordinates": [], "parameters": {},
        "features": [
            {"type": "Prism", "group": "operator", "coordinates": [], "parameters": {}, "features": []}
        ],
    }
    report = validate_cadjson_schema(bad)
    assert report["valid"] is False
    codes = {e["code"] for e in report["errors"]}
    assert {"prism_missing_height", "prism_arity"} <= codes


def test_edge_wrong_coordinate_count_rejected():
    bad = copy.deepcopy(VALID)
    bad["features"][0]["features"][0]["features"][0]["features"][0]["coordinates"] = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]
    ]
    report = validate_cadjson_schema(bad)
    codes = {e["code"] for e in report["errors"]}
    assert "edge_coordinate_count" in codes
