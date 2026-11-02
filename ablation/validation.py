# -*- coding: utf-8 -*-
"""Validation barrier: JSON extraction/parse -> schema -> geometry."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from scoring import extract_json_from_text
from schema_check import validate_cadjson_schema
from geometry_check import check_geometry_validity


def _syntax_error(exc: Exception) -> Dict[str, str]:
    return {
        "stage": "syntax",
        "code": "json_parse_failed",
        "path": "root",
        "node_type": "",
        "severity": "error",
        "message": f"{type(exc).__name__}: {exc}",
    }


def validate_for_repair(
    text: str,
    *,
    enable_schema: bool = True,
    enable_geometry: bool = True,
    geometry_no_effect_is_error: bool = True,
) -> Dict[str, Any]:
    """Run the complete repair-time validation barrier.

    Geometry only runs after syntax + schema pass, preventing noisy OCC exceptions
    caused by malformed structure. A failed enabled stage makes ``valid`` false.
    """
    report: Dict[str, Any] = {
        "valid": False,
        "json_valid": False,
        "schema_valid": None,
        "geometry_valid": None,
        "obj": None,
        "normalized_json": "",
        "errors": [],
        "warnings": [],
        "syntax": {},
        "schema": {},
        "geometry": {},
    }

    try:
        obj = json.loads(extract_json_from_text(text))
    except Exception as exc:
        err = _syntax_error(exc)
        report["syntax"] = {"valid": False, "errors": [err]}
        report["errors"] = [err]
        return report

    report["obj"] = obj
    report["json_valid"] = True
    report["syntax"] = {"valid": True, "errors": []}

    if enable_schema:
        schema = validate_cadjson_schema(obj)
        report["schema"] = schema
        report["schema_valid"] = bool(schema["valid"])
        report["errors"].extend(schema.get("errors", []))
        report["warnings"].extend(schema.get("warnings", []))
        if not schema["valid"]:
            return report
    else:
        report["schema_valid"] = None
        report["schema"] = {"valid": None, "errors": [], "warnings": []}

    if enable_geometry:
        geo = check_geometry_validity(obj, no_effect_is_error=geometry_no_effect_is_error)
        report["geometry"] = geo
        report["geometry_valid"] = geo.get("valid")
        report["errors"].extend(geo.get("errors", []))
        report["warnings"].extend(geo.get("warnings", []))
        # OCC unavailable is not silently treated as success.
        if geo.get("valid") is not True:
            return report
    else:
        report["geometry_valid"] = None
        report["geometry"] = {"available": None, "valid": None, "errors": [], "warnings": []}

    report["valid"] = True
    report["normalized_json"] = json.dumps(obj, ensure_ascii=False, indent=2)
    return report


def compact_repair_errors(report: Dict[str, Any], max_errors: int = 24) -> List[Dict[str, str]]:
    """Return concise structured errors suitable for an LLM repair prompt."""
    out: List[Dict[str, str]] = []
    for e in report.get("errors", [])[:max_errors]:
        if isinstance(e, dict):
            out.append({
                "stage": str(e.get("stage", "")),
                "code": str(e.get("code", "")),
                "path": str(e.get("path", "")),
                "message": str(e.get("message", "")),
            })
        else:
            out.append({"stage": "unknown", "code": "error", "path": "", "message": str(e)})
    return out
