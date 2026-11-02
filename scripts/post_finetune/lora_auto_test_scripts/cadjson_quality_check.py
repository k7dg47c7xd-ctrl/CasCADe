#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cadjson_quality_check.py

Does exactly one thing: receive a CAD-JSON that has already been extracted and reorganized into tree form, run normalization checks,
and then emit only the final normalized JSON.

It does not:
- call a model;
- extract JSON from chat text;
- generate scoring reports, CSV, or summaries;
- print extra OK/FAIL content to stdout.

Normal stdout / --output contains only the final normalized JSON.
Error messages are written to stderr only.

Usage:
  # Read from a file, write to stdout
  python cadjson_quality_check.py --input output_tree.json

  # Read from stdin, suitable for script pipelines
  python format_model_json.py --input model.json | python cadjson_quality_check.py

  # Write to a file
  python cadjson_quality_check.py --input output_tree.json --output normalized.json

  # Batch JSONL, one tree-form CAD-JSON per line, output JSONL
  python cadjson_quality_check.py --input trees.jsonl --output normalized.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


ALLOWED_TYPES: Set[str] = {
    "Group", "Fuse", "Cut", "Prism", "Face", "Wire", "Edge", "Arc", "Circle"
}

REQUIRED_KEY_ORDER: List[str] = ["type", "features", "group", "coordinates", "parameters"]
REQUIRED_KEYS: Set[str] = set(REQUIRED_KEY_ORDER)

GROUP_RULES: Dict[str, str] = {
    "Group": "operator",
    "Fuse": "operator",
    "Cut": "operator",
    "Prism": "operator",
    "Face": "operator",
    "Wire": "operator",
    "Edge": "creator",
    "Arc": "creator",
    "Circle": "creator",
}

COORD_COUNT_RULES: Dict[str, int] = {
    "Group": 0,
    "Fuse": 0,
    "Cut": 0,
    "Prism": 2,
    "Face": 3,
    "Wire": 3,
    "Edge": 2,
    "Arc": 3,
    "Circle": 4,
}

PARAM_KEYS: Dict[str, Set[str]] = {
    "Group": {"use_result_filter", "exclude", "return_index_list"},
    "Fuse": set(),
    "Cut": set(),
    "Prism": {"height", "use_height", "auto_normal_vector", "combin"},
    "Face": {"combin", "mode_flag", "uv_list"},
    "Wire": {"vertex_input", "edge_sort", "linear_to_lerance"},
    "Edge": {"pline", "combin"},
    "Arc": {"r", "alpha1", "alpha2", "circle_mode", "arc_mode", "sense"},
    "Circle": {"r", "wire_type"},
}

DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "Group": {"use_result_filter": False, "exclude": False, "return_index_list": []},
    "Fuse": {},
    "Cut": {},
    "Prism": {"height": 1.0, "use_height": True, "auto_normal_vector": True, "combin": False},
    "Face": {"combin": False, "mode_flag": 3, "uv_list": []},
    "Wire": {"vertex_input": False, "edge_sort": False, "linear_to_lerance": 0.000001},
    "Edge": {"pline": False, "combin": False},
    "Arc": {"r": 1.0, "alpha1": 0.0, "alpha2": 0.0, "circle_mode": "0p", "arc_mode": "3p", "sense": True},
    "Circle": {"r": 1.0, "wire_type": True},
}

DEFAULT_COORDS: Dict[str, List[List[float]]] = {
    "Group": [],
    "Fuse": [],
    "Cut": [],
    "Prism": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    "Face": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    "Wire": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    "Edge": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    "Arc": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]],
    "Circle": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
}

SOLID_TYPES = {"Prism", "Fuse", "Cut"}
CREATOR_TYPES = {"Edge", "Arc", "Circle"}


class CadJsonError(ValueError):
    pass


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def is_3d_point(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(is_number(x) for x in value)


def _as_float(value: Any, default: float) -> float:
    if is_number(value):
        return float(value)
    try:
        f = float(value)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return default


def _as_point3(value: Any, default: List[float]) -> List[float]:
    if isinstance(value, list) and len(value) == 3:
        return [_as_float(value[0], default[0]), _as_float(value[1], default[1]), _as_float(value[2], default[2])]
    return copy.deepcopy(default)


def _normalize_coordinates(type_: str, coordinates: Any) -> List[List[float]]:
    expected_len = COORD_COUNT_RULES[type_]
    defaults = copy.deepcopy(DEFAULT_COORDS[type_])
    if expected_len == 0:
        return []
    if not isinstance(coordinates, list):
        coordinates = []
    result: List[List[float]] = []
    for idx in range(expected_len):
        default_point = defaults[idx]
        if idx < len(coordinates):
            result.append(_as_point3(coordinates[idx], default_point))
        else:
            result.append(default_point)
    return result


def _normalize_parameters(type_: str, parameters: Any) -> Dict[str, Any]:
    raw = parameters if isinstance(parameters, dict) else {}
    params = copy.deepcopy(DEFAULT_PARAMS[type_])

    for key in params.keys():
        if key in raw:
            params[key] = raw[key]

    if type_ == "Group":
        params["use_result_filter"] = _as_bool(params.get("use_result_filter"), False)
        params["exclude"] = _as_bool(params.get("exclude"), False)
        if not isinstance(params.get("return_index_list"), list):
            params["return_index_list"] = []
    elif type_ == "Prism":
        params["height"] = _as_float(params.get("height"), 1.0)
        params["use_height"] = True
        params["auto_normal_vector"] = True
        params["combin"] = False
    elif type_ == "Face":
        params["combin"] = False
        params["mode_flag"] = 3
        if not isinstance(params.get("uv_list"), list):
            params["uv_list"] = []
    elif type_ == "Wire":
        params["vertex_input"] = False
        params["edge_sort"] = False
        params["linear_to_lerance"] = _as_float(params.get("linear_to_lerance"), 0.000001)
    elif type_ == "Edge":
        params["pline"] = False
        params["combin"] = False
    elif type_ == "Arc":
        params["r"] = _as_float(params.get("r"), 1.0)
        params["alpha1"] = _as_float(params.get("alpha1"), 0.0)
        params["alpha2"] = _as_float(params.get("alpha2"), 0.0)
        params["circle_mode"] = "0p"
        params["arc_mode"] = "3p"
        params["sense"] = _as_bool(params.get("sense"), True)
    elif type_ == "Circle":
        params["r"] = _as_float(params.get("r"), 1.0)
        params["wire_type"] = True
    elif type_ in {"Fuse", "Cut"}:
        params = {}

    return params


def make_node(type_: str, features: Optional[List[Dict[str, Any]]] = None, coordinates: Any = None, parameters: Any = None) -> Dict[str, Any]:
    return {
        "type": type_,
        "features": features or [],
        "group": GROUP_RULES[type_],
        "coordinates": _normalize_coordinates(type_, coordinates if coordinates is not None else DEFAULT_COORDS[type_]),
        "parameters": _normalize_parameters(type_, parameters if parameters is not None else DEFAULT_PARAMS[type_]),
    }


def _child_type(child: Any) -> str:
    if isinstance(child, dict) and isinstance(child.get("type"), str):
        return child["type"]
    return ""


def _is_circle_wrapper_group(node: Any) -> bool:
    if not isinstance(node, dict) or node.get("type") != "Group":
        return False
    features = node.get("features")
    return isinstance(features, list) and len(features) == 1 and _child_type(features[0]) == "Circle"


def _unwrap_single_solid(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if node.get("type") in SOLID_TYPES:
        return node
    if node.get("type") == "Group":
        features = node.get("features")
        if isinstance(features, list) and len(features) == 1 and isinstance(features[0], dict):
            if features[0].get("type") in SOLID_TYPES:
                return features[0]
    return None


def normalize_cadjson_tree(obj: Any, parent_type: str = "", is_root: bool = False) -> Dict[str, Any]:
    """
    Normalize a model output that is already a JSON tree:
    - fix the field order;
    - correct group;
    - fill in / regularize coordinates and parameters;
    - remove child nodes that are not allowed at that level;
    - wrap a Circle that appears directly under Face as Group -> Circle.

    Note: this does not extract JSON from chat text and does not rewrite the model output content.
    """
    if not isinstance(obj, dict):
        raise CadJsonError("Input must be a JSON object in tree-form CAD-JSON")

    type_ = obj.get("type")
    if type_ not in ALLOWED_TYPES:
        raise CadJsonError(f"Invalid node type: {type_!r}")

    raw_features = obj.get("features")
    if not isinstance(raw_features, list):
        raw_features = []

    children: List[Dict[str, Any]] = []
    if type_ not in CREATOR_TYPES:
        for raw_child in raw_features:
            if not isinstance(raw_child, dict):
                continue
            try:
                child = normalize_cadjson_tree(raw_child, parent_type=type_, is_root=False)
            except CadJsonError:
                continue

            child_type = child.get("type")
            if type_ == "Prism":
                if child_type == "Face" and not children:
                    children.append(child)
            elif type_ == "Face":
                if child_type == "Wire":
                    children.append(child)
                elif child_type == "Circle":
                    children.append(make_node("Group", [child]))
                elif _is_circle_wrapper_group(child):
                    children.append(child)
            elif type_ == "Wire":
                if child_type in {"Edge", "Arc"}:
                    children.append(child)
            elif type_ in {"Fuse", "Cut"}:
                solid = _unwrap_single_solid(child)
                if solid is not None:
                    children.append(solid)
            elif type_ == "Group":
                # Root Group / plain Group must not contain Edge/Arc/Circle directly.
                # A Group inside Face may only wrap a Circle.
                if parent_type == "Face":
                    if _is_circle_wrapper_group(child):
                        children.append(child)
                else:
                    if child_type not in CREATOR_TYPES:
                        children.append(child)

    node = make_node(
        type_,
        features=children,
        coordinates=obj.get("coordinates"),
        parameters=obj.get("parameters"),
    )

    # When a boolean node has too few sub-entities, a second entity cannot be invented; degrade to Group and keep the existing entities.
    if type_ in {"Fuse", "Cut"} and len(children) < 2:
        return make_node("Group", children)

    if is_root and node["type"] != "Group":
        # The root must be a Group. Non-Group roots only get an outer wrapper; internal geometry is unchanged.
        return make_node("Group", [node])

    return node


def _check_param_values(node_type: str, params: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    def require_bool(key: str, expected: Optional[bool] = None) -> None:
        value = params.get(key)
        if not isinstance(value, bool):
            errors.append(f"parameters.{key} must be bool")
        elif expected is not None and value is not expected:
            errors.append(f"parameters.{key} must be {expected}")

    def require_number(key: str) -> None:
        if not is_number(params.get(key)):
            errors.append(f"parameters.{key} must be a finite number")

    def require_list(key: str) -> None:
        if not isinstance(params.get(key), list):
            errors.append(f"parameters.{key} must be a list")

    if node_type == "Group":
        require_bool("use_result_filter", False)
        require_bool("exclude", False)
        require_list("return_index_list")
    elif node_type in {"Fuse", "Cut"}:
        if params != {}:
            errors.append("Fuse/Cut parameters must be {}")
    elif node_type == "Prism":
        require_number("height")
        require_bool("use_height", True)
        require_bool("auto_normal_vector", True)
        require_bool("combin", False)
    elif node_type == "Face":
        require_bool("combin", False)
        if params.get("mode_flag") != 3:
            errors.append("parameters.mode_flag must be 3")
        require_list("uv_list")
    elif node_type == "Wire":
        require_bool("vertex_input", False)
        require_bool("edge_sort", False)
        require_number("linear_to_lerance")
    elif node_type == "Edge":
        require_bool("pline", False)
        require_bool("combin", False)
    elif node_type == "Arc":
        require_number("r")
        require_number("alpha1")
        require_number("alpha2")
        if params.get("circle_mode") != "0p":
            errors.append("parameters.circle_mode must be '0p'")
        if params.get("arc_mode") != "3p":
            errors.append("parameters.arc_mode must be '3p'")
        require_bool("sense", True)
    elif node_type == "Circle":
        require_number("r")
        require_bool("wire_type", True)

    return errors


def validate_cadjson_tree(obj: Any) -> List[str]:
    errors: List[str] = []

    def walk(node: Any, path: str, parent_type: str = "", is_root: bool = False) -> None:
        if not isinstance(node, dict):
            errors.append(f"{path}: node must be an object")
            return

        keys = list(node.keys())
        if set(keys) != REQUIRED_KEYS:
            errors.append(f"{path}: fields must be exactly {REQUIRED_KEY_ORDER}, got {keys}")
        if keys != REQUIRED_KEY_ORDER:
            errors.append(f"{path}: field order must be {REQUIRED_KEY_ORDER}, got {keys}")

        node_type = node.get("type")
        if node_type not in ALLOWED_TYPES:
            errors.append(f"{path}: invalid type {node_type!r}")
            return

        if is_root and node_type != "Group":
            errors.append(f"{path}: root node type must be Group")

        if node.get("group") != GROUP_RULES[node_type]:
            errors.append(f"{path}: {node_type}.group must be {GROUP_RULES[node_type]!r}")

        features = node.get("features")
        if not isinstance(features, list):
            errors.append(f"{path}: features must be a list")
            features = []

        coordinates = node.get("coordinates")
        expected_coord_count = COORD_COUNT_RULES[node_type]
        if not isinstance(coordinates, list):
            errors.append(f"{path}: coordinates must be a list")
        else:
            if len(coordinates) != expected_coord_count:
                errors.append(f"{path}: {node_type}.coordinates must have {expected_coord_count} points, got {len(coordinates)}")
            if expected_coord_count == 0 and coordinates != []:
                errors.append(f"{path}: {node_type}.coordinates must be []")
            if expected_coord_count > 0 and not all(is_3d_point(p) for p in coordinates):
                errors.append(f"{path}: {node_type}.coordinates must all be 3D numeric points")

        parameters = node.get("parameters")
        if not isinstance(parameters, dict):
            errors.append(f"{path}: parameters must be an object")
            parameters = {}
        else:
            expected_keys = PARAM_KEYS[node_type]
            if set(parameters.keys()) != expected_keys:
                errors.append(f"{path}: {node_type}.parameters fields must be {sorted(expected_keys)}, got {sorted(parameters.keys())}")
            errors.extend(f"{path}: {e}" for e in _check_param_values(node_type, parameters))

        child_types = [_child_type(child) for child in features]
        if any(not isinstance(child, dict) for child in features):
            errors.append(f"{path}: features may contain only object child nodes")

        if node_type in CREATOR_TYPES:
            if features:
                errors.append(f"{path}: {node_type}.features must be empty")
        elif node_type == "Prism":
            if len(features) != 1 or child_types != ["Face"]:
                errors.append(f"{path}: Prism must contain exactly one Face")
        elif node_type == "Face":
            if not features:
                errors.append(f"{path}: Face must contain at least one Wire or Circle-wrapper Group")
            for i, child in enumerate(features):
                ct = _child_type(child)
                if ct == "Wire":
                    pass
                elif ct == "Group" and _is_circle_wrapper_group(child):
                    pass
                else:
                    errors.append(f"{path}.features[{i}]: Face children must be Wire or Circle-wrapper Group, got {ct!r}")
        elif node_type == "Wire":
            if not features:
                errors.append(f"{path}: Wire must contain at least one Edge or Arc")
            for i, ct in enumerate(child_types):
                if ct not in {"Edge", "Arc"}:
                    errors.append(f"{path}.features[{i}]: Wire children must be Edge or Arc, got {ct!r}")
        elif node_type in {"Fuse", "Cut"}:
            if len(features) < 2:
                errors.append(f"{path}: {node_type} must contain at least two entity child nodes")
            for i, ct in enumerate(child_types):
                if ct not in SOLID_TYPES:
                    errors.append(f"{path}.features[{i}]: {node_type} children must be Prism/Fuse/Cut, got {ct!r}")
        elif node_type == "Group":
            if is_root and not features:
                errors.append(f"{path}: root Group must contain at least one child node")
            if parent_type == "Face" and not _is_circle_wrapper_group(node):
                errors.append(f"{path}: a Group inside Face may only wrap one Circle")
            if parent_type != "Face":
                for i, ct in enumerate(child_types):
                    if ct in CREATOR_TYPES:
                        errors.append(f"{path}.features[{i}]: Group must not directly contain creator node {ct!r}")

        for i, child in enumerate(features):
            walk(child, f"{path}.features[{i}]", parent_type=node_type, is_root=False)

    walk(obj, "root", is_root=True)
    return errors


def load_json_text(text: str) -> Any:
    text = text.strip()
    if not text:
        raise CadJsonError("Input is empty")
    try:
        obj = json.loads(text)
    except Exception as exc:
        raise CadJsonError(f"Input is not valid JSON. Extract/organize it into tree-form JSON first with the bundled scripts. Original error: {exc}") from exc
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception as exc:
            raise CadJsonError(f"Input is a string, but its content is not valid JSON: {exc}") from exc
    return obj


def normalize_and_validate(obj: Any) -> Dict[str, Any]:
    normalized = normalize_cadjson_tree(obj, is_root=True)
    errors = validate_cadjson_tree(normalized)
    if errors:
        preview = "\n".join(errors[:20])
        more = f"\n... {len(errors) - 20} more error(s)" if len(errors) > 20 else ""
        raise CadJsonError("After normalization the output still violates the CAD-JSON rules:\n" + preview + more)
    return normalized


def normalize_text(text: str) -> Dict[str, Any]:
    return normalize_and_validate(load_json_text(text))


def read_text(path: Optional[str]) -> str:
    if not path or path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def write_output(text: str, output: Optional[str]) -> None:
    if output:
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")


def dumps_json(obj: Any, indent: int) -> str:
    if indent > 0:
        return json.dumps(obj, ensure_ascii=False, indent=indent) + "\n"
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


def looks_like_jsonl(path: Optional[str], text: str, forced_jsonl: bool) -> bool:
    if forced_jsonl:
        return True
    if path and path != "-" and Path(path).suffix.lower() == ".jsonl":
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(lines) > 1 and all(line.startswith("{") and line.endswith("}") for line in lines)


def process_jsonl(text: str, indent: int) -> str:
    output_lines: List[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            normalized = normalize_text(line)
        except Exception as exc:
            raise CadJsonError(f"JSONL line {line_no} failed: {exc}") from exc
        # JSONL must be one object per line; force compact output.
        output_lines.append(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(output_lines) + ("\n" if output_lines else "")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize and validate tree CAD-JSON. stdout only contains final normalized JSON.")
    parser.add_argument("--input", "-i", default="-", help="Tree-form CAD-JSON file; defaults to stdin")
    parser.add_argument("--output", "-o", default="", help="Final normalized JSON output file; stdout when omitted")
    parser.add_argument("--indent", type=int, default=2, help="Output indentation. 0 means compact JSON; JSONL is always compact")
    parser.add_argument("--jsonl", action="store_true", help="Treat the input as JSONL, one CAD-JSON per line")
    args = parser.parse_args(argv)

    try:
        text = read_text(args.input)
        if looks_like_jsonl(args.input, text, args.jsonl):
            output_text = process_jsonl(text, args.indent)
        else:
            normalized = normalize_text(text)
            output_text = dumps_json(normalized, args.indent)
        write_output(output_text, args.output or None)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
