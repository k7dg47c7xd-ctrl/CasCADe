"""
cadjson_template_framework.py

Fixed Cadmium-lite CAD-JSON node templates and post-processing normalizer.

This version intentionally does NOT use the IR route. It does not convert
natural language into high-level templates such as box/cylinder/washer.

Intended pipeline:
1. dataset_prompts.py lets the model directly generate a full Cadmium-lite tree.
2. normalize_cadjson_text() post-processes that generated tree.
3. The post-processor keeps geometry semantics where possible, but normalizes:
   - fixed field order: type, features, group, coordinates, parameters
   - fixed group values
   - fixed parameter keys and defaults
   - coordinate vector shape/count
   - primitive feature rules
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

JsonNode = Dict[str, Any]
Point3 = List[float]

FIELD_ORDER = ["type", "features", "group", "coordinates", "parameters"]

ALLOWED_TYPES = {
    "Group", "Fuse", "Cut", "Prism", "Face", "Wire", "Edge", "Arc", "Circle"
}

GROUP_RULES = {
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

DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "Group": {
        "use_result_filter": False,
        "exclude": False,
        "return_index_list": [],
    },
    "Fuse": {},
    "Cut": {},
    "Prism": {
        "height": 1.0,
        "use_height": True,
        "auto_normal_vector": True,
        "combin": False,
    },
    "Face": {
        "combin": False,
        "mode_flag": 3,
        "uv_list": [],
    },
    "Wire": {
        "vertex_input": False,
        "edge_sort": False,
        "linear_to_lerance": 0.000001,
    },
    "Edge": {
        "pline": False,
        "combin": False,
    },
    "Arc": {
        "r": 1.0,
        "alpha1": 0.0,
        "alpha2": 0.0,
        "circle_mode": "0p",
        "arc_mode": "3p",
        "sense": True,
    },
    "Circle": {
        "r": 1.0,
        "wire_type": True,
    },
}

DEFAULT_COORDS: Dict[str, List[Point3]] = {
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

COORD_COUNT = {
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


# -----------------------------------------------------------------------------
# Fixed node template constructors
# -----------------------------------------------------------------------------

def _node(
    type_: str,
    features: List[JsonNode],
    group: str,
    coordinates: List[Point3],
    parameters: Dict[str, Any],
) -> JsonNode:
    """Create a node with the exact dataset field order."""
    return {
        "type": type_,
        "features": features,
        "group": group,
        "coordinates": coordinates,
        "parameters": parameters,
    }


def group(features: Optional[List[JsonNode]] = None) -> JsonNode:
    return _node("Group", features or [], "operator", [], copy.deepcopy(DEFAULT_PARAMS["Group"]))


def fuse(features: List[JsonNode]) -> JsonNode:
    return _node("Fuse", features, "operator", [], copy.deepcopy(DEFAULT_PARAMS["Fuse"]))


def cut(base: JsonNode, cutters: Sequence[JsonNode]) -> JsonNode:
    return _node("Cut", [base] + list(cutters), "operator", [], copy.deepcopy(DEFAULT_PARAMS["Cut"]))


def prism(face_node: JsonNode, height: float = 1.0, coordinates: Optional[List[Point3]] = None) -> JsonNode:
    params = copy.deepcopy(DEFAULT_PARAMS["Prism"])
    params["height"] = float(height)
    return _node(
        "Prism",
        [face_node],
        "operator",
        _normalize_coordinates("Prism", coordinates),
        params,
    )


def face(features: Optional[List[JsonNode]] = None, coordinates: Optional[List[Point3]] = None) -> JsonNode:
    return _node(
        "Face",
        features or [],
        "operator",
        _normalize_coordinates("Face", coordinates),
        copy.deepcopy(DEFAULT_PARAMS["Face"]),
    )


def wire(features: Optional[List[JsonNode]] = None, coordinates: Optional[List[Point3]] = None) -> JsonNode:
    return _node(
        "Wire",
        features or [],
        "operator",
        _normalize_coordinates("Wire", coordinates),
        copy.deepcopy(DEFAULT_PARAMS["Wire"]),
    )


def edge(p1: Point3, p2: Point3) -> JsonNode:
    return _node("Edge", [], "creator", [p1, p2], copy.deepcopy(DEFAULT_PARAMS["Edge"]))


def arc(p1: Point3, p2: Point3, p3: Point3, r: float = 1.0) -> JsonNode:
    params = copy.deepcopy(DEFAULT_PARAMS["Arc"])
    params["r"] = float(r)
    return _node("Arc", [], "creator", [p1, p2, p3], params)


def circle(center: Point3, radius: float = 1.0, normal: Optional[Point3] = None) -> JsonNode:
    normal = normal or [0.0, 0.0, 1.0]
    params = copy.deepcopy(DEFAULT_PARAMS["Circle"])
    params["r"] = float(radius)
    return _node(
        "Circle",
        [],
        "creator",
        [center, [0.0, 0.0, 0.0], normal, [0.0, 0.0, 0.0]],
        params,
    )


def circle_group(center: Point3, radius: float = 1.0, normal: Optional[Point3] = None) -> JsonNode:
    return group([circle(center, radius, normal)])


# -----------------------------------------------------------------------------
# Post-processing normalizer
# -----------------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _as_float(value: Any, default: float) -> float:
    if _is_number(value):
        return float(value)
    try:
        if isinstance(value, str) and value.strip():
            return float(value)
    except Exception:
        pass
    return float(default)


def _as_point3(value: Any, default: Point3) -> Point3:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        converted: Point3 = []
        for item in value:
            if not _is_number(item):
                return copy.deepcopy(default)
            converted.append(float(item))
        return converted
    return copy.deepcopy(default)


def _normalize_coordinates(type_: str, coordinates: Any) -> List[Point3]:
    expected_len = COORD_COUNT[type_]
    defaults = copy.deepcopy(DEFAULT_COORDS[type_])

    if expected_len == 0:
        return []

    if not isinstance(coordinates, list):
        coordinates = []

    result: List[Point3] = []
    for idx in range(expected_len):
        default_point = defaults[idx]
        if idx < len(coordinates):
            result.append(_as_point3(coordinates[idx], default_point))
        else:
            result.append(default_point)
    return result


def _normalize_parameters(type_: str, parameters: Any) -> Dict[str, Any]:
    """Keep only fixed parameter keys and fill missing keys from the template."""
    params = copy.deepcopy(DEFAULT_PARAMS[type_])
    raw = parameters if isinstance(parameters, dict) else {}

    for key in list(params.keys()):
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

    return params


def _unwrap_boolean_solid(node: JsonNode) -> Optional[JsonNode]:
    """Allow a harmless Group wrapper around one boolean-solid child to be removed."""
    node_type = node.get("type")
    if node_type in {"Prism", "Fuse", "Cut"}:
        return node
    if node_type == "Group":
        children = node.get("features", [])
        if isinstance(children, list) and len(children) == 1:
            child = children[0]
            if isinstance(child, dict) and child.get("type") in {"Prism", "Fuse", "Cut"}:
                return child
    return None


def _normalize_features(type_: str, features: Any) -> List[JsonNode]:
    if type_ in {"Edge", "Arc", "Circle"}:
        return []

    raw_children = features if isinstance(features, list) else []
    normalized_children: List[JsonNode] = []

    for raw_child in raw_children:
        if not isinstance(raw_child, dict):
            continue
        child = normalize_cadjson_tree(raw_child)
        child_type = child.get("type")

        if type_ == "Prism":
            if child_type == "Face" and not normalized_children:
                normalized_children.append(child)

        elif type_ == "Face":
            if child_type == "Wire":
                normalized_children.append(child)
            elif child_type == "Group":
                # In dataset shape, a Group inside Face is a Circle wrapper.
                group_children = child.get("features", [])
                group_types = [c.get("type") for c in group_children if isinstance(c, dict)]
                if group_types == ["Circle"]:
                    normalized_children.append(child)
            elif child_type == "Circle":
                normalized_children.append(group([child]))

        elif type_ == "Wire":
            if child_type in {"Edge", "Arc"}:
                normalized_children.append(child)

        elif type_ == "Fuse":
            solid = _unwrap_boolean_solid(child)
            if solid is not None:
                normalized_children.append(solid)

        elif type_ == "Cut":
            solid = _unwrap_boolean_solid(child)
            if solid is not None:
                normalized_children.append(solid)

        elif type_ == "Group":
            normalized_children.append(child)

    return normalized_children


def normalize_cadjson_tree(obj: Any) -> JsonNode:
    """
    Normalize a model-generated Cadmium-lite tree to fixed templates.

    The function is deliberately conservative: it does not infer new modeling
    intent from natural language and does not convert the geometry into box,
    cylinder, washer, or any IR template. It only repairs structural/schema
    variance in the tree already generated by the model.
    """
    if not isinstance(obj, dict):
        return group([])

    type_ = obj.get("type")
    if type_ not in ALLOWED_TYPES:
        type_ = "Group"

    features = _normalize_features(type_, obj.get("features", []))
    coordinates = _normalize_coordinates(type_, obj.get("coordinates", []))
    parameters = _normalize_parameters(type_, obj.get("parameters", {}))
    group_value = GROUP_RULES[type_]

    # The scoring schema expects one Face child for Prism. Add the thinnest
    # possible placeholder only when the model failed to supply a valid Face.
    if type_ == "Prism" and not features:
        features = [face([wire([])])]

    # The scoring schema expects Fuse/Cut to have at least two solid children.
    # If not, degrade to a Group instead of inventing a second solid.
    if type_ == "Fuse" and len(features) < 2:
        return group(features)
    if type_ == "Cut" and len(features) < 2:
        return group(features)

    return _node(type_, features, group_value, coordinates, parameters)


def normalize_cadjson_text(json_text: str) -> str:
    """Parse, normalize, and dump a Cadmium-lite JSON string."""
    obj = json.loads(json_text)
    normalized = normalize_cadjson_tree(obj)
    return json.dumps(normalized, ensure_ascii=False, indent=2)


def normalize_cadjson_file(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        normalized_text = normalize_cadjson_text(f.read())
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(normalized_text)
