# -*- coding: utf-8 -*-
"""CAD-JSON structural/schema validator used by the repair loop.

The checker is deliberately reference-free: it validates only the declared CAD-JSON
contract and local node constraints. Ground-truth JSON is never consulted.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

ALLOWED_TYPES = {"Group", "Fuse", "Cut", "Prism", "Face", "Wire", "Edge", "Arc", "Circle"}
REQUIRED_FIELDS = {"type", "features", "group", "coordinates", "parameters"}
GROUP_RULES = {
    "Group": "operator", "Fuse": "operator", "Cut": "operator",
    "Prism": "operator", "Face": "operator", "Wire": "operator",
    "Edge": "creator", "Arc": "creator", "Circle": "creator",
}

# Conservative parent-child constraints matching the current OCC reconstructor.
ALLOWED_CHILDREN = {
    "Group": ALLOWED_TYPES,
    "Fuse": {"Group", "Fuse", "Cut", "Prism"},
    "Cut": {"Group", "Fuse", "Cut", "Prism"},
    "Prism": {"Face"},
    "Face": {"Wire", "Circle", "Group"},
    "Wire": {"Edge", "Arc", "Circle"},
    "Edge": set(),
    "Arc": set(),
    "Circle": set(),
}


def _err(code: str, path: str, message: str, *, node_type: str = "", severity: str = "error") -> Dict[str, str]:
    return {
        "stage": "schema",
        "code": code,
        "path": path,
        "node_type": node_type,
        "severity": severity,
        "message": message,
    }


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _is_point3(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and len(x) >= 3 and all(_is_number(v) for v in x[:3])


def _check_coordinates(node_type: str, coords: Any, path: str, errors: List[Dict[str, str]]) -> None:
    if not isinstance(coords, list):
        errors.append(_err("coordinates_not_list", path, "coordinates 必须是 list", node_type=node_type))
        return

    for i, p in enumerate(coords):
        if not _is_point3(p):
            errors.append(_err("invalid_point", f"{path}[{i}]", "坐标点必须至少包含 3 个有限数值", node_type=node_type))

    if node_type == "Edge" and len(coords) != 2:
        errors.append(_err("edge_coordinate_count", path, "Edge 必须恰好包含 2 个端点", node_type=node_type))
    elif node_type == "Arc" and len(coords) != 3:
        errors.append(_err("arc_coordinate_count", path, "Arc 必须恰好包含 start/mid/end 三个点", node_type=node_type))
    elif node_type == "Circle" and len(coords) < 1:
        errors.append(_err("circle_missing_center", path, "Circle 至少需要中心点坐标", node_type=node_type))


def _check_parameters(node_type: str, params: Any, path: str, errors: List[Dict[str, str]]) -> None:
    if not isinstance(params, dict):
        errors.append(_err("parameters_not_object", path, "parameters 必须是 JSON object", node_type=node_type))
        return

    if node_type == "Prism":
        if "height" not in params:
            errors.append(_err("prism_missing_height", path, "Prism.parameters 缺少 height", node_type=node_type))
        elif not _is_number(params.get("height")) or abs(float(params["height"])) < 1e-12:
            errors.append(_err("prism_invalid_height", path + ".height", "Prism height 必须是非零有限数值", node_type=node_type))

    if node_type == "Circle":
        if "r" not in params:
            errors.append(_err("circle_missing_radius", path, "Circle.parameters 缺少 r", node_type=node_type))
        elif not _is_number(params.get("r")) or float(params["r"]) <= 0:
            errors.append(_err("circle_invalid_radius", path + ".r", "Circle r 必须大于 0", node_type=node_type))


def _check_cardinality(node_type: str, features: Sequence[Any], path: str, errors: List[Dict[str, str]]) -> None:
    n = len(features)
    if node_type == "Group" and n < 1:
        errors.append(_err("group_empty", path, "Group 至少需要 1 个 feature", node_type=node_type))
    elif node_type == "Fuse" and n < 2:
        errors.append(_err("fuse_arity", path, "Fuse 至少需要 2 个实体", node_type=node_type))
    elif node_type == "Cut" and n < 2:
        errors.append(_err("cut_arity", path, "Cut 至少需要基体和刀具体共 2 个实体", node_type=node_type))
    elif node_type == "Prism" and n != 1:
        errors.append(_err("prism_arity", path, "Prism 必须恰好包含 1 个 Face", node_type=node_type))
    elif node_type == "Face" and n < 1:
        errors.append(_err("face_empty", path, "Face 至少需要 1 个 Wire/Circle 轮廓", node_type=node_type))
    elif node_type == "Wire" and n < 1:
        errors.append(_err("wire_empty", path, "Wire 至少需要 1 个 Edge/Arc/Circle", node_type=node_type))
    elif node_type in {"Edge", "Arc", "Circle"} and n != 0:
        errors.append(_err("creator_has_children", path, f"{node_type} 是 creator，features 必须为空", node_type=node_type))


def validate_cadjson_schema(obj: Any, *, require_exact_fields: bool = True) -> Dict[str, Any]:
    """Validate a parsed CAD-JSON object.

    Returns a structured, repair-friendly report. The result is independent of any
    ground-truth/reference object and is therefore safe to use at test time.
    """
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    node_count = 0

    if not isinstance(obj, dict):
        errors.append(_err("root_not_object", "root", "根对象必须是 JSON object"))
        return {"valid": False, "errors": errors, "warnings": warnings, "node_count": 0}

    if obj.get("type") != "Group":
        errors.append(_err("root_not_group", "root.type", "根节点 type 必须为 Group", node_type=str(obj.get("type", ""))))

    def walk(node: Any, path: str, parent_type: Optional[str]) -> None:
        nonlocal node_count
        if not isinstance(node, dict):
            errors.append(_err("node_not_object", path, "features 中的每个节点必须是 object", node_type=parent_type or ""))
            return
        node_count += 1

        keys = set(node.keys())
        missing = REQUIRED_FIELDS - keys
        extra = keys - REQUIRED_FIELDS
        for k in sorted(missing):
            errors.append(_err("missing_field", path, f"缺少必需字段: {k}", node_type=str(node.get("type", ""))))
        if require_exact_fields:
            for k in sorted(extra):
                errors.append(_err("extra_field", path, f"存在不允许的额外字段: {k}", node_type=str(node.get("type", ""))))

        node_type = node.get("type")
        if not isinstance(node_type, str) or node_type not in ALLOWED_TYPES:
            errors.append(_err("invalid_type", path + ".type", f"type 必须属于 {sorted(ALLOWED_TYPES)}", node_type=str(node_type)))
            # Continue shallow checks where possible, but child rules cannot be trusted.

        if isinstance(node_type, str) and node_type in GROUP_RULES:
            expected_group = GROUP_RULES[node_type]
            if node.get("group") != expected_group:
                errors.append(_err("invalid_group", path + ".group", f"{node_type}.group 必须为 {expected_group!r}", node_type=node_type))

        features = node.get("features")
        if not isinstance(features, list):
            errors.append(_err("features_not_list", path + ".features", "features 必须是 list", node_type=str(node_type)))
            features = []

        coords = node.get("coordinates")
        _check_coordinates(str(node_type), coords, path + ".coordinates", errors)
        _check_parameters(str(node_type), node.get("parameters"), path + ".parameters", errors)

        if isinstance(node_type, str) and node_type in ALLOWED_TYPES:
            _check_cardinality(node_type, features, path + ".features", errors)
            allowed = ALLOWED_CHILDREN[node_type]
            for i, child in enumerate(features):
                if isinstance(child, dict):
                    ct = child.get("type")
                    if isinstance(ct, str) and ct in ALLOWED_TYPES and ct not in allowed:
                        errors.append(_err(
                            "invalid_child_type", f"{path}.features[{i}].type",
                            f"{node_type} 不允许直接包含 {ct}; 允许: {sorted(allowed)}",
                            node_type=node_type,
                        ))
                walk(child, f"{path}.features[{i}]", node_type)
        else:
            for i, child in enumerate(features):
                walk(child, f"{path}.features[{i}]", str(node_type))

    walk(obj, "root", None)
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "node_count": node_count,
    }
