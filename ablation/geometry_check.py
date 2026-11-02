# -*- coding: utf-8 -*-
"""Repair-time geometric/topological validation for CAD-JSON.

Compared with the original post-hoc checker, this version:
1) records per-node build/validity checks with JSON paths;
2) validates Wire closure, Face/Prism B-Rep validity and positive solid volume;
3) checks Fuse/Cut effectiveness and can promote no-effect booleans to hard errors;
4) returns structured errors suitable for direct LLM repair feedback.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

try:
    from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Ax2, gp_Dir, gp_Circ
    from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
    from OCC.Core.GC import GC_MakeArcOfCircle
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop_VolumeProperties
    OCC_AVAILABLE = True
    OCC_IMPORT_ERROR = ""
except Exception as _occ_exc:  # pragma: no cover
    OCC_AVAILABLE = False
    OCC_IMPORT_ERROR = f"{type(_occ_exc).__name__}: {_occ_exc}"

EPS = 1e-9


def _issue(code: str, path: str, message: str, *, node_type: str = "", severity: str = "error") -> Dict[str, str]:
    return {
        "stage": "geometry",
        "code": code,
        "path": path,
        "node_type": node_type,
        "severity": severity,
        "message": message,
    }


def _extract_json_text(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    s, e = text.find("{"), text.rfind("}")
    return text[s:e + 1].strip() if s != -1 and e > s else text


def _as_obj(text_or_obj: Any) -> Any:
    if isinstance(text_or_obj, (dict, list)):
        return text_or_obj
    return json.loads(_extract_json_text(str(text_or_obj)))


def _pnt(coord: Any):
    vals = list(coord)
    return gp_Pnt(float(vals[0]), float(vals[1]), float(vals[2]))


def _vec(coord: Any):
    p = _pnt(coord)
    return gp_Vec(p.X(), p.Y(), p.Z())


def _volume(shape) -> float:
    props = GProp_GProps()
    brepgprop_VolumeProperties(shape, props)
    return float(props.Mass())


def _brep_valid(shape) -> bool:
    return bool(BRepCheck_Analyzer(shape).IsValid())


def _record(checks: List[Dict[str, Any]], path: str, node_type: str, valid: bool, detail: str = "") -> None:
    checks.append({"path": path, "node_type": node_type, "valid": bool(valid), "detail": detail})


def _build_edge(node: Dict[str, Any], path: str, checks: List[Dict[str, Any]]):
    coords = node.get("coordinates") or []
    if len(coords) != 2:
        raise ValueError("Edge 必须恰好包含两个端点")
    p0, p1 = _pnt(coords[0]), _pnt(coords[1])
    if p0.Distance(p1) <= EPS:
        raise ValueError("Edge 两端点重合")
    edge = BRepBuilderAPI_MakeEdge(p0, p1).Edge()
    ok = _brep_valid(edge)
    _record(checks, path, "Edge", ok, "nonzero edge")
    if not ok:
        raise ValueError("Edge B-Rep 无效")
    return edge


def _build_arc(node: Dict[str, Any], path: str, checks: List[Dict[str, Any]]):
    coords = node.get("coordinates") or []
    if len(coords) != 3:
        raise ValueError("Arc 必须包含 start/mid/end 三点")
    arc = GC_MakeArcOfCircle(_pnt(coords[0]), _pnt(coords[1]), _pnt(coords[2]))
    if not arc.IsDone():
        raise ValueError("Arc 三点无法构造圆弧（可能共线或重合）")
    edge = BRepBuilderAPI_MakeEdge(arc.Value()).Edge()
    ok = _brep_valid(edge)
    _record(checks, path, "Arc", ok, "arc build")
    if not ok:
        raise ValueError("Arc B-Rep 无效")
    return edge


def _circle_axis(node: Dict[str, Any]):
    coords = node.get("coordinates") or []
    params = node.get("parameters") or {}
    radius = float(params.get("r", 0.0) or 0.0)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Circle 半径 r 必须大于 0")
    if not coords:
        raise ValueError("Circle 缺少中心点")
    center = _pnt(coords[0])
    normal = _vec(coords[2]) if len(coords) >= 3 else gp_Vec(0.0, 0.0, 1.0)
    if normal.Magnitude() <= EPS:
        raise ValueError("Circle 法向量长度为 0")
    return gp_Ax2(center, gp_Dir(normal)), radius


def _build_circle(node: Dict[str, Any], path: str, checks: List[Dict[str, Any]]):
    ax, radius = _circle_axis(node)
    edge = BRepBuilderAPI_MakeEdge(gp_Circ(ax, radius)).Edge()
    ok = _brep_valid(edge)
    _record(checks, path, "Circle", ok, f"r={radius}")
    if not ok:
        raise ValueError("Circle B-Rep 无效")
    return edge


def _circle_wire(node: Dict[str, Any], path: str, checks: List[Dict[str, Any]]):
    maker = BRepBuilderAPI_MakeWire()
    maker.Add(_build_circle(node, path, checks))
    if not maker.IsDone():
        raise ValueError("Circle wire 构造失败")
    wire = maker.Wire()
    if not bool(wire.Closed()):
        raise ValueError("Circle wire 未闭合")
    return wire


def _build_wire(node: Dict[str, Any], path: str, checks: List[Dict[str, Any]]):
    maker = BRepBuilderAPI_MakeWire()
    children = node.get("features") or []
    if not children:
        raise ValueError("Wire 下没有边")
    for i, child in enumerate(children):
        if not isinstance(child, dict):
            raise ValueError("Wire child 不是 object")
        cp = f"{path}.features[{i}]"
        t = child.get("type")
        if t == "Edge": maker.Add(_build_edge(child, cp, checks))
        elif t == "Arc": maker.Add(_build_arc(child, cp, checks))
        elif t == "Circle": maker.Add(_build_circle(child, cp, checks))
        else: raise ValueError(f"Wire 不支持 child type={t!r}")
    if not maker.IsDone():
        raise ValueError("Wire 构造失败：边可能不连通或顺序错误")
    wire = maker.Wire()
    closed = bool(wire.Closed())
    ok = closed and _brep_valid(wire)
    _record(checks, path, "Wire", ok, f"closed={closed}")
    if not closed:
        raise ValueError("Wire 未闭合")
    if not _brep_valid(wire):
        raise ValueError("Wire B-Rep 无效")
    return wire


def _build_face(node: Dict[str, Any], path: str, checks: List[Dict[str, Any]]):
    wires = []
    for i, child in enumerate(node.get("features") or []):
        if not isinstance(child, dict):
            raise ValueError("Face child 不是 object")
        cp = f"{path}.features[{i}]"
        t = child.get("type")
        if t == "Wire":
            wires.append(_build_wire(child, cp, checks))
        elif t == "Circle":
            wires.append(_circle_wire(child, cp, checks))
        elif t == "Group":
            for j, gc in enumerate(child.get("features") or []):
                if isinstance(gc, dict) and gc.get("type") == "Circle":
                    wires.append(_circle_wire(gc, f"{cp}.features[{j}]", checks))
                else:
                    raise ValueError("Face 内 Group 目前只允许 Circle children")
        else:
            raise ValueError(f"Face 不支持 child type={t!r}")
    if not wires:
        raise ValueError("Face 下没有可用轮廓")
    fb = BRepBuilderAPI_MakeFace(wires[0])

    # Keep the original orientation for outer contours and reverse inner holes
    for w in wires[1:]:
        w.Reverse()
        fb.Add(w)
    if not fb.IsDone():
        raise ValueError("Face 构造失败：轮廓可能不闭合、相交或非共面")
    face = fb.Face()
    ok = _brep_valid(face)
    _record(checks, path, "Face", ok, f"wires={len(wires)}")
    if not ok:
        raise ValueError("Face B-Rep 无效")
    return face


def _face_normal(face_node: Dict[str, Any]):
    coords = face_node.get("coordinates") or []
    if len(coords) >= 2:
        v = _vec(coords[1])
        if v.Magnitude() > EPS:
            return v
    return gp_Vec(0.0, 0.0, 1.0)


def _build_prism(node: Dict[str, Any], path: str, checks: List[Dict[str, Any]]):
    features = node.get("features") or []
    if len(features) != 1 or not isinstance(features[0], dict) or features[0].get("type") != "Face":
        raise ValueError("Prism 必须恰好包含一个 Face")
    face_node = features[0]
    face = _build_face(face_node, f"{path}.features[0]", checks)
    height = float((node.get("parameters") or {}).get("height", 0.0) or 0.0)
    if not math.isfinite(height) or abs(height) <= EPS:
        raise ValueError("Prism height 必须是非零有限数值")
    normal = _face_normal(face_node)
    vec = gp_Vec(normal.X() * height, normal.Y() * height, normal.Z() * height)
    prism = BRepPrimAPI_MakePrism(face, vec).Shape()
    vol = _volume(prism)
    ok = _brep_valid(prism) and abs(vol) > EPS
    _record(checks, path, "Prism", ok, f"volume={vol:.9g}")
    if abs(vol) <= EPS:
        raise ValueError("Prism 体积为 0")
    if not _brep_valid(prism):
        raise ValueError("Prism B-Rep 无效")
    return prism


def _boolean_effective(kind: str, before_a: float, before_b: float, after: float) -> bool:
    scale = max(abs(before_a), abs(before_b), abs(after), 1.0)
    tol = 1e-8 * scale
    if kind == "Fuse":
        return abs((before_a + before_b) - after) > tol
    return abs(before_a - after) > tol


def _fuse_pair(a, b, path: str, checks: List[Dict[str, Any]], errors: List[Dict[str, str]], warnings: List[Dict[str, str]], no_effect_is_error: bool):
    op = BRepAlgoAPI_Fuse(a, b); op.Build()
    if not op.IsDone(): raise ValueError("Fuse 布尔融合失败")
    result = op.Shape()
    va, vb, vr = _volume(a), _volume(b), _volume(result)
    effective = _boolean_effective("Fuse", va, vb, vr)
    ok = _brep_valid(result) and effective
    _record(checks, path, "Fuse", ok, f"effective={effective}, volume={vr:.9g}")
    if not _brep_valid(result): raise ValueError("Fuse 结果 B-Rep 无效")
    if not effective:
        issue = _issue("fuse_no_effect", path, "Fuse 未产生几何交叠/融合效果", node_type="Fuse", severity="error" if no_effect_is_error else "warning")
        (errors if no_effect_is_error else warnings).append(issue)
    return result


def _cut_pair(a, b, path: str, checks: List[Dict[str, Any]], errors: List[Dict[str, str]], warnings: List[Dict[str, str]], no_effect_is_error: bool):
    op = BRepAlgoAPI_Cut(a, b); op.Build()
    if not op.IsDone(): raise ValueError("Cut 布尔剪切失败")
    result = op.Shape()
    va, vr = _volume(a), _volume(result)
    effective = _boolean_effective("Cut", va, _volume(b), vr)
    ok = _brep_valid(result) and effective
    _record(checks, path, "Cut", ok, f"effective={effective}, volume={vr:.9g}")
    if not _brep_valid(result): raise ValueError("Cut 结果 B-Rep 无效")
    if not effective:
        issue = _issue("cut_no_effect", path, "Cut 未改变基体体积，刀具可能未与基体相交", node_type="Cut", severity="error" if no_effect_is_error else "warning")
        (errors if no_effect_is_error else warnings).append(issue)
    return result


def build_shape(node: Any, *, path: str = "root", checks=None, errors=None, warnings=None, no_effect_is_error: bool = True):
    checks = checks if checks is not None else []
    errors = errors if errors is not None else []
    warnings = warnings if warnings is not None else []
    if not isinstance(node, dict): raise ValueError("节点不是 object")
    t = node.get("type"); features = node.get("features") or []

    if t == "Group":
        if not features: raise ValueError("Group 下没有 features")
        shapes = [build_shape(c, path=f"{path}.features[{i}]", checks=checks, errors=errors, warnings=warnings, no_effect_is_error=no_effect_is_error) for i, c in enumerate(features)]
        if len(shapes) == 1:
            result = shapes[0]
        else:
            builder = BRep_Builder(); result = TopoDS_Compound(); builder.MakeCompound(result)
            for s in shapes: builder.Add(result, s)
        _record(checks, path, "Group", _brep_valid(result), f"children={len(shapes)}")
        return result

    if t in {"Fuse", "Cut"}:
        if len(features) < 2: raise ValueError(f"{t} 至少需要两个实体")
        result = build_shape(features[0], path=f"{path}.features[0]", checks=checks, errors=errors, warnings=warnings, no_effect_is_error=no_effect_is_error)
        for i, c in enumerate(features[1:], start=1):
            s = build_shape(c, path=f"{path}.features[{i}]", checks=checks, errors=errors, warnings=warnings, no_effect_is_error=no_effect_is_error)
            if t == "Fuse": result = _fuse_pair(result, s, path, checks, errors, warnings, no_effect_is_error)
            else: result = _cut_pair(result, s, path, checks, errors, warnings, no_effect_is_error)
        return result

    if t == "Prism": return _build_prism(node, path, checks)
    if t == "Face": return _build_face(node, path, checks)
    if t == "Wire": return _build_wire(node, path, checks)
    if t == "Edge": return _build_edge(node, path, checks)
    if t == "Arc": return _build_arc(node, path, checks)
    if t == "Circle": return _build_circle(node, path, checks)
    raise ValueError(f"暂不支持的节点类型: {t!r}")


def check_geometry_validity(text_or_obj: Any, *, no_effect_is_error: bool = True) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "available": OCC_AVAILABLE, "valid": None, "volume": 0.0,
        "error": "", "errors": [], "warnings": [], "checks": [],
    }
    if not OCC_AVAILABLE:
        result["error"] = f"pythonocc 未安装: {OCC_IMPORT_ERROR}"
        result["errors"] = [_issue("occ_unavailable", "root", result["error"])]
        return result
    try:
        obj = _as_obj(text_or_obj)
    except Exception as exc:
        result["valid"] = False
        result["error"] = f"json_parse_failed: {type(exc).__name__}: {exc}"
        result["errors"].append(_issue("json_parse_failed", "root", result["error"]))
        return result

    try:
        shape = build_shape(obj, path="root", checks=result["checks"], errors=result["errors"], warnings=result["warnings"], no_effect_is_error=no_effect_is_error)
        vol = _volume(shape); result["volume"] = vol
        if abs(vol) <= EPS:
            result["errors"].append(_issue("empty_solid", "root", "最终实体体积为 0"))
        if not _brep_valid(shape):
            result["errors"].append(_issue("brepcheck_invalid", "root", "最终 B-Rep 拓扑无效"))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["errors"].append(_issue("build_failed", "root", result["error"]))

    result["valid"] = len(result["errors"]) == 0
    if not result["valid"] and not result["error"]:
        result["error"] = "; ".join(e["code"] for e in result["errors"][:4])
    return result


if __name__ == "__main__":
    import argparse, pathlib
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--allow_no_effect_boolean", action="store_true")
    a = p.parse_args()
    text = pathlib.Path(a.input).read_text(encoding="utf-8")
    print(json.dumps(check_geometry_validity(text, no_effect_is_error=not a.allow_no_effect_boolean), ensure_ascii=False, indent=2))
