"""
Generate Cadmium-style English procedural descriptions from Cadmium-lite CAD JSON.

This version calls Qwen / Alibaba Cloud Model Studio (DashScope) through the
OpenAI-compatible Chat Completions API, using qwen3.7-max by default.

Target output style:
- English Cadmium dataset style.
- One coherent procedural paragraph.
- Describe base solid, sketches on faces, closed loops, lines, arcs, circles, extrusions, cuts, fusions,
  holes, cutouts, and final result when those features exist.
- Do not force a short summary.
- Do not truncate generated descriptions.
- Do not round, simplify, shorten, or rewrite numeric values. If a number is included, copy it exactly
  from the raw JSON whenever possible.

PowerShell example:
$env:DASHSCOPE_API_KEY="<your-dashscope-api-key>"
$env:QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:QWEN_MODEL="qwen3.7-max"
$env:OUTPUT_NAME_MODE="description"
$env:OUTPUT_DIR="outputs/descriptions/simple"
$env:DATASET_ROOT="data/simple"
$env:MAX_TOKENS="2048"
$env:USE_IMAGES="0"

python generate_description_qwen.py
"""


from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

ROOT_DIR = Path(os.getenv("DATASET_ROOT", ""))
OUTPUT_DIR_TEXT = os.getenv("OUTPUT_DIR", "").strip()
OUTPUT_DIR = Path(OUTPUT_DIR_TEXT) if OUTPUT_DIR_TEXT else None
MAX_JSON_CHARS = int(os.getenv("MAX_JSON_CHARS", "50000"))
SLEEP_BETWEEN = float(os.getenv("SLEEP_BETWEEN", "0.5"))
DESCRIPTION_STYLE = os.getenv("DESCRIPTION_STYLE", "face_steps").strip().lower()

# =====================
# Qwen / DashScope OpenAI-compatible model configuration
# =====================

QWEN_BASE_URL = (
    os.getenv("QWEN_BASE_URL")
    or os.getenv("DASHSCOPE_BASE_URL")
    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

QWEN_MODEL = (
    os.getenv("QWEN_DESC_MODEL")
    or os.getenv("QWEN_MODEL")
    or os.getenv("DASHSCOPE_MODEL")
    or "qwen3.7-max"
)

USE_IMAGES = os.getenv("USE_IMAGES", "0").strip().lower() in {"1", "true", "yes"}

LLM_CONFIG = {
    "base_url": QWEN_BASE_URL,
    "api_key": None,  # read at runtime from QWEN_API_KEY / DASHSCOPE_API_KEY
    "model": QWEN_MODEL,
    "generation_timeout": int(os.getenv("GENERATION_TIMEOUT", "180")),
    "max_retries": int(os.getenv("MAX_RETRIES", "3")),
    "retry_sleep_seconds": float(os.getenv("RETRY_SLEEP_SECONDS", "1")),
    "max_tokens": int(os.getenv("MAX_TOKENS", "2048")),
    "temperature": float(os.getenv("TEMPERATURE", "0.2")),
    "top_p": float(os.getenv("TOP_P", "0.9")),
}

KEY_DIM_FIELDS = [
    "height", "radius", "r", "h", "length", "width", "thickness", "diameter",
    "alpha1", "alpha2",
]

FIELD_ALIASES = {"r": "radius", "h": "height"}

CN_NAME_MAP = {
    "height": "高度",
    "radius": "半径",
    "r": "半径",
    "h": "高度",
    "diameter": "直径",
    "length": "长度",
    "width": "宽度",
    "thickness": "厚度",
    "alpha1": "起始角",
    "alpha2": "终止角",
}


def get_qwen_api_key() -> str:
    """Get the Qwen / DashScope API key."""
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "No Qwen / DashScope API key detected. Set one of the following environment variables first:\n"
            "Windows PowerShell:\n"
            "  $env:DASHSCOPE_API_KEY=\"<your-dashscope-api-key>\"\n"
            "or:\n"
            "  $env:QWEN_API_KEY=\"<your-qwen-api-key>\"\n\n"
            "macOS / Linux:\n"
            "  export DASHSCOPE_API_KEY=\"<your-dashscope-api-key>\"\n"
        )
    return api_key


def get_client() -> OpenAI:
    if OpenAI is None:
        raise ImportError("The openai package is missing; run: pip install -U openai")

    return OpenAI(
        base_url=str(LLM_CONFIG["base_url"]),
        api_key=get_qwen_api_key(),
        timeout=int(LLM_CONFIG["generation_timeout"]),
    )


# =====================
# Basic utility functions
# =====================

def pretty_json(obj: Any, max_chars: int = MAX_JSON_CHARS) -> str:
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(s) > max_chars:
        s = s[:max_chars] + "\n...（已截断，仅保留前部 JSON；结构摘要已单独给出）"
    return s


def image_to_data_url(img_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(img_path))
    if mime_type is None:
        mime_type = "image/png"
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def extract_chat_text(response: Any) -> Tuple[Optional[str], Optional[str]]:
    if response is None:
        return None, "response is None"
    try:
        if not getattr(response, "choices", None):
            return None, "choices is empty"
        message = response.choices[0].message
        content = getattr(message, "content", None)
        if content is None:
            return None, "message.content is None"
        if isinstance(content, str):
            text = content.strip()
            return (text, None) if text else (None, "empty string content")
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        texts.append(item["text"])
                    elif item.get("text"):
                        texts.append(str(item["text"]))
                else:
                    text_attr = getattr(item, "text", None)
                    if text_attr:
                        texts.append(str(text_attr))
            text = "\n".join(t.strip() for t in texts if str(t).strip()).strip()
            return (text, None) if text else (None, "content list has no usable text")
        text = str(content).strip()
        return (text, None) if text else (None, "unrecognized empty content")
    except Exception as e:
        return None, f"extract_chat_text error: {e}"


def walk_nodes(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for child in node.get("features", []):
            yield from walk_nodes(child)
    elif isinstance(node, list):
        for item in node:
            yield from walk_nodes(item)


def nodes_by_type(json_data: Dict[str, Any], type_name: str) -> List[Dict[str, Any]]:
    return [node for node in walk_nodes(json_data) if node.get("type") == type_name]


def normalize_dim_key(key: str) -> str:
    return FIELD_ALIASES.get(key, key)


def extract_raw_number_by_key(raw_json_text: str, field_name: str) -> Optional[str]:
    pattern = rf'"{re.escape(field_name)}"\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)'
    m = re.search(pattern, raw_json_text)
    return m.group(1) if m else None


def extract_key_dims_from_raw_json(raw_json_text: str) -> Dict[str, str]:
    dims: Dict[str, str] = {}
    for raw_key in KEY_DIM_FIELDS:
        value = extract_raw_number_by_key(raw_json_text, raw_key)
        if value is None:
            continue
        norm_key = normalize_dim_key(raw_key)
        if norm_key not in dims:
            dims[norm_key] = value
    return dims


def short_num(x: Any, digits: int = 17) -> str:
    """
    Format a numeric value without adding approximate wording.

    The default 17 significant digits preserve round-trippable Python float
    values. Values with tiny numerical noise near zero are still rendered as 0
    to avoid meaningless 1e-33 artifacts in natural-language descriptions.
    """
    try:
        v = float(x)
    except Exception:
        return str(x)

    if abs(v) < 1e-12:
        return "0"

    return format(v, f".{digits}g")

def fmt_dim_value(value: Optional[Any]) -> str:
    if value is None:
        return ""
    return short_num(value, digits=17)

def point_xyztxt(point: Sequence[Any], digits: int = 17) -> str:
    if not isinstance(point, Sequence) or len(point) < 3:
        return str(point)
    return f"({short_num(point[0], digits)},{short_num(point[1], digits)},{short_num(point[2], digits)})"

def _vec3(values: Any) -> Optional[Tuple[float, float, float]]:
    """Return a numeric 3D vector/point, or None if the value is unusable."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) < 3:
        return None
    try:
        return float(values[0]), float(values[1]), float(values[2])
    except Exception:
        return None


def _vec_len(vec: Optional[Sequence[float]]) -> float:
    if vec is None:
        return 0.0
    try:
        return math.sqrt(sum(float(v) * float(v) for v in vec[:3]))
    except Exception:
        return 0.0


def _normalize_vec(vec: Any, eps: float = 1e-9) -> Optional[Tuple[float, float, float]]:
    v = _vec3(vec)
    length = _vec_len(v)
    if not v or length <= eps:
        return None
    return tuple(float(x) / length for x in v)  # type: ignore[return-value]


def _scale_vec(vec: Optional[Sequence[float]], factor: float) -> Optional[Tuple[float, float, float]]:
    if vec is None:
        return None
    return tuple(float(v) * factor for v in vec[:3])  # type: ignore[return-value]


def _same_direction(a: Any, b: Any, tol: float = 0.98) -> bool:
    va = _normalize_vec(a)
    vb = _normalize_vec(b)
    if va is None or vb is None:
        return False
    return sum(va[i] * vb[i] for i in range(3)) >= tol


def vec_txt(vec: Any, digits: int = 5) -> str:
    v = _normalize_vec(vec)
    if v is None:
        raw = _vec3(vec)
        return point_xyztxt(raw, digits) if raw else "未知方向"
    return point_xyztxt(v, digits)


AXIS_EPS = 1e-8


def dominant_axis_label(vec: Any, eps: float = AXIS_EPS) -> str:
    """
    Convert a direction vector to an exact direction phrase.

    If the normalized vector is axis-aligned within numerical tolerance, return
    +X/-X/+Y/-Y/+Z/-Z. Otherwise return the normalized vector itself. This
    function intentionally avoids approximate words such as “约” or “近似”.
    """
    v = _normalize_vec(vec)
    if v is None:
        return "未知方向"

    names = ["X", "Y", "Z"]
    abs_values = [abs(v[0]), abs(v[1]), abs(v[2])]
    axis = max(range(3), key=lambda i: abs_values[i])

    if abs_values[axis] >= 1.0 - eps and all(abs(v[i]) <= eps for i in range(3) if i != axis):
        return ("+" if v[axis] >= 0 else "-") + names[axis]

    return vec_txt(v, digits=17)

def plane_label_from_normal(normal: Any) -> str:
    """Infer the sketch plane name from its normal."""
    label = dominant_axis_label(normal)
    if label.endswith("X"):
        return "YZ平面"
    if label.endswith("Y"):
        return "XZ平面"
    if label.endswith("Z"):
        return "XY平面"
    if label != "未知方向":
        return f"垂直于法向{label}的平面"
    return "未知平面"

def local_frame_phrase(frame: Optional[Dict[str, Any]]) -> str:
    """Natural-language phrase for a face/local sketch frame."""
    if not frame:
        return ""
    normal = frame.get("normal")
    x_dir = frame.get("x_dir")
    parts: List[str] = []
    if normal:
        parts.append(f"所在平面为{plane_label_from_normal(normal)}，法向为{dominant_axis_label(normal)}")
    if x_dir:
        parts.append(f"局部X方向为{dominant_axis_label(x_dir)}")
    return "，".join(parts)

def get_face_frame(face_node: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract the local sketch frame from a Face.

    In this Cadmium-lite style data, Face.coordinates commonly stores:
    [origin/reference point, normal vector, local x direction].  The exact names
    are not surfaced to the final description, but preserving these vectors
    prevents circular profiles from being assumed to lie in the global XY plane.
    """
    coords = face_node.get("coordinates", []) if isinstance(face_node, dict) else []
    origin = coords[0] if isinstance(coords, list) and len(coords) > 0 else None
    normal = _normalize_vec(coords[1]) if isinstance(coords, list) and len(coords) > 1 else None
    x_dir = _normalize_vec(coords[2]) if isinstance(coords, list) and len(coords) > 2 else None
    return {
        "origin": list(origin) if isinstance(origin, list) else origin,
        "normal": list(normal) if normal else None,
        "x_dir": list(x_dir) if x_dir else None,
        "plane": plane_label_from_normal(normal) if normal else "",
    }


def get_prism_declared_direction(prism_node: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    coords = prism_node.get("coordinates", [])
    if isinstance(coords, list) and len(coords) > 1:
        return _normalize_vec(coords[1])
    return None


def get_prism_extrude_info(
    prism_node: Dict[str, Any],
    face_node: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Decide the effective extrusion direction.

    When auto_normal_vector is true, the sketch face normal is the safer
    direction cue. Otherwise, fall back to the direction vector in Prism
    coordinates, then to the face normal if no explicit vector is available.
    """
    params = prism_node.get("parameters", {}) if isinstance(prism_node, dict) else {}
    frame = get_face_frame(face_node or first_face(prism_node))
    face_normal = _normalize_vec(frame.get("normal"))
    declared = get_prism_declared_direction(prism_node)
    auto_normal = bool(params.get("auto_normal_vector"))

    if auto_normal and face_normal is not None:
        direction = face_normal
        source = "面法向"
    elif declared is not None:
        direction = declared
        source = "给定方向"
    elif face_normal is not None:
        direction = face_normal
        source = "面法向"
    else:
        direction = None
        source = "未知方向"

    height = get_height(prism_node)
    signed_direction = _scale_vec(direction, -1.0) if isinstance(height, (int, float)) and height < 0 else direction

    return {
        "direction": list(direction) if direction else None,
        "signed_direction": list(signed_direction) if signed_direction else None,
        "source": source,
        "auto_normal_vector": auto_normal,
        "declared_direction": list(declared) if declared else None,
        "face_normal": list(face_normal) if face_normal else None,
    }


def extrude_phrase(extrude_info: Optional[Dict[str, Any]]) -> str:
    if not extrude_info:
        return ""
    direction = extrude_info.get("signed_direction") or extrude_info.get("direction")
    if not direction:
        return ""
    source = extrude_info.get("source") or "方向"
    return f"沿{source}（{dominant_axis_label(direction)}）"

def cleanup_text(text: str) -> str:
    """
    Clean the model output without changing its geometric content.

    Important:
    - Do not truncate by length.
    - Do not round, shorten, simplify, or otherwise rewrite numbers.
    - Only remove wrappers, code fences, common prefixes, and extra whitespace.
    """
    if not text:
        return ""

    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[：:，,。\.\s]+", "", text)
    text = re.sub(r"[,，]{2,}", ",", text)
    text = re.sub(r"[。]{2,}", ".", text)

    bad_prefixes = [
        "自然语言描述：",
        "描述：",
        "最终描述：",
        "模型描述：",
        "输出：",
        "答案：",
        "Description:",
        "Final description:",
        "Final Description:",
        "Output:",
        "Answer:",
    ]
    for p in bad_prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()
            break

    # Remove quote wrappers if the model wrapped the whole paragraph in quotes.
    quote_pairs = [("\"", "\""), ("'", "'"), ("“", "”")]
    for left, right in quote_pairs:
        if len(text) >= 2 and text.startswith(left) and text.endswith(right):
            text = text[len(left):-len(right)].strip()
            break

    return text.strip()


# =====================
# Geometric structure summary and model classification
# =====================

def get_type_counter(json_data: Dict[str, Any]) -> Counter:
    return Counter(node.get("type") for node in walk_nodes(json_data) if isinstance(node, dict))


def get_height(prism_node: Dict[str, Any]) -> Optional[float]:
    try:
        return float(prism_node.get("parameters", {}).get("height"))
    except Exception:
        return None


def get_circle_info(circle_node: Dict[str, Any]) -> Dict[str, Any]:
    coords = circle_node.get("coordinates", [])
    center = coords[0] if isinstance(coords, list) and len(coords) > 0 else None
    # Circle.coordinates commonly carries the circle axis/normal around index 2.
    # Keep both names for compatibility with existing downstream code.
    circle_axis = _normalize_vec(coords[2]) if isinstance(coords, list) and len(coords) > 2 else None
    local_x = _normalize_vec(coords[1]) if isinstance(coords, list) and len(coords) > 1 else None
    radius = circle_node.get("parameters", {}).get("r")
    return {
        "center": center,
        "normal": list(circle_axis) if circle_axis else None,
        "axis": list(circle_axis) if circle_axis else None,
        "local_x": list(local_x) if local_x else None,
        "plane": plane_label_from_normal(circle_axis) if circle_axis else "",
        "radius": radius,
    }


def first_face(prism_node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for child in prism_node.get("features", []):
        if isinstance(child, dict) and child.get("type") == "Face":
            return child
    return None


def face_children_by_type(face_node: Dict[str, Any], type_name: str) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for child in face_node.get("features", []):
        if not isinstance(child, dict):
            continue
        if child.get("type") == type_name:
            result.append(child)
        # A Circle is usually wrapped by a Group.
        if type_name == "Circle" and child.get("type") == "Group":
            for g in child.get("features", []):
                if isinstance(g, dict) and g.get("type") == "Circle":
                    result.append(g)
    return result


def wire_primitives(wire_node: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        child for child in wire_node.get("features", [])
        if isinstance(child, dict) and child.get("type") in {"Edge", "Arc"}
    ]


def wire_points(wire_node: Dict[str, Any], max_points: int = 8) -> List[List[float]]:
    points: List[List[float]] = []
    for prim in wire_primitives(wire_node):
        coords = prim.get("coordinates", [])
        if not isinstance(coords, list):
            continue
        if prim.get("type") == "Edge":
            candidates = coords[:2]
        else:
            candidates = coords[:3]
        for p in candidates:
            if isinstance(p, list) and len(p) == 3 and all(isinstance(v, (int, float)) for v in p):
                if p not in points:
                    points.append(p)
                if len(points) >= max_points:
                    return points
    return points


def edge_lengths(wire_node: Dict[str, Any]) -> List[float]:
    lengths: List[float] = []
    for prim in wire_primitives(wire_node):
        if prim.get("type") != "Edge":
            continue
        coords = prim.get("coordinates", [])
        if not isinstance(coords, list) or len(coords) < 2:
            continue
        p1, p2 = coords[0], coords[1]
        if (
            isinstance(p1, list) and isinstance(p2, list)
            and len(p1) == 3 and len(p2) == 3
            and all(isinstance(v, (int, float)) for v in p1 + p2)
        ):
            length = math.dist([float(v) for v in p1], [float(v) for v in p2])
            lengths.append(length)
    return lengths


def profile_bbox_from_wire(wire_node: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Estimate the 2D profile size from a Wire.

    Dataset sketches are not always in the XY plane. Some rectangles lie in XZ
    or YZ, so using only x/y can produce a false 0 width. We therefore compute
    ranges on x/y/z and return the two largest non-zero ranges.
    """
    pts = wire_points(wire_node, max_points=1000)
    if not pts:
        return None

    ranges = []
    for axis in range(3):
        values = [float(p[axis]) for p in pts]
        ranges.append(max(values) - min(values))

    ranges = sorted([r for r in ranges if r > 1e-9], reverse=True)
    if not ranges:
        return None
    if len(ranges) == 1:
        return ranges[0], 0.0
    return ranges[0], ranges[1]


def prism_profile_summary(prism_node: Dict[str, Any]) -> Dict[str, Any]:
    face = first_face(prism_node)
    height = get_height(prism_node)
    face_frame = get_face_frame(face)
    extrude_info = get_prism_extrude_info(prism_node, face)

    summary: Dict[str, Any] = {
        "height": height,
        "profile": "unknown",
        "circles": [],
        "wires": [],
        "face_frame": face_frame,
        "extrude": extrude_info,
    }

    if not face:
        return summary

    circles = face_children_by_type(face, "Circle")
    wires = face_children_by_type(face, "Wire")

    for circle_node in circles:
        info = get_circle_info(circle_node)
        # If the circle itself lacks a stable axis, inherit the parent face normal.
        if not info.get("axis") and face_frame.get("normal"):
            info["axis"] = face_frame.get("normal")
            info["normal"] = face_frame.get("normal")
            info["plane"] = face_frame.get("plane")
        summary["circles"].append(info)

    for wire_node in wires:
        primitives = wire_primitives(wire_node)
        edge_count = sum(1 for p in primitives if p.get("type") == "Edge")
        arc_count = sum(1 for p in primitives if p.get("type") == "Arc")
        bbox = profile_bbox_from_wire(wire_node)
        summary["wires"].append({
            "edge_count": edge_count,
            "arc_count": arc_count,
            "bbox": bbox,
            "points": wire_points(wire_node, max_points=8),
            "face_frame": face_frame,
        })

    if len(circles) == 1 and not wires:
        summary["profile"] = "circle"
    elif len(circles) >= 2 and not wires:
        summary["profile"] = "multi_circle"
    elif wires:
        w = summary["wires"][0]
        if w["edge_count"] == 4 and w["arc_count"] == 0:
            bbox = w.get("bbox")
            if bbox:
                a, b = bbox
                summary["profile"] = "square" if abs(a - b) <= max(a, b, 1.0) * 0.03 else "rectangle"
            else:
                summary["profile"] = "quadrilateral"
        elif w["edge_count"] == 3 and w["arc_count"] == 0:
            summary["profile"] = "triangle"
        elif w["arc_count"] > 0:
            summary["profile"] = "arc_wire"
        else:
            summary["profile"] = "polygon"
    elif circles:
        summary["profile"] = "circle_group"

    return summary


def model_complexity(json_data: Dict[str, Any]) -> str:
    c = get_type_counter(json_data)
    node_total = sum(c.values())
    primitive_total = c.get("Edge", 0) + c.get("Arc", 0) + c.get("Circle", 0)
    boolean_total = c.get("Fuse", 0) + c.get("Cut", 0)

    if node_total <= 15 and c.get("Prism", 0) <= 2 and primitive_total <= 8:
        return "simple"
    if node_total <= 80 and c.get("Prism", 0) <= 8:
        return "medium"
    if boolean_total >= 1 or c.get("Prism", 0) >= 8 or primitive_total >= 40:
        return "complex"
    return "medium"


def detect_standard_shape(json_data: Dict[str, Any]) -> Optional[str]:
    c = get_type_counter(json_data)
    prism_nodes = nodes_by_type(json_data, "Prism")
    prism_summaries = [prism_profile_summary(p) for p in prism_nodes]

    # Complex gear/teeth shapes: many alternating edges/arcs are usually hard to name point by point.
    if c.get("Arc", 0) >= 20 and c.get("Edge", 0) >= 20:
        return "gear-like or toothed-profile complex part"

    # Single circular profile extrusion: cylinder.
    if c.get("Prism", 0) == 1 and c.get("Circle", 0) == 1 and c.get("Wire", 0) == 0:
        return "solid cylinder"

    # Two concentric circles or boolean holes: ring/washer tendency; complex gear shapes with many arcs are not simplified here.
    if c.get("Circle", 0) >= 2 and c.get("Cut", 0) >= 1 and c.get("Prism", 0) >= 2 and c.get("Arc", 0) < 10:
        return "cylinder with a circular hole or washer-like solid"

    if c.get("Circle", 0) >= 2 and c.get("Prism", 0) == 1 and c.get("Wire", 0) == 0:
        return "annular extruded solid"

    # Single quadrilateral profile: rectangular/square prism.
    if c.get("Prism", 0) == 1 and c.get("Wire", 0) == 1 and c.get("Edge", 0) == 4 and c.get("Arc", 0) == 0:
        s = prism_summaries[0] if prism_summaries else {}
        if s.get("profile") == "square":
            return "square prism"
        return "rectangular prism"

    if c.get("Prism", 0) == 1 and c.get("Wire", 0) == 1 and c.get("Edge", 0) == 3 and c.get("Arc", 0) == 0:
        return "triangular prism"

    if c.get("Prism", 0) == 1 and c.get("Arc", 0) > 0 and c.get("Edge", 0) <= 10:
        return "prism with an arc-based boundary"

    if c.get("Circle", 0) >= 3 and c.get("Wire", 0) >= 1 and c.get("Prism", 0) == 1:
        return "plate-like prism with multiple circular holes or circular features"

    if c.get("Prism", 0) >= 10 and c.get("Fuse", 0) >= 1:
        return "complex assembly formed by fusing multiple extruded solids"

    return None


def summarize_prisms(json_data: Dict[str, Any], max_items: int = 6) -> List[str]:
    items: List[str] = []
    for idx, prism_node in enumerate(nodes_by_type(json_data, "Prism")[:max_items], start=1):
        s = prism_profile_summary(prism_node)
        h = fmt_dim_value(s.get("height"))
        profile = s.get("profile", "unknown")
        frame_text = local_frame_phrase(s.get("face_frame"))
        extrude_text = extrude_phrase(s.get("extrude"))

        if profile == "circle" and s["circles"]:
            circle = s["circles"][0]
            r = fmt_dim_value(circle.get("radius"))
            center = circle.get("center")
            text = f"拉伸体{idx}: 圆形截面"
            if r:
                text += f" 半径{r}"
            if center:
                text += f" 圆心{point_xyztxt(center)}"
            if circle.get("axis"):
                text += f" 圆面法向为{dominant_axis_label(circle.get('axis'))}"
            if frame_text:
                text += f"，{frame_text}"
            if h:
                text += f" 高度{h}"
            if extrude_text:
                text += f"，{extrude_text}拉伸"
            items.append(text)
            continue

        if profile in {"rectangle", "square", "quadrilateral", "triangle", "polygon", "arc_wire"}:
            wire = s["wires"][0] if s.get("wires") else {}
            bbox = wire.get("bbox")
            edge_count = wire.get("edge_count", 0)
            arc_count = wire.get("arc_count", 0)
            points = wire.get("points", [])
            text = f"拉伸体{idx}: "
            if profile == "rectangle":
                text += "矩形轮廓"
            elif profile == "square":
                text += "正方形轮廓"
            elif profile == "triangle":
                text += "三角形轮廓"
            elif profile == "arc_wire":
                text += f"含{edge_count}条直线和{arc_count}段圆弧的轮廓"
            else:
                text += f"含{edge_count}条直线"
            if bbox:
                text += f" 外包尺寸{short_num(bbox[0])}×{short_num(bbox[1])}"
            if frame_text:
                text += f"，{frame_text}"
            if h:
                text += f" 高度{h}"
            if extrude_text:
                text += f"，{extrude_text}拉伸"
            if points:
                text += "，关键点" + "、".join(point_xyztxt(p) for p in points[:4])
            items.append(text)
            continue

        text = f"拉伸体{idx}: 轮廓类型{profile}"
        if frame_text:
            text += f"，{frame_text}"
        if h:
            text += f" 高度{h}"
        if extrude_text:
            text += f"，{extrude_text}拉伸"
        items.append(text)

    return items


def iter_prism_contexts(
    node: Any,
    parent_type: str = "",
    child_index: int = -1,
    path: str = "$",
) -> Iterable[Tuple[Dict[str, Any], str, int, str]]:
    """
    Yield Prism nodes together with their immediate modeling context.

    Context is used to describe the operation after a face is formed:
    - direct child of Cut at index 0: base solid;
    - direct child of Cut at index >= 1: cutter solid;
    - direct child of Fuse: additive/fused solid;
    - otherwise: standalone/additive solid.
    """
    if isinstance(node, dict):
        node_type = node.get("type")
        if node_type == "Prism":
            yield node, parent_type, child_index, path

        features = node.get("features", [])
        if isinstance(features, list):
            for idx, child in enumerate(features):
                yield from iter_prism_contexts(
                    child,
                    parent_type=str(node_type or ""),
                    child_index=idx,
                    path=f"{path}.features[{idx}]",
                )
    elif isinstance(node, list):
        for idx, child in enumerate(node):
            yield from iter_prism_contexts(child, parent_type=parent_type, child_index=idx, path=f"{path}[{idx}]")


def _circle_phrase(circle_info: Dict[str, Any]) -> str:
    center = circle_info.get("center")
    radius = fmt_dim_value(circle_info.get("radius"))
    parts: List[str] = []
    if center:
        parts.append(f"圆心{point_xyztxt(center)}")
    if radius:
        parts.append(f"半径{radius}")
    return "、".join(parts) if parts else "圆形特征"

def _append_frame_to_desc(desc: str, frame: Optional[Dict[str, Any]]) -> str:
    phrase = local_frame_phrase(frame)
    if phrase:
        return f"{desc}，{phrase}"
    return desc


def describe_face_for_prism(prism_node: Dict[str, Any]) -> str:
    """
    Describe the 2D face/profile before extrusion.

    Preference:
    1. If the profile is standard/simple, describe dimensions first;
    2. If hard to name, describe ordered key contour points and arc/edge composition;
    3. Preserve face local frame/normal so downstream text-to-CAD does not assume XY/+Z.
    """
    summary = prism_profile_summary(prism_node)
    profile = summary.get("profile", "unknown")
    circles = summary.get("circles") or []
    wires = summary.get("wires") or []
    face_frame = summary.get("face_frame") or {}

    circle_phrases = [_circle_phrase(c) for c in circles[:4]]

    if profile == "circle" and circles:
        return _append_frame_to_desc(f"先画一个圆形面，{circle_phrases[0]}", face_frame)

    if profile == "multi_circle" and circle_phrases:
        return _append_frame_to_desc("先画多个圆形特征形成面，" + "；".join(circle_phrases), face_frame)

    if wires:
        w = wires[0]
        bbox = w.get("bbox")
        edge_count = int(w.get("edge_count") or 0)
        arc_count = int(w.get("arc_count") or 0)
        points = w.get("points") or []
        point_text = "、".join(point_xyztxt(p) for p in points[:6])

        if profile == "square" and bbox:
            side = short_num(max(bbox))
            desc = f"先画一个边长{side}的正方形面"
        elif profile == "rectangle" and bbox:
            desc = f"先画一个尺寸{short_num(bbox[0])}×{short_num(bbox[1])}的矩形面"
        elif profile == "triangle":
            if point_text:
                desc = f"先按关键点{point_text}连接成三角形面"
            else:
                desc = "先画一个三角形面"
        elif profile == "arc_wire":
            if point_text:
                desc = (
                    f"先按关键点{point_text}构成轮廓，"
                    f"轮廓含{edge_count}条直线和{arc_count}段圆弧，并闭合成一个面"
                )
            else:
                desc = f"先画一个含{edge_count}条直线和{arc_count}段圆弧的闭合轮廓面"
        elif profile in {"polygon", "quadrilateral"}:
            if point_text:
                desc = f"先按关键点{point_text}依次连线形成闭合多边形面"
            else:
                desc = f"先画一个由{edge_count}条直线构成的多边形面"
        else:
            if point_text:
                desc = f"先按关键点{point_text}形成一个闭合轮廓面"
            else:
                desc = "先画一个轮廓面"

        if circle_phrases:
            desc += "，面内包含" + "；".join(circle_phrases)
        return _append_frame_to_desc(desc, face_frame)

    if circle_phrases:
        return _append_frame_to_desc("先画圆形特征形成面，" + "；".join(circle_phrases), face_frame)

    return _append_frame_to_desc("先构造一个二维轮廓面", face_frame)


def describe_prism_operation(
    prism_node: Dict[str, Any],
    parent_type: str,
    child_index: int,
) -> str:
    """Describe what happens after the face is created, including direction."""
    h_value = get_height(prism_node)
    h = fmt_dim_value(h_value)
    height_text = f"高度{h}" if h else "给定高度"
    reverse = "，方向相反" if isinstance(h_value, (int, float)) and h_value < 0 else ""
    direction = extrude_phrase(get_prism_extrude_info(prism_node))
    direction_text = f"{direction}" if direction else "沿该面法向或给定方向"

    if parent_type == "Cut" and child_index == 0:
        return f"再{direction_text}拉伸{height_text}{reverse}，作为切除前的基础实体"
    if parent_type == "Cut" and child_index >= 1:
        return f"再{direction_text}拉伸{height_text}{reverse}形成切除体，并从主体中切除"
    if parent_type == "Fuse":
        return f"再{direction_text}拉伸{height_text}{reverse}形成实体，并与其他实体融合"
    return f"再{direction_text}拉伸{height_text}{reverse}形成三维实体"

def build_face_operation_steps(json_data: Dict[str, Any], max_items: int = 8) -> List[str]:
    """
    Build face-first modeling steps.

    Each step follows:
    Describe face/profile dimensions or coordinates first, then the extrusion/fusion/cut operations.
    """
    steps: List[str] = []
    for idx, (prism_node, parent_type, child_index, _path) in enumerate(
        iter_prism_contexts(json_data),
        start=1,
    ):
        if idx > max_items:
            remaining = len(nodes_by_type(json_data, "Prism")) - max_items
            if remaining > 0:
                steps.append(f"其余{remaining}个拉伸体按同样方式由各自轮廓面拉伸后参与融合或切除。")
            break

        face_desc = describe_face_for_prism(prism_node)
        op_desc = describe_prism_operation(prism_node, parent_type, child_index)
        steps.append(f"步骤{idx}：{face_desc}，{op_desc}")

    return steps


def face_steps_to_sentence(steps: Sequence[str], max_steps: int = 3) -> str:
    if not steps:
        return ""

    selected = list(steps[:max_steps])
    text = "；".join(s.rstrip("。；; ") for s in selected)
    if len(steps) > max_steps:
        text += f"；其余{len(steps) - max_steps}个局部特征按各自轮廓面继续拉伸、融合或切除"
    return text + "。"


def build_coordinate_brief(json_data: Dict[str, Any], max_wires: int = 2, max_circles: int = 4) -> str:
    pieces: List[str] = []

    # Take the most complex Wire to prioritize key profile points of complex models.
    wires = nodes_by_type(json_data, "Wire")
    wires_sorted = sorted(
        wires,
        key=lambda w: len(wire_primitives(w)),
        reverse=True,
    )
    for i, wire_node in enumerate(wires_sorted[:max_wires], start=1):
        primitives = wire_primitives(wire_node)
        pts = wire_points(wire_node, max_points=6)
        if not pts:
            continue
        edge_count = sum(1 for p in primitives if p.get("type") == "Edge")
        arc_count = sum(1 for p in primitives if p.get("type") == "Arc")
        pieces.append(
            f"轮廓{i}含{edge_count}条直线、{arc_count}段圆弧，关键点"
            + "、".join(point_xyztxt(p) for p in pts[:6])
        )

    circles = nodes_by_type(json_data, "Circle")[:max_circles]
    circle_texts = []
    for circle_node in circles:
        info = get_circle_info(circle_node)
        r = fmt_dim_value(info.get("radius"))
        center = info.get("center")
        axis = info.get("axis") or info.get("normal")
        if center and r:
            t = f"圆心{point_xyztxt(center)}半径{r}"
            if axis:
                t += f"，圆面法向为{dominant_axis_label(axis)}，位于{plane_label_from_normal(axis)}"
            circle_texts.append(t)
    if circle_texts:
        pieces.append("圆特征：" + "；".join(circle_texts))

    frame_texts = []
    for idx, prism_node in enumerate(nodes_by_type(json_data, "Prism")[:6], start=1):
        s = prism_profile_summary(prism_node)
        frame_text = local_frame_phrase(s.get("face_frame"))
        extrude_text = extrude_phrase(s.get("extrude"))
        if frame_text or extrude_text:
            frame_texts.append(f"拉伸体{idx}：{frame_text}{'，' if frame_text and extrude_text else ''}{extrude_text}")
    if frame_texts:
        pieces.append("局部方向：" + "；".join(frame_texts))

    heights = []
    for prism_node in nodes_by_type(json_data, "Prism")[:6]:
        h = get_height(prism_node)
        if h is not None:
            heights.append(short_num(h))
    if heights:
        pieces.append("主要拉伸高度：" + "、".join(heights))

    return "；".join(pieces)


def build_coordinate_frame_brief(json_data: Dict[str, Any], max_items: int = 8) -> List[str]:
    brief: List[str] = []
    for idx, prism_node in enumerate(nodes_by_type(json_data, "Prism")[:max_items], start=1):
        s = prism_profile_summary(prism_node)
        frame_text = local_frame_phrase(s.get("face_frame"))
        extrude_text = extrude_phrase(s.get("extrude"))
        pieces = []
        if frame_text:
            pieces.append(frame_text)
        if extrude_text:
            pieces.append(f"{extrude_text}拉伸")
        if pieces:
            brief.append(f"拉伸体{idx}: " + "，".join(pieces))
    return brief


def build_model_analysis(json_data: Dict[str, Any], dims: Dict[str, str]) -> Dict[str, Any]:
    """
    Build a compact English topology summary for the prompt.

    Numeric values in this summary should not be treated as the authoritative source.
    The raw JSON text is passed separately to the model so that long decimals and original
    numeric formatting can be copied exactly when needed.
    """
    counter = get_type_counter(json_data)
    standard_shape = detect_standard_shape(json_data)
    complexity = model_complexity(json_data)

    boolean_text: List[str] = []
    if counter.get("Fuse", 0):
        boolean_text.append(f"{counter.get('Fuse', 0)} fuse operation(s)")
    if counter.get("Cut", 0):
        boolean_text.append(f"{counter.get('Cut', 0)} cut operation(s)")

    analysis = {
        "complexity": complexity,
        "standard_shape_hint": standard_shape or "",
        "node_counts": {k: int(v) for k, v in sorted(counter.items()) if k},
        "boolean_summary": ", ".join(boolean_text) if boolean_text else "no obvious Boolean operation",
        "raw_key_dimensions_found": dict(dims),
        "modeling_guidance": (
            "Use the raw JSON as the source of truth for exact numeric values and sketch topology. "
            "Describe the model procedurally in Cadmium style: base solid first, then sketches, "
            "closed loops, line/arc/circle elements, extrusions, cuts, fusions, holes, and final result."
        ),
    }
    return analysis


def analysis_to_text(analysis: Dict[str, Any]) -> str:
    return json.dumps(analysis, ensure_ascii=False, indent=2)


# =====================
# Prompt: generate descriptions according to the dataset modeling logic
# =====================

CADMIUM_STYLE_EXAMPLES = """
Example 1:
Begin by creating a solid cylinder with a diameter of 0.4167 and a height of 0.3333. On the top face of the cylinder, sketch a circle centered at (0.2083, 0.2083) with a radius of 0.2083. Inside this circle, define a closed loop comprised of two horizontal lines and an arc: draw the first line from (0.1383, 0.1176) to (0.1994, 0.1176), the second line from (0.1994, 0.1176) to (0.2783, 0.1176), and connect the end points of these lines back to the starting point with an arc that has its midpoint at (0.2083, 0.3229), forming a segment that cuts across the base of the circle. Extrude this internal profile fully through the cylinder to create a cylindrical hole with a flat chord across its lower portion as defined by the lines and arc, resulting in a cylinder with a partial circular cutout running through its center.

Example 2:
Begin by creating a rectangular block with a length of 1.5, a width of 0.75, and a thickness of 0.25. On the top face, sketch two circular profiles positioned symmetrically along the length of the block. Extrude both circular profiles downward through the block to create two vertical holes, resulting in a rectangular plate with through-holes.

Example 3:
Begin by creating a circular disk with the specified radius and thickness. On its top face, sketch a smaller concentric circle and extrude it fully through the disk to form a central hole, resulting in a washer-like solid.

Example 4:
Begin by creating the main solid from a closed polygonal sketch. The sketch is defined by straight edges and circular arcs, forming an irregular plate outline. Extrude this closed profile to the given thickness, then add or remove circular features according to the hole profiles, resulting in a plate-like solid with arc-based boundaries and cutouts.
""".strip()


def build_description_prompt(
    json_data: Dict[str, Any],
    dims: Dict[str, str],
    analysis: Dict[str, Any],
    raw_json_text: Optional[str] = None,
) -> str:
    analysis_text = analysis_to_text(analysis)

    # Use raw JSON text as the numeric source of truth. json.dumps(json_data) may reformat floats,
    # while raw_json_text preserves the original decimal strings from the dataset file.
    json_text = raw_json_text if raw_json_text is not None else pretty_json(json_data)
    if len(json_text) > MAX_JSON_CHARS:
        json_text = json_text[:MAX_JSON_CHARS] + "\n... truncated; use the structure summary for topology and the visible raw JSON for exact numbers ..."

    return f"""
You are writing CAD modeling descriptions in the style of the Cadmium dataset.

Your task is to convert the given Cadmium-lite CAD JSON and compact structure summary into one detailed English procedural modeling description.

Use the examples only as style references. Choose the closest style according to the model structure, but do not explicitly mention the example number.

Cadmium-style requirements:
1. Output exactly one coherent English paragraph.
2. Do not output JSON, Markdown, titles, bullet points, code, or explanations.
3. Start with the base solid when one exists, using phrases such as "Begin by creating...".
4. Then describe sketches on faces, closed loops, circles, straight lines, arcs, polygonal profiles, holes, cutouts, fusions, and extrusions when those features exist.
5. Use procedural phrases such as "On the top face...", "sketch...", "draw...", "connect...", "define a closed loop...", "extrude...", "cut through...", and "resulting in...".
6. Do not make the description overly short. Do not collapse a detailed sketch into only a generic phrase when line, arc, circle, or cutout details are present.
7. Do not mention internal JSON field names or CAD schema terms such as Group, Prism, Face, Wire, Edge, Arc, Circle, coordinates, parameters, features, schema, node, or operation tree.
8. Preserve numeric values exactly as provided in the raw JSON whenever you include them. Do not round, simplify, shorten, expand, or rewrite decimal values.
9. If the compact structure summary and the raw JSON disagree on a number, prefer the raw JSON number.
10. If a number is not necessary for a natural Cadmium-style description, you may omit it; but every number you do include must be copied exactly.
11. Do not use approximate wording such as approximately, about, roughly, around, close to, near, estimated, or similar phrases.
12. Do not invent material, color, manufacturing process, or function that is not present in the JSON.

Cadmium-style examples:
{CADMIUM_STYLE_EXAMPLES}

Compact structure summary:
{analysis_text}

Raw Cadmium-lite JSON:
{json_text}

Now output only the final Cadmium-style English description.
""".strip()


# =====================
# LLM calls and rule-based fallback
# =====================

def call_llm_desc(
    json_data: Dict[str, Any],
    dims: Dict[str, str],
    analysis: Dict[str, Any],
    images: List[Path],
    raw_json_text: Optional[str] = None,
    retry: Optional[int] = None,
) -> Optional[str]:
    if retry is None:
        retry = int(LLM_CONFIG["max_retries"])

    user_text = build_description_prompt(
        json_data=json_data,
        dims=dims,
        analysis=analysis,
        raw_json_text=raw_json_text,
    )

    if USE_IMAGES:
        content: Any = [{"type": "text", "text": user_text}]
        for img in images:
            try:
                content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img)}})
            except Exception as e:
                print(f"[IMAGE ERROR] {img.name}: {e}")
    else:
        content = user_text

    messages = [
        {
            "role": "system",
            "content": (
                "You are a CAD dataset annotator. Write detailed English procedural CAD modeling "
                "descriptions in the style of the Cadmium dataset. Do not output JSON or internal "
                "field names. Do not round, simplify, shorten, expand, or rewrite numeric values."
            ),
        },
        {"role": "user", "content": content},
    ]

    client = get_client()

    for attempt in range(1, retry + 1):
        try:
            response = client.chat.completions.create(
                model=str(LLM_CONFIG["model"]),
                messages=messages,
                temperature=float(LLM_CONFIG["temperature"]),
                top_p=float(LLM_CONFIG["top_p"]),
                max_tokens=int(LLM_CONFIG["max_tokens"]),
                timeout=int(LLM_CONFIG["generation_timeout"]),
            )

            text, err = extract_chat_text(response)
            if err is None and text:
                return cleanup_text(text)

            print(f"[LLM ERROR] attempt={attempt}, err={err}")

        except Exception as e:
            print(f"[LLM EXCEPTION] attempt={attempt}, err={e}")

        if attempt < retry:
            time.sleep(float(LLM_CONFIG["retry_sleep_seconds"]))

    return None


FORBIDDEN_APPROX_WORDS = [
    "约",
    "大约",
    "大概",
    "约等",
    "近似",
    "大致",
    "左右",
    "估计",
    "差不多",
    "基本上",
    "approximately",
    "approximate",
    "about",
    "roughly",
    "around",
    "close to",
    "near",
    "estimated",
]


def is_good_desc(text: str) -> bool:
    if not text:
        return False

    # Keep lowercase words like "circle" and "arc" allowed, because Cadmium-style descriptions
    # naturally use them. Block only schema-looking tokens and wrappers.
    bad = [
        "```", "{", "}",
        "Group", "Prism", "Face", "Wire", "Edge", "Arc", "Circle",
        "coordinates", "parameters", "features", "schema", "JSON", "json",
    ]

    if any(x in text for x in bad):
        return False

    lowered = text.lower()
    if any(x.lower() in lowered for x in FORBIDDEN_APPROX_WORDS):
        return False

    return True


def enforce_exact_wording(text: str) -> str:
    """
    Final safety pass: remove or rewrite imprecise wording before writing the file.

    This is a guardrail for both LLM output and rule fallback output. It does not
    change numeric values; it only removes approximate-language markers.
    """
    replacements = {
        "大约": "",
        "大概": "",
        "约等于": "等于",
        "约等": "",
        "近似": "",
        "大致": "",
        "左右": "",
        "估计": "",
        "差不多": "",
        "基本上": "",
        "半径约": "半径",
        "高度约": "高度",
        "厚度约": "厚度",
        "边长约": "边长",
        "尺寸约": "尺寸",
        "截面约": "截面",
        "外径约": "外径",
        "内径约": "内径",
        "外包尺寸约": "外包尺寸",
        "圆面法向约": "圆面法向为",
        "法向约": "法向为",
        "方向约": "方向为",
        "局部X方向约": "局部X方向为",
        "约": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"，{2,}", "，", text)
    text = re.sub(r"。{2,}", "。", text)
    return text


def build_rule_fallback_desc(json_data: Dict[str, Any], dims: Dict[str, str], analysis: Optional[Dict[str, Any]] = None) -> str:
    """
    Conservative English fallback used only when the LLM output is empty or invalid.

    This fallback intentionally avoids rewriting or rounding detailed numeric values. It may be less
    detailed than the LLM output, but it keeps the output language and style aligned with Cadmium.
    """
    if analysis is None:
        analysis = build_model_analysis(json_data, dims)

    counter = get_type_counter(json_data)
    standard = analysis.get("standard_shape_hint") or "solid model"
    boolean_text = analysis.get("boolean_summary") or "no obvious Boolean operation"

    parts: List[str] = [f"Begin by creating a {standard}."]

    if counter.get("Circle", 0):
        parts.append("Sketch the circular profiles indicated by the source geometry on the relevant faces.")
    if counter.get("Wire", 0) or counter.get("Edge", 0) or counter.get("Arc", 0):
        parts.append("Define the closed sketch loops from the straight line and arc segments specified by the source geometry.")
    if counter.get("Prism", 0):
        parts.append("Extrude the sketched profiles to form the solid features.")
    if counter.get("Cut", 0):
        parts.append("Use the subtractive profiles to cut through or remove material from the base solid.")
    if counter.get("Fuse", 0):
        parts.append("Fuse the additive extruded features with the main body.")

    if boolean_text and boolean_text != "no obvious Boolean operation":
        parts.append("The final object is produced by combining the described extrusions and Boolean operations.")
    else:
        parts.append("The final object is produced from the described extruded sketch geometry.")

    return " ".join(parts)


# =====================
# File processing
# =====================

def build_output_path(json_path: Path) -> Path:
    mode = os.getenv("OUTPUT_NAME_MODE", "same_id").strip().lower()

    stem = json_path.stem
    base = stem[:-5] if stem.endswith("_lite") else stem

    if mode == "description":
        filename = f"{base}_description.txt"
    else:
        filename = f"{base}_description.txt"

    # When OUTPUT_DIR is set, write all outputs there
    if OUTPUT_DIR is not None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return OUTPUT_DIR / filename

    # When OUTPUT_DIR is not set, keep the original behavior: write next to the JSON file
    return json_path.with_name(filename)



def select_images(folder: Path, max_images: int = 2) -> List[Path]:
    images = sorted(folder.glob("*.png"))
    if not images:
        return []
    priority_keywords = ["iso", "front", "top", "left", "right"]
    selected: List[Path] = []
    for key in priority_keywords:
        for img in images:
            if key in img.name.lower() and img not in selected:
                selected.append(img)
                break
    if not selected:
        selected = images[:max_images]
    return selected[:max_images]


def select_images_for_json(json_path: Path, max_images: int = 2) -> List[Path]:
    folder = json_path.parent
    stem = json_path.stem
    base = stem[:-5] if stem.endswith("_lite") else stem
    matched = sorted(folder.glob(f"{base}*.png"))
    if matched:
        return matched[:max_images]
    return select_images(folder, max_images=max_images)


def process_json_file(json_path: Path) -> None:
    print(f"processing: {json_path.name}")

    try:
        raw_json_text = json_path.read_text(encoding="utf-8")
        json_data = json.loads(raw_json_text)
    except Exception as e:
        print(f"error: {json_path.name}, failed to read JSON: {e}")
        return

    dims = extract_key_dims_from_raw_json(raw_json_text)
    analysis = build_model_analysis(json_data, dims)

    images: List[Path] = []
    if USE_IMAGES:
        images = select_images_for_json(json_path, max_images=2)
        if not images:
            print(f"warn: {json_path.name}, no related png images found; JSON only will be used")

    llm_desc = call_llm_desc(json_data, dims, analysis, images, raw_json_text=raw_json_text)

    if llm_desc and is_good_desc(llm_desc):
        final_desc = cleanup_text(llm_desc)
    else:
        print(f"[FALLBACK] {json_path.name}: LLM output is empty or invalid; using English rule fallback")
        final_desc = cleanup_text(build_rule_fallback_desc(json_data, dims, analysis))

    final_desc = enforce_exact_wording(cleanup_text(final_desc))

    if not final_desc:
        print(f"error: {json_path.name}, description generation failed")
        return

    out_path = build_output_path(json_path)
    out_path.write_text(final_desc, encoding="utf-8")
    print(f"done: {json_path.name} -> {out_path.name} -> {final_desc}")


def process_folder(folder: Path) -> None:
    json_files = sorted(folder.glob("*_lite.json"))
    if not json_files:
        print(f"skip: {folder.name}, no *_lite.json")
        return
    for json_path in json_files:
        process_json_file(json_path)
        time.sleep(SLEEP_BETWEEN)


def discover_json_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*_lite.json"))


def main() -> None:
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR does not exist: {ROOT_DIR}")

    json_files = discover_json_files(ROOT_DIR)

    print("model:", LLM_CONFIG["model"])
    print("base_url:", LLM_CONFIG["base_url"])
    print("use_images:", USE_IMAGES)
    print("root:", ROOT_DIR)
    print("total_json:", len(json_files))

    if not json_files:
        print("No *_lite.json files found. Make sure file names look like 00000007_lite.json.")
        return

    for json_path in json_files:
        try:
            process_json_file(json_path)
            time.sleep(SLEEP_BETWEEN)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"fatal error: {json_path.name}, {e}")


if __name__ == "__main__":
    main()