"""
Strict LLM-only generator for concise English natural-language CAD descriptions.

Workflow:
1. Load each Teach2CAD-lite JSON item from configs/1batch_generated.yaml.
2. Send the JSON-derived structure summary, the original JSON text, and selected PNG views
   to Qwen API through Alibaba Cloud Model Studio OpenAI-compatible chat/completions.
3. Save exactly one English description per item.
4. If the LLM call fails, returns empty content, or returns invalid content, record the
   failed filename to configs/failed_name.yaml and continue with the next item. No
   rule-based fallback is used.

Recommended environment variables for Qwen API / Alibaba Cloud Model Studio compatible OpenAI API:
export DASHSCOPE_API_KEY="<your-dashscope-api-key>"
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export QWEN_MODEL="qwen3.7-plus"
export OUTPUT_NAME_MODE="description"
export DATASET_ROOT="data/verify_data"
unset OUTPUT_DIR

# This strict version is designed to let the LLM read both image(s) and JSON.
export USE_IMAGES=1
export REQUIRE_IMAGES=1
export IMAGE_VIEWS="iso,front,top"

# Failures are recorded and the batch continues. Increase timeout only when needed.
export MAX_RETRIES=1
export GENERATION_TIMEOUT=600
export MAX_TOKENS=2048

python 1generation_description_qwen3_7.py
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import re
import time
import yaml
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]

ROOT_DIR = Path(os.getenv("DATASET_ROOT", ""))
OUTPUT_DIR_TEXT = os.getenv("OUTPUT_DIR", "").strip()
OUTPUT_DIR = Path(OUTPUT_DIR_TEXT) if OUTPUT_DIR_TEXT else None
MAX_JSON_CHARS = int(os.getenv("MAX_JSON_CHARS", "0"))
SLEEP_BETWEEN = float(os.getenv("SLEEP_BETWEEN", "0"))
DESCRIPTION_STYLE = os.getenv("DESCRIPTION_STYLE", "face_steps").strip().lower()
USE_IMAGES = os.getenv("USE_IMAGES", "1").strip().lower() in {"1", "true", "yes"}
REQUIRE_IMAGES = os.getenv("REQUIRE_IMAGES", "1").strip().lower() in {"1", "true", "yes"}
IMAGE_VIEWS = [
    v.strip().lower()
    for v in os.getenv("IMAGE_VIEWS", "iso,front").split(",")
    if v.strip()
]
# For cloud Qwen API, keep auto-detect disabled by default so a wrong model name fails fast.
AUTO_DETECT_MODEL = os.getenv("AUTO_DETECT_MODEL", "0").strip().lower() in {"1", "true", "yes"}
DISABLE_LLM = os.getenv("DISABLE_LLM", "0").strip().lower() in {"1", "true", "yes"}
BATCH_CONFIG = Path(
    "configs/batch_generated.yaml"
)
FAILED_CONFIG = Path(
    os.getenv(
        "FAILED_CONFIG",
        "configs/failed_name.yaml",
    )
)
QWEN_BASE_URL = (
    os.getenv("QWEN_BASE_URL")
    or os.getenv("DASHSCOPE_BASE_URL")
    or os.getenv("LOCAL_OPENAI_BASE_URL")
    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
QWEN_MODEL = (
    os.getenv("QWEN_MODEL")
    or os.getenv("LOCAL_DESC_MODEL")
    or os.getenv("LOCAL_MODEL")
    or "qwen3.7-plus"
)
QWEN_API_KEY = (
    os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("QWEN_API_KEY")
    or os.getenv("LOCAL_OPENAI_API_KEY")
    or ""
)

LLM_CONFIG: Dict[str, Any] = {
    "base_url": QWEN_BASE_URL,
    "api_key": QWEN_API_KEY,
    "model": QWEN_MODEL,
    "generation_timeout": int(os.getenv("GENERATION_TIMEOUT", "180")),
    "max_retries": int(os.getenv("MAX_RETRIES", "1")),
    "retry_sleep_seconds": float(os.getenv("RETRY_SLEEP_SECONDS", "1")),
    "max_tokens": int(os.getenv("MAX_TOKENS", "256")),
    "temperature": float(os.getenv("TEMPERATURE", "0")),
    "top_p": float(os.getenv("TOP_P", "1")),
}

KEY_DIM_FIELDS = [
    "height", "radius", "r", "h", "length", "width", "thickness", "diameter",
    "alpha1", "alpha2",
]
FIELD_ALIASES = {"r": "radius", "h": "height"}
EN_NAME_MAP = {
    "height": "height",
    "radius": "radius",
    "r": "radius",
    "h": "height",
    "diameter": "diameter",
    "length": "length",
    "width": "width",
    "thickness": "thickness",
    "alpha1": "start angle",
    "alpha2": "end angle",
}

_MODEL_CACHE: Optional[List[str]] = None


def _now() -> float:
    return time.perf_counter()


def _log_elapsed(name: str, start_time: float) -> None:
    print(f"[TIME] {name}: {time.perf_counter() - start_time:.2f}s")


def load_batch_items():
    if not BATCH_CONFIG.exists():
        raise FileNotFoundError(
            f"batch config not found: {BATCH_CONFIG}"
        )

    with open(BATCH_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    batch = cfg["batch"]

    start_uid = str(
        batch.get("start_uid", "")
    ).strip()

    items = batch["items"]

    return start_uid, items


def get_client() -> OpenAI:
    if OpenAI is None:
        raise ImportError("Missing dependency: please run `pip install openai`.")
    api_key = str(LLM_CONFIG["api_key"] or "").strip()
    if not api_key:
        raise EnvironmentError(
            "Missing Qwen API key. Please set DASHSCOPE_API_KEY or QWEN_API_KEY."
        )
    return OpenAI(
        base_url=str(LLM_CONFIG["base_url"]),
        api_key=api_key,
        timeout=int(LLM_CONFIG["generation_timeout"]),
        # Critical: disable automatic retries inside the OpenAI SDK.
        # Otherwise a timeout of 180 can actually take 180*3 = 540 seconds before returning.
        max_retries=0,
    )


def get_available_model_ids(client: OpenAI) -> List[str]:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    ids: List[str] = []
    try:
        models = client.models.list()
        for item in getattr(models, "data", []) or []:
            model_id = getattr(item, "id", None)
            if model_id:
                ids.append(str(model_id))
    except Exception as e:
        print(f"[MODEL WARNING] Could not list models from {LLM_CONFIG['base_url']}: {e}")
    _MODEL_CACHE = ids
    return ids


def resolve_model_name(client: OpenAI) -> str:
    requested = str(LLM_CONFIG.get("model") or "").strip()
    if not AUTO_DETECT_MODEL:
        return requested

    ids = get_available_model_ids(client)
    if not ids:
        return requested
    if requested and requested in ids:
        return requested

    chosen = ids[0]
    if requested:
        print(f"[MODEL WARNING] Requested model `{requested}` is not served by this endpoint. Using `{chosen}` instead.")
    else:
        print(f"[MODEL WARNING] LOCAL_MODEL is empty. Using served model `{chosen}`.")
    LLM_CONFIG["model"] = chosen
    return chosen


# ---------------------
# Basic utilities
# ---------------------

def pretty_json(obj: Any, max_chars: int = MAX_JSON_CHARS) -> str:
    s = json.dumps(obj, ensure_ascii=False, indent=2)

    # max_chars <= 0 means no local truncation
    if max_chars and max_chars > 0 and len(s) > max_chars:
        s = s[:max_chars] + "\n...[truncated; use the structure summary above as the primary reference]"

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
            texts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        texts.append(str(item["text"]))
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
    if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) < 3:
        return str(point)
    return f"({short_num(point[0], digits)},{short_num(point[1], digits)},{short_num(point[2], digits)})"


def _vec3(values: Any) -> Optional[Tuple[float, float, float]]:
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


AXIS_EPS = 1e-8


def vec_txt(vec: Any, digits: int = 5) -> str:
    v = _normalize_vec(vec)
    if v is None:
        raw = _vec3(vec)
        return point_xyztxt(raw, digits) if raw else "unknown direction"
    return point_xyztxt(v, digits)


def dominant_axis_label(vec: Any, eps: float = AXIS_EPS) -> str:
    v = _normalize_vec(vec)
    if v is None:
        return "unknown direction"
    names = ["X", "Y", "Z"]
    abs_values = [abs(v[0]), abs(v[1]), abs(v[2])]
    axis = max(range(3), key=lambda i: abs_values[i])
    if abs_values[axis] >= 1.0 - eps and all(abs(v[i]) <= eps for i in range(3) if i != axis):
        return ("+" if v[axis] >= 0 else "-") + names[axis]
    return vec_txt(v, digits=17)


def plane_label_from_normal(normal: Any) -> str:
    label = dominant_axis_label(normal)
    if label.endswith("X"):
        return "YZ plane"
    if label.endswith("Y"):
        return "XZ plane"
    if label.endswith("Z"):
        return "XY plane"
    if label != "unknown direction":
        return f"a plane perpendicular to normal {label}"
    return "unknown plane"


def local_frame_phrase(frame: Optional[Dict[str, Any]]) -> str:
    if not frame:
        return ""
    normal = frame.get("normal")
    x_dir = frame.get("x_dir")
    parts: List[str] = []
    if normal:
        parts.append(f"on the {plane_label_from_normal(normal)} with normal {dominant_axis_label(normal)}")
    if x_dir:
        parts.append(f"local X direction {dominant_axis_label(x_dir)}")
    return " and ".join(parts)


def get_face_frame(face_node: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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


def first_face(prism_node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for child in prism_node.get("features", []):
        if isinstance(child, dict) and child.get("type") == "Face":
            return child
    return None


def get_height(prism_node: Dict[str, Any]) -> Optional[float]:
    try:
        return float(prism_node.get("parameters", {}).get("height"))
    except Exception:
        return None


def get_prism_extrude_info(prism_node: Dict[str, Any], face_node: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = prism_node.get("parameters", {}) if isinstance(prism_node, dict) else {}
    frame = get_face_frame(face_node or first_face(prism_node))
    face_normal = _normalize_vec(frame.get("normal"))
    declared = get_prism_declared_direction(prism_node)
    auto_normal = bool(params.get("auto_normal_vector"))

    if auto_normal and face_normal is not None:
        direction = face_normal
        source = "face normal"
    elif declared is not None:
        direction = declared
        source = "given direction"
    elif face_normal is not None:
        direction = face_normal
        source = "face normal"
    else:
        direction = None
        source = "unknown direction"

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
    source = extrude_info.get("source") or "direction"
    return f"along the {source} ({dominant_axis_label(direction)})"


# ---------------------
# Geometry summary
# ---------------------

def get_type_counter(json_data: Dict[str, Any]) -> Counter:
    return Counter(node.get("type") for node in walk_nodes(json_data) if isinstance(node, dict))


def get_circle_info(circle_node: Dict[str, Any]) -> Dict[str, Any]:
    coords = circle_node.get("coordinates", [])
    center = coords[0] if isinstance(coords, list) and len(coords) > 0 else None
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


def face_children_by_type(face_node: Dict[str, Any], type_name: str) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for child in face_node.get("features", []):
        if not isinstance(child, dict):
            continue
        if child.get("type") == type_name:
            result.append(child)
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
        candidates = coords[:2] if prim.get("type") == "Edge" else coords[:3]
        for p in candidates:
            if isinstance(p, list) and len(p) == 3 and all(isinstance(v, (int, float)) for v in p):
                if p not in points:
                    points.append(p)
                if len(points) >= max_points:
                    return points
    return points


def profile_bbox_from_wire(wire_node: Dict[str, Any]) -> Optional[Tuple[float, float]]:
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


def detect_standard_shape(json_data: Dict[str, Any]) -> Optional[str]:
    c = get_type_counter(json_data)
    prism_nodes = nodes_by_type(json_data, "Prism")
    prism_summaries = [prism_profile_summary(p) for p in prism_nodes]
    if c.get("Arc", 0) >= 20 and c.get("Edge", 0) >= 20:
        return "gear-like or multi-tooth complex part"
    if c.get("Prism", 0) == 1 and c.get("Circle", 0) == 1 and c.get("Wire", 0) == 0:
        return "cylinder"
    if c.get("Circle", 0) >= 2 and c.get("Cut", 0) >= 1 and c.get("Prism", 0) >= 2 and c.get("Arc", 0) < 10:
        return "cylinder with circular holes or ring washer"
    if c.get("Circle", 0) >= 2 and c.get("Prism", 0) == 1 and c.get("Wire", 0) == 0:
        return "annular extruded body"
    if c.get("Prism", 0) == 1 and c.get("Wire", 0) == 1 and c.get("Edge", 0) == 4 and c.get("Arc", 0) == 0:
        s = prism_summaries[0] if prism_summaries else {}
        return "square prism" if s.get("profile") == "square" else "rectangular prism"
    if c.get("Prism", 0) == 1 and c.get("Wire", 0) == 1 and c.get("Edge", 0) == 3 and c.get("Arc", 0) == 0:
        return "triangular prism"
    if c.get("Prism", 0) == 1 and c.get("Arc", 0) > 0 and c.get("Edge", 0) <= 10:
        return "prism with an arc boundary"
    if c.get("Circle", 0) >= 3 and c.get("Wire", 0) >= 1 and c.get("Prism", 0) == 1:
        return "plate-like prism with multiple circular features"
    if c.get("Prism", 0) >= 10 and c.get("Fuse", 0) >= 1:
        return "complex assembly formed by fusing multiple extrusions"
    return None


def _circle_phrase(circle_info: Dict[str, Any]) -> str:
    center = circle_info.get("center")
    radius = fmt_dim_value(circle_info.get("radius"))
    parts: List[str] = []
    if center:
        parts.append(f"center {point_xyztxt(center)}")
    if radius:
        parts.append(f"radius {radius}")
    return " and ".join(parts) if parts else "a circular feature"


def _append_frame_to_desc(desc: str, frame: Optional[Dict[str, Any]]) -> str:
    phrase = local_frame_phrase(frame)
    if phrase:
        return f"{desc}, {phrase}"
    return desc


def describe_face_for_prism(prism_node: Dict[str, Any]) -> str:
    summary = prism_profile_summary(prism_node)
    profile = summary.get("profile", "unknown")
    circles = summary.get("circles") or []
    wires = summary.get("wires") or []
    face_frame = summary.get("face_frame") or {}
    circle_phrases = [_circle_phrase(c) for c in circles[:4]]

    if profile == "circle" and circles:
        return _append_frame_to_desc(f"Draw a circular face with {circle_phrases[0]}", face_frame)

    if profile == "multi_circle" and circle_phrases:
        return _append_frame_to_desc("Draw multiple circular features to form a face, " + "; ".join(circle_phrases), face_frame)

    if wires:
        w = wires[0]
        bbox = w.get("bbox")
        edge_count = int(w.get("edge_count") or 0)
        arc_count = int(w.get("arc_count") or 0)
        points = w.get("points") or []
        point_text = ", ".join(point_xyztxt(p) for p in points[:6])

        if profile == "square" and bbox:
            side = short_num(max(bbox))
            desc = f"Draw a square face with side length {side}"
        elif profile == "rectangle" and bbox:
            desc = f"Draw a rectangular face of size {short_num(bbox[0])}×{short_num(bbox[1])}"
        elif profile == "triangle":
            desc = f"Draw a triangular face by connecting key points {point_text}" if point_text else "Draw a triangular face"
        elif profile == "arc_wire":
            if point_text:
                desc = (
                    f"Draw a closed profile face through key points {point_text}; "
                    f"the profile contains {edge_count} straight line(s) and {arc_count} circular arc(s)"
                )
            else:
                desc = f"Draw a closed profile face containing {edge_count} straight line(s) and {arc_count} circular arc(s)"
        elif profile in {"polygon", "quadrilateral"}:
            desc = f"Draw a closed polygon face by connecting key points {point_text}" if point_text else f"Draw a polygon face made of {edge_count} straight line(s)"
        else:
            desc = f"Draw a closed profile face through key points {point_text}" if point_text else "Draw a 2D profile face"

        if circle_phrases:
            desc += ", with internal circular feature(s): " + "; ".join(circle_phrases)
        return _append_frame_to_desc(desc, face_frame)

    if circle_phrases:
        return _append_frame_to_desc("Draw circular feature(s) to form a face, " + "; ".join(circle_phrases), face_frame)
    return _append_frame_to_desc("Draw a 2D profile face", face_frame)


def describe_prism_operation(prism_node: Dict[str, Any], parent_type: str, child_index: int) -> str:
    h_value = get_height(prism_node)
    h = fmt_dim_value(h_value)
    height_text = f"with height {h}" if h else "with the given height"
    reverse = ", in the reverse direction" if isinstance(h_value, (int, float)) and h_value < 0 else ""
    direction = extrude_phrase(get_prism_extrude_info(prism_node))
    direction_text = direction if direction else "along the face normal or given direction"

    if parent_type == "Cut" and child_index == 0:
        return f"then extrude it {direction_text} {height_text}{reverse} as the base solid before cutting"
    if parent_type == "Cut" and child_index >= 1:
        return f"then extrude it {direction_text} {height_text}{reverse} to form a cutting solid and subtract it from the main body"
    if parent_type == "Fuse":
        return f"then extrude it {direction_text} {height_text}{reverse} to form a solid and fuse it with the other solid(s)"
    return f"then extrude it {direction_text} {height_text}{reverse} to form a 3D solid"


def iter_prism_contexts(
    node: Any,
    parent_type: str = "",
    child_index: int = -1,
    path: str = "$",
) -> Iterable[Tuple[Dict[str, Any], str, int, str]]:
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


def build_face_operation_steps(json_data: Dict[str, Any], max_items: int = 8) -> List[str]:
    steps: List[str] = []
    for idx, (prism_node, parent_type, child_index, _path) in enumerate(iter_prism_contexts(json_data), start=1):
        if idx > max_items:
            remaining = len(nodes_by_type(json_data, "Prism")) - max_items
            if remaining > 0:
                steps.append(f"Continue the remaining {remaining} extrusion(s) from their respective profile faces, then fuse or cut them as defined.")
            break
        face_desc = describe_face_for_prism(prism_node)
        op_desc = describe_prism_operation(prism_node, parent_type, child_index)
        steps.append(f"{face_desc}, {op_desc}")
    return steps


def face_steps_to_sentence(steps: Sequence[str], max_steps: int = 3) -> str:
    if not steps:
        return ""
    selected = [s.rstrip(" .;；。") for s in list(steps[:max_steps])]
    text = "; ".join(selected)
    if len(steps) > max_steps:
        text += f"; continue the remaining {len(steps) - max_steps} local feature(s) from their own profile faces, then extrude, fuse, or cut them"
    return text + "."


def build_coordinate_brief(json_data: Dict[str, Any], max_wires: int = 2, max_circles: int = 4) -> str:
    pieces: List[str] = []
    wires = nodes_by_type(json_data, "Wire")
    wires_sorted = sorted(wires, key=lambda w: len(wire_primitives(w)), reverse=True)
    for i, wire_node in enumerate(wires_sorted[:max_wires], start=1):
        primitives = wire_primitives(wire_node)
        pts = wire_points(wire_node, max_points=6)
        if not pts:
            continue
        edge_count = sum(1 for p in primitives if p.get("type") == "Edge")
        arc_count = sum(1 for p in primitives if p.get("type") == "Arc")
        pieces.append(
            f"profile {i} has {edge_count} straight line(s), {arc_count} circular arc(s), and key points "
            + ", ".join(point_xyztxt(p) for p in pts[:6])
        )

    circle_texts = []
    for circle_node in nodes_by_type(json_data, "Circle")[:max_circles]:
        info = get_circle_info(circle_node)
        r = fmt_dim_value(info.get("radius"))
        center = info.get("center")
        axis = info.get("axis") or info.get("normal")
        if center and r:
            t = f"center {point_xyztxt(center)} and radius {r}"
            if axis:
                t += f", circle normal {dominant_axis_label(axis)}, on the {plane_label_from_normal(axis)}"
            circle_texts.append(t)
    if circle_texts:
        pieces.append("circular feature(s): " + "; ".join(circle_texts))

    frame_texts = []
    for idx, prism_node in enumerate(nodes_by_type(json_data, "Prism")[:6], start=1):
        s = prism_profile_summary(prism_node)
        frame_text = local_frame_phrase(s.get("face_frame"))
        extrude_text = extrude_phrase(s.get("extrude"))
        if frame_text or extrude_text:
            frame_texts.append(f"extrusion {idx}: " + ", ".join(p for p in [frame_text, f"extruded {extrude_text}" if extrude_text else ""] if p))
    if frame_texts:
        pieces.append("local directions: " + "; ".join(frame_texts))

    heights = [short_num(h) for p in nodes_by_type(json_data, "Prism")[:6] if (h := get_height(p)) is not None]
    if heights:
        pieces.append("main extrusion height(s): " + ", ".join(heights))
    return "; ".join(pieces)


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
            pieces.append(f"extruded {extrude_text}")
        if pieces:
            brief.append(f"extrusion {idx}: " + ", ".join(pieces))
    return brief


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
            text = f"extrusion {idx}: circular cross-section"
            if r:
                text += f", radius {r}"
            if center:
                text += f", center {point_xyztxt(center)}"
            if circle.get("axis"):
                text += f", circle normal {dominant_axis_label(circle.get('axis'))}"
            if frame_text:
                text += f", {frame_text}"
            if h:
                text += f", height {h}"
            if extrude_text:
                text += f", extruded {extrude_text}"
            items.append(text)
            continue

        if profile in {"rectangle", "square", "quadrilateral", "triangle", "polygon", "arc_wire"}:
            wire = s["wires"][0] if s.get("wires") else {}
            bbox = wire.get("bbox")
            edge_count = wire.get("edge_count", 0)
            arc_count = wire.get("arc_count", 0)
            points = wire.get("points", [])
            text = f"extrusion {idx}: "
            if profile == "rectangle":
                text += "rectangular profile"
            elif profile == "square":
                text += "square profile"
            elif profile == "triangle":
                text += "triangular profile"
            elif profile == "arc_wire":
                text += f"profile with {edge_count} straight line(s) and {arc_count} circular arc(s)"
            else:
                text += f"profile with {edge_count} straight line(s)"
            if bbox:
                text += f", bounding size {short_num(bbox[0])}×{short_num(bbox[1])}"
            if frame_text:
                text += f", {frame_text}"
            if h:
                text += f", height {h}"
            if extrude_text:
                text += f", extruded {extrude_text}"
            if points:
                text += ", key points " + ", ".join(point_xyztxt(p) for p in points[:4])
            items.append(text)
            continue

        text = f"extrusion {idx}: profile type {profile}"
        if frame_text:
            text += f", {frame_text}"
        if h:
            text += f", height {h}"
        if extrude_text:
            text += f", extruded {extrude_text}"
        items.append(text)
    return items


def build_model_analysis(json_data: Dict[str, Any], dims: Dict[str, str]) -> Dict[str, Any]:
    counter = get_type_counter(json_data)
    standard_shape = detect_standard_shape(json_data)
    boolean_text = []
    if counter.get("Fuse", 0):
        boolean_text.append(f"{counter.get('Fuse', 0)} fuse operation(s)")
    if counter.get("Cut", 0):
        boolean_text.append(f"{counter.get('Cut', 0)} cut operation(s)")
    return {
        "standard_shape_hint": standard_shape or "",
        "node_counts": {k: int(v) for k, v in sorted(counter.items()) if k},
        "boolean_summary": ", ".join(boolean_text) if boolean_text else "no obvious boolean operation",
        "key_dimensions": {k: fmt_dim_value(v) for k, v in dims.items()},
        "prism_summaries": summarize_prisms(json_data, max_items=8),
        "face_operation_steps": build_face_operation_steps(json_data, max_items=8),
        "coordinate_frame_brief": build_coordinate_frame_brief(json_data, max_items=8),
        "coordinate_brief": build_coordinate_brief(json_data),
    }


def analysis_to_text(analysis: Dict[str, Any]) -> str:
    return json.dumps(analysis, ensure_ascii=False, indent=2)


# ---------------------
# English LLM prompt
# ---------------------

FEW_SHOT_NATURAL_DESCRIPTIONS = """
Reference output style. Learn the wording style only; do not copy the dimensions:
1. Draw a circular face with center (0,0,0) and radius 0.75, on the XY plane with normal +Z, then extrude it along the face normal (+Z) with height 0.209 to form a cylinder.
2. Draw a closed profile face made of 4 straight line(s) and 1 circular arc(s), with key points (0,-0.75,0), (0.75,-0.75,0), (0.75,0,0), and (0.075,0,0), on the XY plane with normal +Z, then extrude it along the face normal (+Z) with height 0.0375 to form a prism with an arc boundary.
3. Draw a square face with side length 0.3, on the XZ plane with normal -Y and local X direction +X, then extrude it along the face normal (-Y) with height 0.75 to form a square prism.
4. Draw a circular face with radius 0.75 and another circular face with radius 0.4004, both concentric at (0,0,0), on the XZ plane with normal -Y and local X direction +X, then extrude the face along the face normal (-Y) with height 0.4629 to form a solid; draw another circular face with center (0,0,0) and radius 0.4004 on the same plane, extrude it with height 0.4629, and fuse the two extruded solids.
5. Draw a circular face with radius 0.75 on the XY plane with normal +Z, then extrude it along the face normal (+Z) with height 0.0381 to form a thin cylindrical disk.
""".strip()


def build_description_prompt(
    json_data: Dict[str, Any],
    dims: Dict[str, str],
    analysis: Dict[str, Any],
    raw_json_text: str,
) -> str:
    json_text = raw_json_text.strip() or pretty_json(json_data, max_chars=MAX_JSON_CHARS)

    # MAX_JSON_CHARS <= 0 means no local JSON truncation
    if MAX_JSON_CHARS and MAX_JSON_CHARS > 0 and len(json_text) > MAX_JSON_CHARS:
        json_text = json_text[:MAX_JSON_CHARS] + "\n...[truncated because MAX_JSON_CHARS was reached]"

    analysis_text = analysis_to_text(analysis)

    return f"""
/no_think
You are an English CAD description writer. Generate one concise natural-language CAD modeling description from the provided CAD structure summary, raw CAD JSON, and render image(s).

Important priority:
1. Use the render image(s) and CAD structure summary first.
2. Use the raw JSON only to verify dimensions, extrusion directions, planes, and boolean operations.
3. Do not expand every low-level edge, vertex, or coordinate.
4. For complex profiles with many small edges, summarize the repeated pattern as a high-level shape.

CAD structure summary:
{analysis_text}

Raw Teach2CAD-lite JSON:
{json_text}

Special rule for gear-like or multi-tooth parts:
If the model contains a toothed outer rim, annular ring, radial spokes, central hub, or center hole, describe it at feature level.
Use words such as "toothed outer annular rim", "inner circular opening", "radial spokes", "central cylindrical hub", and "concentric circular hole".
Do not list individual tooth coordinates or individual edge points.

Reference examples:
1. Draw a toothed annular gear rim on the XY plane, add four radial spokes and a central cylindrical hub with a concentric circular hole, then extrude the profiles along +Z and fuse them into one gear-like wheel.
2. Draw an annular wheel with an outer toothed rim, an inner circular opening, four evenly spaced spokes, and a raised central hub, then extrude all profiles along +Z with the given height.
3. Draw a circular gear-like plate with repeated outer teeth, four radial supports, and a central through-hole, then extrude it along the face normal to form a single fused solid.

Output requirements:
1. Output only one English paragraph.
2. Describe the overall CAD modeling process, not every JSON primitive.
3. Preserve only the important exact values: main radii, extrusion height, plane, direction, and boolean relation when available.
4. Do not mention JSON field/type names such as Group, Prism, Face, Wire, Edge, Arc, Circle, coordinates, parameters, schema, or node.
5. Do not invent material, color, function, or purpose.
6. Do not use vague words such as approximately, about, roughly, around, estimated, nearly, or almost.
7. The final output must be no more than 300 characters.

Return the final English description only.
""".strip()


# ---------------------
# Strict LLM call; no rule fallback is used
# ---------------------

FORBIDDEN_APPROX_WORDS = [
    "approximately", "approx.", "about", "roughly", "around", "estimated", "nearly", "almost", "circa",
    "约", "大约", "大概", "近似", "左右", "估计",
]

class FatalLLMError(RuntimeError):
    pass

def call_llm_desc(
    json_data: Dict[str, Any],
    dims: Dict[str, str],
    analysis: Dict[str, Any],
    images: List[Path],
    raw_json_text: str,
    retry: Optional[int] = None,
) -> Optional[str]:
    if DISABLE_LLM:
        print("[LLM ERROR] DISABLE_LLM=1, but this script is LLM-only. Generation cannot continue.")
        return None
    if retry is None:
        retry = int(LLM_CONFIG["max_retries"])

    user_text = build_description_prompt(json_data, dims, analysis, raw_json_text)
    if USE_IMAGES:
        content: Any = [{"type": "text", "text": user_text}]
        for img in images:
            try:
                content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img)}})
            except Exception as e:
                print(f"[IMAGE ERROR] {img}: {e}")
                return None
    else:
        content = user_text

    messages = [
        {
            "role": "system",
            "content": (
                "You are an English CAD dataset annotator. Convert Teach2CAD-lite CAD JSON "
                "and CAD render image(s) into one concise English natural-language modeling description. "
                "Preserve geometry, dimensions, directions, planes, local frames, and boolean relations. "
                "Do not expose internal JSON fields. "
                "Do not output reasoning, analysis, or chain-of-thought; return only the final paragraph."
            ),
        },
        {"role": "user", "content": content},
    ]

    client = get_client()
    model_name = resolve_model_name(client)
    if not model_name:
        print("[MODEL ERROR] No model id is configured and model auto-detection found nothing. Generation cannot continue.")
        return None

    print(
        "[LLM CONFIG] "
        f"model={model_name}, images={len(images) if USE_IMAGES else 0}, "
        f"prompt_chars={len(user_text)}, max_tokens={LLM_CONFIG['max_tokens']}, "
        f"timeout={LLM_CONFIG['generation_timeout']}, retries={retry}"
    )

    for attempt in range(1, retry + 1):
        t_llm = _now()
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=float(LLM_CONFIG["temperature"]),
                top_p=float(LLM_CONFIG["top_p"]),
                max_tokens=int(LLM_CONFIG["max_tokens"]),
                timeout=int(LLM_CONFIG["generation_timeout"]),
                extra_body={
                    "enable_thinking": False
                },
            )

            choice = response.choices[0] if getattr(response, "choices", None) else None
            finish_reason = getattr(choice, "finish_reason", None) if choice else None
            usage = getattr(response, "usage", None)

            print(
                f"[LLM TIME] attempt={attempt}, "
                f"elapsed={time.perf_counter() - t_llm:.2f}s, "
                f"finish_reason={finish_reason}, "
                f"usage={usage}"
            )

            if finish_reason == "length":
                raise FatalLLMError(
                    f"LLM output was truncated because it reached max_tokens={LLM_CONFIG['max_tokens']}. "
                    f"Increase MAX_TOKENS or reduce prompt input."
                )

            completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
            if completion_tokens is not None and completion_tokens >= int(LLM_CONFIG["max_tokens"]):
                raise FatalLLMError(
                    f"LLM output may be truncated: completion_tokens={completion_tokens}, "
                    f"max_tokens={LLM_CONFIG['max_tokens']}."
                )

            if finish_reason not in {"stop", None}:
                raise FatalLLMError(f"LLM stopped abnormally: finish_reason={finish_reason}")

            text, err = extract_chat_text(response)
            if err is None and text:
                return cleanup_text(text)
            print(f"[LLM ERROR] attempt={attempt}, err={err}")
        except FatalLLMError:
            raise
        except Exception as e:
            print(f"[LLM EXCEPTION] attempt={attempt}, elapsed={time.perf_counter() - t_llm:.2f}s, err={e}")
        if attempt < retry:
            time.sleep(float(LLM_CONFIG["retry_sleep_seconds"]))
    return None

def enforce_exact_wording(text: str) -> str:
    replacements = {
        "approximately": "",
        "Approximately": "",
        "approx.": "",
        "Approx.": "",
        "about": "",
        "About": "",
        "roughly": "",
        "Roughly": "",
        "around": "",
        "Around": "",
        "estimated": "",
        "Estimated": "",
        "nearly": "",
        "Nearly": "",
        "almost": "",
        "Almost": "",
        "circa": "",
        "Circa": "",

    # Avoid being falsely rejected by is_good_desc
        "coordinates": "points",
        "coordinate": "point",
        "features": "elements",
        "feature": "element",
        "parameters": "values",
        "parameter": "value",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\.\s*\.+", ".", text)
    return text

def cleanup_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[：:，,。\.\s]+", "", text)
    text = re.sub(r"[,，]{2,}", ",", text)
    text = re.sub(r"[。\.]{2,}", ".", text)

    bad_prefixes = [
        "Natural language description:",
        "Description:",
        "Final description:",
        "Model description:",
        "Output:",
        "Answer:",
        "自然语言描述：",
        "描述：",
        "最终描述：",
        "模型描述：",
        "输出：",
        "答案：",
    ]

    lower_text = text.lower()
    for p in bad_prefixes:
        if lower_text.startswith(p.lower()):
            text = text[len(p):].strip()
            lower_text = text.lower()

    # Critical: do not truncate length anywhere
    return text.strip()


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def has_unfinished_tail(text: str) -> bool:
    """
    Detect obviously incomplete English CAD descriptions.

    Examples to reject:
    - then extrude it along the +.
    - then extrude it along the.
    - with a height of.
    - along the +Z axis with a height of
    """
    if not text:
        return True

    s = text.strip()
    s_no_period = s.rstrip(" .;，；。").strip()

    # 1. CAD description should normally end with a complete punctuation mark.
    if not s.endswith((".", "!", "?")):
        return True

    # 2. Reject broken endings.
    bad_endings = [
        "along the +",
        "along the -",
        "along the",
        "along",
        "with height",
        "with a height",
        "with a height of",
        "height of",
        "with",
        "of",
        "to",
        "from",
        "and",
        "or",
        "the",
        "a",
        "an",
        "+",
        "-",
        "+x",
        "-x",
        "+y",
        "-y",
        "+z",
        "-z",
        "+x axis",
        "-x axis",
        "+y axis",
        "-y axis",
        "+z axis",
        "-z axis",
    ]

    low = s_no_period.lower()
    if any(low.endswith(x) for x in bad_endings):
        return True

    # 3. Reject unbalanced brackets.
    if s.count("(") != s.count(")"):
        return True

    # 4. Reject sentence ending with a dangling comma/semicolon.
    if s.endswith((",", ";", ":", "，", "；", "：")):
        return True

    return False


def validate_desc(text: str) -> List[str]:
    reasons: List[str] = []

    if not text or not text.strip():
        reasons.append("empty text")
        return reasons

    bad = [
        "```", "{", "}",
        "Group", "Prism", "Face", "Wire", "Edge", "Arc", "Circle",
        "coordinates", "parameters", "schema", "JSON", "json", "node",
    ]

    hits = [x for x in bad if x in text]
    if hits:
        reasons.append("contains forbidden internal words: " + ", ".join(hits))

    approx_hits = [x for x in FORBIDDEN_APPROX_WORDS if x.lower() in text.lower()]
    if approx_hits:
        reasons.append("contains approximate words: " + ", ".join(approx_hits))

    if contains_cjk(text):
        reasons.append("contains CJK characters")

    if has_unfinished_tail(text):
        reasons.append("looks incomplete or has an unfinished tail")

    return reasons


def is_good_desc(text: str) -> bool:
    if not text:
        return False

    bad = [
        "```", "{", "}",
        "Group", "Prism", "Wire", "Edge", "Arc", "Circle",
        "schema", "JSON", "json", "node",
    ]

    if any(x in text for x in bad):
        return False

    if any(x.lower() in text.lower() for x in FORBIDDEN_APPROX_WORDS):
        return False

    if contains_cjk(text):
        return False

    # No length limit anymore; length control is requested in the prompt only
    return True

# ---------------------
# Failure recording
# ---------------------

def _safe_reason_text(reason: Any, max_chars: int = 2000) -> str:
    text = str(reason).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        text = text[:max_chars] + "...[truncated]"
    return text


def load_failed_records() -> List[Dict[str, Any]]:
    if not FAILED_CONFIG.exists():
        return []
    try:
        cfg = yaml.safe_load(FAILED_CONFIG.read_text(encoding="utf-8")) or {}
        records = cfg.get("failed", [])
        return records if isinstance(records, list) else []
    except Exception as e:
        print(f"[FAILED LOG WARNING] could not read {FAILED_CONFIG}: {e}")
        return []


def write_failed_records(records: List[Dict[str, Any]]) -> None:
    FAILED_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "failed": records,
        "count": len(records),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    FAILED_CONFIG.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def append_failed_record(
    *,
    uid: str,
    json_path: Path,
    reason: Any,
    item: Optional[Dict[str, Any]] = None,
) -> None:
    """Append or update one failed item in FAILED_CONFIG.

    The key is uid when available; otherwise the absolute JSON path.
    The file is written immediately after each failure so interrupted runs still keep records.
    """
    records = load_failed_records()
    key = uid or str(json_path)

    # Avoid duplicate failed entries for the same uid/path across reruns.
    deduped: List[Dict[str, Any]] = []
    for rec in records:
        rec_key = str(rec.get("uid") or rec.get("minimal_json") or "")
        if rec_key and rec_key != key:
            deduped.append(rec)

    output_path = ""
    try:
        output_path = str(build_output_path(json_path))
    except Exception:
        output_path = ""

    record: Dict[str, Any] = {
        "uid": str(uid),
        "file_name": json_path.name,
        "minimal_json": str(json_path),
        "output_path": output_path,
        "reason": _safe_reason_text(reason),
        "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if isinstance(item, dict) and "images" in item:
        record["images"] = item.get("images")

    deduped.append(record)
    write_failed_records(deduped)
    print(f"[FAILED LOG] recorded failed item to {FAILED_CONFIG}: {json_path.name}")

# ---------------------
# Failure recording
# ---------------------

def _safe_reason_text(reason: Any, max_chars: int = 2000) -> str:
    text = str(reason).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        text = text[:max_chars] + "...[truncated]"
    return text


def load_failed_records() -> List[Dict[str, Any]]:
    if not FAILED_CONFIG.exists():
        return []

    try:
        cfg = yaml.safe_load(FAILED_CONFIG.read_text(encoding="utf-8")) or {}
        records = cfg.get("failed", [])
        return records if isinstance(records, list) else []
    except Exception as e:
        print(f"[FAILED LOG WARNING] could not read {FAILED_CONFIG}: {e}")
        return []


def write_failed_records(records: List[Dict[str, Any]]) -> None:
    FAILED_CONFIG.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "failed": records,
        "count": len(records),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    FAILED_CONFIG.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def append_failed_record(
    *,
    uid: str,
    json_path: Path,
    reason: Any,
    item: Optional[Dict[str, Any]] = None,
) -> None:
    records = load_failed_records()

    # Avoid recording the same uid too many times within one run or across runs
    deduped: List[Dict[str, Any]] = []
    for rec in records:
        old_uid = str(rec.get("uid", ""))
        old_json = str(rec.get("minimal_json", ""))
        if old_uid == str(uid) or old_json == str(json_path):
            continue
        deduped.append(rec)

    try:
        output_path = str(build_output_path(json_path))
    except Exception:
        output_path = ""

    record: Dict[str, Any] = {
        "uid": str(uid),
        "file_name": json_path.name,
        "minimal_json": str(json_path),
        "output_path": output_path,
        "reason": _safe_reason_text(reason),
        "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if isinstance(item, dict) and "images" in item:
        record["images"] = item.get("images")

    deduped.append(record)
    write_failed_records(deduped)

    print(f"[FAILED LOG] recorded failed item to {FAILED_CONFIG}: {json_path.name}")

# ---------------------
# File processing
# ---------------------

def build_output_path(json_path: Path) -> Path:
    mode = os.getenv("OUTPUT_NAME_MODE", "same_id").strip().lower()
    stem = json_path.stem
    base = stem[:-5] if stem.endswith("_lite") else stem
    filename = f"{base}_description.txt" if mode == "description" else f"{base}_description.txt"
    if OUTPUT_DIR is not None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return OUTPUT_DIR / filename
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


def select_images_from_batch_item(item: Dict[str, Any], fallback_json_path: Path) -> List[Path]:
    """Select image paths for the strict multimodal LLM call.

    Priority:
    1. Use image paths from 1batch_generated.yaml according to IMAGE_VIEWS.
    2. If the batch item has no usable image paths, fall back to files next to the JSON.
    3. If REQUIRE_IMAGES=1 and no valid image is found, raise and stop generation.
    """
    selected: List[Path] = []

    img_dict = item.get("images", {}) if isinstance(item, dict) else {}
    if USE_IMAGES and isinstance(img_dict, dict):
        for view in IMAGE_VIEWS:
            path_text = img_dict.get(view)
            if path_text:
                selected.append(Path(path_text))

    if USE_IMAGES and not selected:
        selected = select_images_for_json(fallback_json_path, max_images=max(1, len(IMAGE_VIEWS)))

    valid: List[Path] = []
    missing: List[Path] = []
    for p in selected:
        if p.exists():
            valid.append(p)
        else:
            missing.append(p)

    if missing:
        raise RuntimeError("image path listed in batch config does not exist: " + ", ".join(str(p) for p in missing))

    if USE_IMAGES and REQUIRE_IMAGES and not valid:
        raise RuntimeError(f"USE_IMAGES=1 and REQUIRE_IMAGES=1, but no valid image was found for {fallback_json_path}")

    return valid


def process_json_file(json_path: Path, item: Optional[Dict[str, Any]] = None, uid: Optional[str] = None) -> bool:
    t_total = _now()
    print(f"processing: {json_path.name}")

    out_path = build_output_path(json_path)
    if out_path.exists():
        print(f"[SKIP] already generated: {out_path.name}")
        return True

    try:
        t = _now()
        raw_json_text = json_path.read_text(encoding="utf-8")
        json_data = json.loads(raw_json_text)
        _log_elapsed("read_json", t)
    except Exception as e:
        raise RuntimeError(f"failed to read json: {json_path.name}\n{e}")

    t = _now()
    dims = extract_key_dims_from_raw_json(raw_json_text)
    analysis = build_model_analysis(json_data, dims)
    _log_elapsed("build_analysis", t)

    images: List[Path] = []
    if USE_IMAGES:
        t = _now()
        images = select_images_from_batch_item(item or {}, json_path)
        total_img_mb = sum(p.stat().st_size for p in images if p.exists()) / 1024 / 1024
        print(f"[IMAGE] views={','.join(IMAGE_VIEWS)}, count={len(images)}, size={total_img_mb:.2f}MB")
        for img in images:
            print(f"[IMAGE] {img}")
        _log_elapsed("select_images", t)

    t = _now()
    llm_desc = call_llm_desc(json_data, dims, analysis, images, raw_json_text)
    _log_elapsed("llm_call", t)

    if llm_desc is None:
        raise RuntimeError(f"LLM generation failed: {json_path.name}. Stop generation; no rule fallback is allowed.")

    final_desc = cleanup_text(llm_desc)
    final_desc = enforce_exact_wording(final_desc)
    final_desc = cleanup_text(final_desc)

    validation_errors = validate_desc(final_desc)
    if validation_errors:
        raise RuntimeError(
            f"Bad LLM output after cleanup: {json_path.name}\n"
            f"Reasons:\n- " + "\n- ".join(validation_errors) + "\n"
            f"Output:\n{final_desc[:1200]}"
    )

    # Saving is allowed only after the completeness check passes
    out_path.write_text(final_desc.rstrip() + "\n", encoding="utf-8")
    print(f"[DONE] {uid or json_path.stem} -> {out_path}")
    _log_elapsed("total_one_file", t_total)
    return True


def process_batch_item(item: Dict[str, Any]) -> bool:
    uid = item["uid"]
    json_path = Path(item["minimal_json"])
    return process_json_file(json_path=json_path, item=item, uid=uid)


def main() -> None:
    t_all = _now()
    start_uid, items = load_batch_items()

    print("====================================")
    print("mode: STRICT LLM-ONLY; failures are recorded and batch continues")
    print("model:", LLM_CONFIG["model"] or "<auto-detect>")
    print("base_url:", LLM_CONFIG["base_url"])
    print("use_images:", USE_IMAGES)
    print("require_images:", REQUIRE_IMAGES)
    print("image_views:", IMAGE_VIEWS)
    print("max_tokens:", LLM_CONFIG["max_tokens"])
    print("generation_timeout:", LLM_CONFIG["generation_timeout"])
    print("max_retries:", LLM_CONFIG["max_retries"])
    print("start_uid:", start_uid)
    print("total_items:", len(items))
    print("failed_config:", FAILED_CONFIG)
    print("====================================")

    started = start_uid == ""
    success_count = 0
    failed_count = 0
    skipped_before_start_uid = 0

    # Clear this run's failure records before every run.
    # Comment out the next line to keep historical failure records.
    write_failed_records([])

    for idx, item in enumerate(items):
        uid = str(item.get("uid", ""))
        json_path = Path(item.get("minimal_json", ""))

        if not started:
            if uid == start_uid:
                started = True
            else:
                skipped_before_start_uid += 1
                continue

        print(f"\n[{idx+1}/{len(items)}] processing uid={uid}")

        try:
            if not json_path.exists():
                raise FileNotFoundError(f"json not found: {json_path}")

            ok = process_json_file(json_path=json_path, item=item, uid=uid)

            if ok:
                success_count += 1
                print(f"[SUCCESS] uid={uid}")

        except KeyboardInterrupt:
            print("\nProgram interrupted by user. Stop generation.")
            raise SystemExit(130)

        except Exception as e:
            failed_count += 1

            print(f"\n[FAILED] uid={uid}")
            print(str(e))

            append_failed_record(
                uid=uid,
                json_path=json_path,
                reason=e,
                item=item,
            )

            print("\nContinue with next item.")
            continue

        if SLEEP_BETWEEN > 0:
            time.sleep(SLEEP_BETWEEN)

    print("\nFinished.")
    print(
        f"[SUMMARY] success={success_count}, failed={failed_count}, "
        f"skipped_before_start_uid={skipped_before_start_uid}, "
        f"failed_config={FAILED_CONFIG}, "
        f"elapsed={time.perf_counter() - t_all:.2f}s"
    )


if __name__ == "__main__":
    main()