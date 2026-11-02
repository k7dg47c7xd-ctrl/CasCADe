# -*- coding: utf-8 -*-
"""
Recursively walk INPUT_DIR for *_lite.json files and split each CAD modeling tree into individually renderable JSON documents.

Output rules:
1. All *_lite.json files under INPUT_DIR are discovered recursively.
2. Each model is written to an OUTPUT_DIR/<eight-digit-id>/ folder.
   Example: 00000131_lite.json -> OUTPUT_DIR/00000131/
3. Split part JSON files are named after model parts.
   Example: 01_Outer_cylinder.json, 02_Inner_bore_cut_reference_cylinder.json
4. The complete-model JSON keeps the original input file name.
   Example: 00000131_lite.json is still named 00000131_lite.json after output.
5. manifest.csv and manifest.json are written only at the top of OUTPUT_DIR.
6. Colors use the software-recognized format:
   "color": [R, G, B, A]
7. Do not write parameters["color"] and do not wrap colors in an extra Color command.
"""

from __future__ import annotations
import os

import csv
import html
import time
import argparse
import json
import math
import re
import shutil
import urllib.error
import urllib.request
import requests
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# Configure only the settings below.
# ============================================================

INPUT_DIR = r"data/40data"
OUTPUT_DIR = r"outputs/split_dataset"

# Whether to overwrite an existing eight-digit model folder in the output directory
OVERWRITE_MODEL_FOLDER = True

# Uniform color for every complete model, software format: [R, G, B, A]
FULL_MODEL_COLOR = [117, 152, 176, 0]

# Colors for split part models; must be unique within one complete model
PART_COLORS = [
    [231, 76, 60, 0],
    [46, 204, 113, 0],
    [241, 196, 15, 0],
    [155, 89, 182, 0],
    [230, 126, 34, 0],
    [26, 188, 156, 0],
    [52, 73, 94, 0],
    [255, 107, 154, 0],
    [127, 140, 141, 0],
    [22, 160, 133, 0],
    [211, 84, 0, 0],
    [142, 68, 173, 0],
    [192, 57, 43, 0],
    [39, 174, 96, 0],
    [41, 128, 185, 0],
    [243, 156, 18, 0],
    [44, 62, 80, 0],
]

# Ollama configuration: start ollama serve on this machine first and make sure the model is pulled
# Uses /api/generate, the same call style as generate_description_ollama.py
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = os.getenv("TEACHING_MODEL", "")
OLLAMA_TEMPERATURE = 0.2
MAX_JSON_CHARS = 12000
MAX_RETRY = 2
SLEEP_BETWEEN = 0.5

# True: when Ollama fails or returns an invalid result, automatically fall back to treating each candidate segment as a separate step
# False: raise an error immediately when Ollama fails, which makes environment/prompt issues easier to debug
ALLOW_RULE_FALLBACK = False

# Output file name for the step explanations
STEP_EXPLANATION_FILE = "Detailed_Step_Explanation.html"

# Extend this with the actual fields of your JSON; it makes Ollama write placeholders for dimensions first, which are then replaced with the original numbers
KEY_DIM_FIELDS = [
    "length", "width", "height", "thickness",
    "radius", "diameter", "depth",
    "outer_length", "outer_width", "outer_height",
    "inner_length", "inner_width", "inner_height",
    "slot_width", "slot_depth", "slot_height",
    "wall_thickness", "fillet_radius", "chamfer",
    "r", "d", "l", "w", "h", "t"
]

FIELD_ALIASES = {
    "r": "radius",
    "d": "diameter",
    "l": "length",
    "w": "width",
    "h": "height",
    "t": "thickness",
    "fillet_radius": "radius",
}

CN_NAME_MAP = {
    "length": "Length",
    "width": "Width",
    "height": "Height",
    "thickness": "Thickness",
    "radius": "Radius",
    "diameter": "Diameter",
    "depth": "Depth",
    "outer_length": "Outer length",
    "outer_width": "Outer width",
    "outer_height": "Outer height",
    "inner_length": "Inner length",
    "inner_width": "Inner width",
    "inner_height": "Inner height",
    "slot_width": "Slot width",
    "slot_depth": "Slot depth",
    "slot_height": "Slot height",
    "wall_thickness": "Wall thickness",
    "fillet_radius": "Fillet radius",
    "chamfer": "Chamfer size",
    "r": "Radius",
    "d": "Diameter",
    "l": "Length",
    "w": "Width",
    "h": "Height",
    "t": "Thickness",
}

# ============================================================


BOOL_TYPES = {
    "Fuse",
    "Cut",
    "Common",
    "Intersect",
    "Intersection",
    "Union",
    "Subtract",
}

JSONDict = Dict[str, Any]


@dataclass
class Part:
    name: str
    tree: JSONDict
    reason: str


def load_json(p: Path) -> JSONDict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: Path, data: JSONDict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def model_id_from_name(p: Path) -> str:
    """
    Extract the eight-digit model id from *_lite.json.
    Examples:
    00000131_lite.json -> 00000131
    full 00000131_lite.json -> 00000131
    """

    m = re.match(r"(.+?)_lite\.json$", p.name, flags=re.I)
    raw = m.group(1) if m else p.stem

    m8 = re.search(r"\d{8}", raw)

    if m8:
        return m8.group(0)

    return raw


def safe_name(s: str, max_len: int = 90) -> str:
    s = re.sub(r"[\\/:*?\"<>|\s]+", "_", s.strip())
    s = re.sub(r"_+", "_", s).strip("_")

    if not s:
        s = "Unnamed part"

    return s[:max_len]


def unique_file_stem(name: str, used: Dict[str, int]) -> str:
    base = safe_name(name)

    used[base] = used.get(base, 0) + 1

    if used[base] == 1:
        return base

    return f"{base}_{used[base]}"


def get_node_type(node: JSONDict) -> str:
    """
    Supports two formats:

    1. lite format:
       {
         "type": "Prism"
       }

    2. software Feature format:
       {
         "type": "Feature",
         "geometry": {
           "creator_type": "Prism"
         }
       }
    """

    if not isinstance(node, dict):
        return ""

    if node.get("type") == "Feature":
        geo = node.get("geometry", {})

        if isinstance(geo, dict):
            return str(geo.get("creator_type", ""))

    return str(node.get("type", ""))


def features(node: Optional[JSONDict]) -> List[JSONDict]:
    if not isinstance(node, dict):
        return []

    fs = node.get("features", [])

    if isinstance(fs, list):
        return fs

    return []


def get_parameters(node: JSONDict) -> Dict[str, Any]:
    """
    Get parameters.

    lite format:
      node["parameters"]

    software Feature format:
      node["geometry"]["parameters"]
    """

    if not isinstance(node, dict):
        return {}

    geo = node.get("geometry")

    if isinstance(geo, dict):
        params = geo.get("parameters", {})

        if isinstance(params, dict):
            return params

    params = node.get("parameters", {})

    if isinstance(params, dict):
        return params

    return {}


def get_coordinates(node: JSONDict) -> List[Any]:
    """
    Get coordinates.

    lite format:
      node["coordinates"]

    software Feature format:
      node["geometry"]["coordinates"]
    """

    if not isinstance(node, dict):
        return []

    geo = node.get("geometry")

    if isinstance(geo, dict):
        coords = geo.get("coordinates", [])

        if isinstance(coords, list):
            return coords

    coords = node.get("coordinates", [])

    if isinstance(coords, list):
        return coords

    return []


def set_coordinates(node: JSONDict, coords: List[Any]) -> None:
    """
    Set coordinates, supporting both lite and Feature formats.
    """

    if not isinstance(node, dict):
        return

    geo = node.get("geometry")

    if isinstance(geo, dict):
        geo["coordinates"] = coords
    else:
        node["coordinates"] = coords


def iter_nodes(x: Any):
    if isinstance(x, dict):
        if "type" in x:
            yield x

        for v in x.values():
            yield from iter_nodes(v)

    elif isinstance(x, list):
        for i in x:
            yield from iter_nodes(i)


def first_node(x: Any, target_type: str) -> Optional[JSONDict]:
    for n in iter_nodes(x):
        if get_node_type(n) == target_type:
            return n

    return None


def color_key(c: List[int]) -> tuple:
    return tuple(int(x) for x in c)


def auto_color(i: int) -> List[int]:
    """
    Generate a new RGBA integer color when PART_COLORS runs out.
    """

    hue = (0.61803398875 * (i + 1)) % 1.0
    sat = 0.62
    val = 0.88

    k = int(hue * 6) % 6
    f = hue * 6 - int(hue * 6)

    p = val * (1 - sat)
    q = val * (1 - f * sat)
    t = val * (1 - (1 - f) * sat)

    rgb_float = [
        (val, t, p),
        (q, val, p),
        (p, val, t),
        (p, q, val),
        (t, p, val),
        (val, p, q),
    ][k]

    return [
        int(round(rgb_float[0] * 255)),
        int(round(rgb_float[1] * 255)),
        int(round(rgb_float[2] * 255)),
        0,
    ]


def pick_part_color(i: int, used: set) -> List[int]:
    """
    Choose a color for a split part.
    Guarantees:
    1. Different from the complete-model color
    2. Mutually different colors for parts of the same complete model
    """

    if i < len(PART_COLORS):
        c = PART_COLORS[i]
    else:
        c = auto_color(i)

    while color_key(c) in used or color_key(c) == color_key(FULL_MODEL_COLOR):
        i += 1
        c = auto_color(i)

    used.add(color_key(c))

    return c


def apply_color(tree: JSONDict, color: List[int]) -> JSONDict:
    """
    Apply a color using the JSON color format the software can open.

    Software format example:
      root node:
      "color": [117, 152, 176, 0]

      Feature node:
      "geometry": {
          "color": [117, 152, 176, 0]
      }

    lite format:
      written directly on Group / Prism / Face / Circle nodes:
      "color": [117, 152, 176, 0]

    Notes:
    1. Do not write parameters["color"]
    2. Do not wrap colors in a Color command
    """

    data = deepcopy(tree)
    rgba = [int(x) for x in color]

    def walk(obj: Any):
        if isinstance(obj, dict):
            node_type = get_node_type(obj)

            # FeatureCollection root node
            if obj.get("type") == "FeatureCollection":
                obj["color"] = rgba
                obj["global_color"] = False

            # software Feature format: geometry.color
            geo = obj.get("geometry")

            if isinstance(geo, dict):
                geo["color"] = rgba

            # Update an existing top-level color field as well
            if "color" in obj and isinstance(obj.get("color"), list):
                obj["color"] = rgba

            # lite format: write color directly on Group / Prism / Face / Circle nodes
            # Avoid adding a stray top-level color to ordinary Feature nodes
            if (
                node_type
                and obj.get("type") not in {"Feature", "FeatureCollection"}
                and not isinstance(geo, dict)
            ):
                obj["color"] = rgba

            for v in obj.values():
                walk(v)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)

    return data


def dist(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def circle_info(circle: Optional[JSONDict]) -> Optional[Dict[str, Any]]:
    if not circle:
        return None

    params = get_parameters(circle)
    coords = get_coordinates(circle)

    center = None

    if (
        isinstance(coords, list)
        and len(coords) > 0
        and isinstance(coords[0], list)
        and len(coords[0]) >= 3
    ):
        center = coords[0][:3]

    return {
        "r": float(params.get("r", 0.0) or 0.0),
        "center": center,
    }


def prism_info(prism: JSONDict) -> Dict[str, Any]:
    params = get_parameters(prism)

    h = params.get("height")

    try:
        h = float(h) if h is not None else None
    except Exception:
        h = None

    face = None

    for c in features(prism):
        if get_node_type(c) == "Face":
            face = c
            break

    contours = features(face) if face else []

    cinfos = []

    for c in contours:
        cinfos.append(circle_info(first_node(c, "Circle")))

    radii = [c["r"] for c in cinfos if c]

    return {
        "height": h,
        "face": face,
        "contours": contours,
        "circle_infos": cinfos,
        "circle_count": len(radii),
        "max_r": max(radii) if radii else None,
        "min_r": min(radii) if radii else None,
        "is_circular": bool(radii),
    }


def shape_name(node: JSONDict) -> str:
    t = get_node_type(node)

    if t == "Prism":
        info = prism_info(node)

        if info["circle_count"] == 1:
            return "cylinder"

        if info["circle_count"] >= 2:
            return "annular body"

        return "extruded body"

    name_map = {
        "Fuse": "fused body",
        "Union": "fused body",
        "Cut": "cut result body",
        "Subtract": "cut result body",
        "Common": "intersected body",
        "Intersect": "intersected body",
        "Intersection": "intersected body",
        "Group": "group",
        "Face": "face",
        "Circle": "circle",
    }

    return name_map.get(t, t or "submodel")


def adjust_single_circle_face(face: JSONDict) -> None:
    """
    After splitting a multi-contour Face into a single-circle Face, fix Face.coordinates[0].
    This makes rendering a standalone cylinder more stable.
    """

    c = circle_info(first_node(face, "Circle"))
    coords = get_coordinates(face)

    if not c:
        return

    if c["center"] is None:
        return

    if not (
        isinstance(coords, list)
        and len(coords) >= 3
        and isinstance(coords[2], list)
        and len(coords[2]) >= 3
    ):
        return

    v = coords[2][:3]
    norm = math.sqrt(sum(x * x for x in v))

    if norm < 1e-12:
        return

    v = [x / norm for x in v]

    coords[0] = [
        c["center"][i] - c["r"] * v[i]
        for i in range(3)
    ]

    set_coordinates(face, coords)


def single_contour_prism(prism: JSONDict, idx: int) -> JSONDict:
    """
    Split a multi-contour Prism into a Prism that contains a single contour.
    Example: a ring Face contains an outer circle and an inner circle:
    - idx=0 outer cylinder
    - idx=1 inner-bore reference cylinder
    """

    p = deepcopy(prism)
    info = prism_info(p)

    face = info["face"]
    contours = info["contours"]

    if face and 0 <= idx < len(contours):
        face["features"] = [deepcopy(contours[idx])]
        adjust_single_circle_face(face)

    return p


def wrap_like_root(root: JSONDict, node: JSONDict) -> JSONDict:
    """
    Keep the outer structure as close to the original root node as possible.

    1. If the original root is a FeatureCollection:
       keep the FeatureCollection shell and replace only features.

    2. If the original root is a Group:
       keep the Group shell and replace only features.

    3. Otherwise return node unchanged.
    """

    root_type = get_node_type(root)

    if root.get("type") == "FeatureCollection":
        w = deepcopy(root)
        w["features"] = [deepcopy(node)]
        return w

    if root_type == "Group":
        w = deepcopy(root)
        w["features"] = [deepcopy(node)]
        return w

    return deepcopy(node)


def contour_names(prism: JSONDict) -> List[str]:
    info = prism_info(prism)

    cinfos = info["circle_infos"]
    n = len(info["contours"])

    if n <= 0:
        return []

    if all(c is not None for c in cinfos):
        centers = [
            c["center"]
            for c in cinfos
            if c and c["center"] is not None
        ]

        if len(centers) <= 1:
            concentric = True
        else:
            concentric = all(
                dist(centers[0], x) <= 1e-6
                for x in centers[1:]
            )

        rs = [c["r"] for c in cinfos if c]

        max_r = max(rs)
        min_r = min(rs)

        names = []

        for i, c in enumerate(cinfos):
            r = c["r"] if c else None

            if n == 2 and concentric:
                if abs(r - max_r) < 1e-8:
                    names.append("Outer cylinder")
                else:
                    names.append("Inner bore cut reference cylinder")

            else:
                if abs(r - max_r) < 1e-8:
                    pos = "outermost"
                elif abs(r - min_r) < 1e-8:
                    pos = "innermost"
                else:
                    pos = "middle"

                names.append(f"{pos} circular profile {i + 1} extrusion")

        return names

    return [
        f"Profile {i + 1} extrusion"
        for i in range(n)
    ]


def bool_child_names(op: str, children: List[JSONDict]) -> List[str]:
    infos = [
        prism_info(c) if get_node_type(c) == "Prism" else {}
        for c in children
    ]

    circular = all(
        get_node_type(c) == "Prism" and inf.get("is_circular")
        for c, inf in zip(children, infos)
    )

    if op in {"Fuse", "Union"} and circular:
        radii = [
            inf.get("max_r")
            for inf in infos
            if inf.get("max_r") is not None
        ]

        heights = [
            inf.get("height")
            for inf in infos
            if inf.get("height") is not None
        ]

        names = []

        for child, inf in zip(children, infos):
            r = inf.get("max_r")
            h = inf.get("height")

            pos = ""

            if len(radii) >= 2 and max(radii) != min(radii) and r is not None:
                if abs(r - max(radii)) < 1e-8:
                    pos = "outer"
                elif abs(r - min(radii)) < 1e-8:
                    pos = "inner"
                else:
                    pos = "middle"

            hw = ""

            if len(heights) >= 2 and max(heights) != min(heights) and h is not None:
                if abs(h - max(heights)) < 1e-8:
                    hw = "taller"
                elif abs(h - min(heights)) < 1e-8:
                    hw = "thinner"
                else:
                    hw = "medium height"

            names.append(f"{pos}{hw}{shape_name(child)}" or shape_name(child))

        return names

    if op in {"Cut", "Subtract"}:
        names = []

        if children:
            names.append(f"cut target body_{shape_name(children[0])}")

        for i, c in enumerate(children[1:], start=1):
            names.append(f"cut tool body {i}_{shape_name(c)}")

        return names

    return [
        f"{op} child body {i}_{shape_name(c)}"
        for i, c in enumerate(children, start=1)
    ]


def split_tree(root: JSONDict, node: JSONDict, prefix: str = "") -> List[Part]:
    t = get_node_type(node)
    fs = features(node)

    out: List[Part] = []

    # Case 1: a Prism whose Face contains multiple contours
    # Example: a ring face with an outer circle and an inner circle
    if t == "Prism":
        info = prism_info(node)

        if len(info["contours"]) > 1:
            names = contour_names(node)

            for i in range(len(info["contours"])):
                if i < len(names):
                    nm = names[i]
                else:
                    nm = f"Profile {i + 1} extrusion"

                piece = single_contour_prism(node, i)

                out.append(
                    Part(
                        name=nm,
                        tree=wrap_like_root(root, piece),
                        reason="Prism multi-contour Face split",
                    )
                )

        return out

    # Case 2: boolean operations
    # Fuse / Cut / Common etc.: split out every child body
    if t in BOOL_TYPES and fs:
        names = bool_child_names(t, fs)

        for i, child in enumerate(fs):
            if i < len(names):
                nm = names[i]
            else:
                nm = f"{t} child body {i + 1}"

            if prefix:
                nm = f"{prefix}_{nm}"

            out.append(
                Part(
                    name=nm,
                    tree=wrap_like_root(root, child),
                    reason=f"{t} boolean primitive body",
                )
            )

            out.extend(split_tree(root, child, nm))

        return out

    # Case 3: Group
    if t == "Group" and fs:
        if len(fs) == 1:
            out.extend(split_tree(root, fs[0], prefix))

        else:
            for i, child in enumerate(fs, start=1):
                nm = f"Submodel {i}_{shape_name(child)}"

                if prefix:
                    nm = f"{prefix}_{nm}"

                out.append(
                    Part(
                        name=nm,
                        tree=wrap_like_root(root, child),
                        reason="Group submodel split",
                    )
                )

                out.extend(split_tree(root, child, nm))

        return out

    # FeatureCollection: keep recursing
    if t == "FeatureCollection" and fs:
        for child in fs:
            out.extend(split_tree(root, child, prefix))

        return out

    # Other nodes: keep recursing
    for child in fs:
        out.extend(split_tree(root, child, prefix))

    return out


def dedupe(parts: List[Part]) -> List[Part]:
    """
    Remove fully duplicate split results.
    """

    seen = set()
    res = []

    for p in parts:
        key = json.dumps(p.tree, ensure_ascii=False, sort_keys=True)

        if key not in seen:
            seen.add(key)
            res.append(p)

    return res


# ============================================================
# Ollama-based step merging + colored HTML explanations generated from the final per-step models
# ============================================================


def pretty_json(obj: Any, max_chars: int = MAX_JSON_CHARS) -> str:
    """Keep readable indentation before compression; truncate when too long so the prompt does not overflow."""
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(s) > max_chars:
        s = s[:max_chars] + "\n...(JSON truncated; this is only to help the model understand the main structure)"
    return s


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def normalize_dim_key(key: str) -> str:
    return FIELD_ALIASES.get(key, key)


def extract_raw_number_by_key(raw_json_text: str, field_name: str):
    """Extract numeric strings from the raw JSON text, preserving original forms such as 0.7500."""
    pattern = rf'"{re.escape(field_name)}"\s*:\s*([-+]?\d+(?:\.\d+)?)'
    m = re.search(pattern, raw_json_text)
    if m:
        return m.group(1)
    return None


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


def dims_from_json_data(json_data: JSONDict) -> Dict[str, str]:
    raw = json.dumps(json_data, ensure_ascii=False)
    return extract_key_dims_from_raw_json(raw)


def build_available_dims_hint(dims: Dict[str, str]) -> str:
    if not dims:
        return "No available dimension placeholders."
    parts = []
    for k in dims.keys():
        cn_name = CN_NAME_MAP.get(k, k)
        parts.append(f"{cn_name} use <{k}>")
    return "; ".join(parts)


def replace_placeholders(text: str, dims: Dict[str, str]) -> str:
    def repl(match):
        key = match.group(1)
        return dims.get(key, match.group(0))
    return re.sub(r"<([a-zA-Z_][a-zA-Z0-9_]*)>", repl, text)


def cleanup_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[，,]{2,}", ",", text)
    text = re.sub(r"[。.]{2,}", ".", text)
    return text


def extract_json_object(text: str) -> JSONDict:
    """
    Extract a JSON object from an Ollama text response.
    Supports raw JSON, ```json code blocks, and JSON with a small amount of surrounding text.
    """
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.S | re.I)
    if fence:
        s = fence.group(1).strip()
    if not s.startswith("{"):
        first = s.find("{")
        last = s.rfind("}")
        if first >= 0 and last > first:
            s = s[first:last + 1]
    data = json.loads(s)
    if not isinstance(data, dict):
        raise ValueError("The top-level value returned by the model is not a JSON object")
    return data


def call_ollama_generate(
    prompt: str,
    model: str,
    ollama_url: str,
    *,
    expect_json: bool = False,
    retry: int = MAX_RETRY,
    timeout: int = 240,
) -> Optional[str]:
    """
    Call /api/generate the same way generate_description_ollama.py does.
    When expect_json=True, ask Ollama to return JSON as much as possible.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
        },
    }
    if expect_json:
        payload["format"] = "json"

    for attempt in range(1, retry + 1):
        try:
            resp = requests.post(ollama_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            text = str(data.get("response", "")).strip()
            if text:
                return text
            print(f"[OLLAMA ERROR] attempt={attempt}, empty response, raw={data}")
        except Exception as e:
            print(f"[OLLAMA EXCEPTION] attempt={attempt}, err={e}")
        time.sleep(1)
    return None


def build_candidate_for_prompt(parts: List[Part]) -> List[Dict[str, Any]]:
    """Candidate segments sent to Ollama for step merging; these use the candidate JSON produced by the original rules."""
    used_colors = {color_key(FULL_MODEL_COLOR)}
    candidates: List[Dict[str, Any]] = []
    for i, part in enumerate(parts, start=1):
        color = pick_part_color(i - 1, used_colors)
        candidates.append(
            {
                "candidate_index": i,
                "candidate_name": part.name,
                "split_reason": part.reason,
                "suggested_color_rgba": color,
                "candidate_json": apply_color(part.tree, color),
            }
        )
    return candidates


def build_merge_prompt(model_id: str, source_name: str, candidates: List[Dict[str, Any]]) -> str:
    return f"""
You are a CAD modeling tutorial step organizer. Below are candidate segment JSONs produced by rule-based splitting.
Group these candidate segments into steps that are suitable for teaching.

Strict requirements:
1. Use only the provided candidate_index values. Do not invent indexes that do not exist.
2. Each candidate_index may appear in at most one step.
3. Merge multiple candidate segments into the same step when they belong to the same teaching action; keep them separate when their meanings differ.
4. Do not output Markdown, code fences, or explanatory text. Output only strict JSON.
5. All output must be in English; step_title must be English and must not use Chinese. The JSON format must be:
{{
  "steps": [
    {{
      "step_title": "Step title",
      "candidate_indices": [1]
    }}
  ]
}}

Model ID: {model_id}
Source file: {source_name}
Candidate segment JSON:
{pretty_json(candidates)}
""".strip()


def call_ollama_for_step_plan(
    model_id: str,
    source_name: str,
    parts: List[Part],
    *,
    model: str,
    ollama_url: str,
) -> JSONDict:
    candidates = build_candidate_for_prompt(parts)
    prompt = build_merge_prompt(model_id, source_name, candidates)
    text = call_ollama_generate(prompt, model=model, ollama_url=ollama_url, expect_json=True)
    if not text:
        raise RuntimeError("Ollama step merging returned an empty response")
    return extract_json_object(text)


def fallback_steps(parts: List[Part], reason: str = "") -> JSONDict:
    steps = []
    for i, part in enumerate(parts, start=1):
        title = part.name
        if reason:
            title = f"{part.name}"
        steps.append(
            {
                "step_title": title,
                "candidate_indices": [i],
                "fallback_reason": reason,
            }
        )
    return {"steps": steps}


def normalize_steps(plan: JSONDict, parts: List[Part]) -> List[Dict[str, Any]]:
    raw_steps = plan.get("steps", [])
    if not isinstance(raw_steps, list):
        raise ValueError("The model returned JSON where steps is not a list")

    valid = set(range(1, len(parts) + 1))
    used = set()
    normalized: List[Dict[str, Any]] = []

    for s in raw_steps:
        if not isinstance(s, dict):
            continue
        inds = s.get("candidate_indices", [])
        if not isinstance(inds, list):
            continue
        clean_inds = []
        for x in inds:
            try:
                ix = int(x)
            except Exception:
                continue
            if ix in valid and ix not in used:
                used.add(ix)
                clean_inds.append(ix)
        if not clean_inds:
            continue
        default_title = " + ".join(parts[ix - 1].name for ix in clean_inds)
        title = str(s.get("step_title") or default_title).strip()
        normalized.append(
            {
                "step_title": title,
                "candidate_indices": clean_inds,
            }
        )

    # Missing candidate segments are added back as standalone steps so no split result is dropped.
    for ix in sorted(valid - used):
        p = parts[ix - 1]
        normalized.append(
            {
                "step_title": p.name,
                "candidate_indices": [ix],
            }
        )

    return normalized


def root_shell_with_features(root: JSONDict, payload_nodes: List[JSONDict]) -> JSONDict:
    root_type = get_node_type(root)
    if root.get("type") == "FeatureCollection":
        w = deepcopy(root)
        w["features"] = payload_nodes
        return w
    if root_type == "Group":
        w = deepcopy(root)
        w["features"] = payload_nodes
        return w
    return {"type": "Group", "features": payload_nodes}


def unwrap_payload_nodes(tree: JSONDict) -> List[JSONDict]:
    if not isinstance(tree, dict):
        return []
    t = get_node_type(tree)
    if tree.get("type") == "FeatureCollection" and isinstance(tree.get("features"), list):
        return [deepcopy(x) for x in tree["features"]]
    if t == "Group" and isinstance(tree.get("features"), list):
        return [deepcopy(x) for x in tree["features"]]
    return [deepcopy(tree)]


def merge_parts_as_step(root: JSONDict, parts: List[Part], candidate_indices: List[int]) -> JSONDict:
    """Each step JSON is the merge of the candidate segment JSONs produced by the original rules."""
    payload_nodes: List[JSONDict] = []
    for ix in candidate_indices:
        p = parts[ix - 1]
        payload_nodes.extend(unwrap_payload_nodes(p.tree))
    if len(payload_nodes) == 1:
        return wrap_like_root(root, payload_nodes[0])
    return root_shell_with_features(root, payload_nodes)


def build_step_explanation_prompt(
    *,
    model_id: str,
    source_name: str,
    step_index: int,
    step_title: str,
    step_json: JSONDict,
    color: List[int],
    dims: Dict[str, str],
    candidate_names: List[str],
    split_reasons: List[str],
    is_full_model: bool = False,
) -> str:
    dims_hint = build_available_dims_hint(dims)
    if is_full_model:
        mode_text = "No independent candidate segments were produced by rule splitting, so explain the complete modeling process directly from the full CAD JSON."
    else:
        mode_text = "The current JSON is an already split/merged step model JSON, so explain only this step."

    return f"""
You are a CAD modeling tutorial explanation assistant. Generate an English teaching explanation for the given CAD JSON. The explanation body must be entirely in English and must not use Chinese or any other language.

Strict requirements:
1. Output only the explanation body. Do not include a title, numbering, Markdown, or HTML.
2. Output 2 to 4 sentences that are detailed, clear, and suitable as teaching instructions.
3. You may describe modeling actions such as sketching profiles, extruding, boolean union, cutting, holes, main bodies, reference bodies, and local features.
4. Do not guess material, purpose, or software names.
5. If you need to state dimensions, use placeholders in the format <field_name>, for example <radius>, <height>, <length>.
6. Do not output placeholders that do not exist. Use only fields listed under "Available dimension placeholders".
7. Text color is assigned automatically by the HTML to match the step model color, so do not mention red/green/color/RGBA in the body.

{mode_text}

Model ID: {model_id}
Source file: {source_name}
Step number: {step_index}
Step title: {step_title}
Candidate segment names: {'; '.join(candidate_names) if candidate_names else 'None'}
Split reasons: {'; '.join(split_reasons) if split_reasons else 'None'}
Step model color RGBA: {json.dumps(color, ensure_ascii=False)}
Available dimension placeholders: {dims_hint}

CAD JSON:
{pretty_json(step_json)}
""".strip()


def build_rule_explanation(
    *,
    step_title: str,
    dims: Dict[str, str],
    candidate_names: List[str],
    split_reasons: List[str],
    is_full_model: bool,
    ollama_error: str = "",
) -> str:
    if is_full_model:
        base = "This model has no candidate segments that can be rendered independently by the original rules, so this step explains the complete modeling tree. Describe the overall shape formation using the main profile, extrusion features, boolean relationships, and local structures in the full JSON."
    else:
        names = "; ".join(candidate_names) if candidate_names else step_title
        reasons = "; ".join(split_reasons) if split_reasons else "Rule-based split"
        base = f"This step is formed from the candidate segments {names}. Split basis: {reasons}. This step JSON can be rendered independently to explain the role of this geometry in the complete model and serves as a separate teaching stage."
    if dims:
        dim_text = ", ".join(f"{CN_NAME_MAP.get(k, k)} is {v}" for k, v in dims.items())
        base += f" The main dimensions identifiable in this step include: {dim_text}."
    if ollama_error:
        base += f" This is rule-based fallback text because the model did not produce a valid explanation: {ollama_error}."
    return cleanup_text(base)


def call_ollama_for_step_explanation(
    *,
    model_id: str,
    source_name: str,
    step_index: int,
    step_title: str,
    step_json: JSONDict,
    color: List[int],
    candidate_names: List[str],
    split_reasons: List[str],
    is_full_model: bool,
    model: str,
    ollama_url: str,
    use_ollama: bool,
) -> str:
    dims = dims_from_json_data(step_json)
    if not use_ollama:
        return build_rule_explanation(
            step_title=step_title,
            dims=dims,
            candidate_names=candidate_names,
            split_reasons=split_reasons,
            is_full_model=is_full_model,
            ollama_error="Ollama calls are disabled",
        )

    prompt = build_step_explanation_prompt(
        model_id=model_id,
        source_name=source_name,
        step_index=step_index,
        step_title=step_title,
        step_json=step_json,
        color=color,
        dims=dims,
        candidate_names=candidate_names,
        split_reasons=split_reasons,
        is_full_model=is_full_model,
    )
    text = call_ollama_generate(prompt, model=model, ollama_url=ollama_url, expect_json=False)
    if text:
        text = replace_placeholders(text, dims)
        text = cleanup_text(text)
        # Avoid models occasionally emitting HTML/Markdown headings; the final HTML is generated centrally by write_step_explanation_html.
        text = re.sub(r"^#+\s*", "", text).strip()
        text = re.sub(r"</?[^>]+>", "", text).strip()
        if text:
            return text

    return build_rule_explanation(
        step_title=step_title,
        dims=dims,
        candidate_names=candidate_names,
        split_reasons=split_reasons,
        is_full_model=is_full_model,
        ollama_error="The response was empty or not usable",
    )


def rgba_to_css(color: List[int]) -> str:
    r, g, b, a = [int(x) for x in color]
    # In CAD JSON, A=0 usually does not mean web transparency; render as opaque in HTML, otherwise the text would be invisible.
    alpha = 1.0 if a == 0 else max(0.0, min(1.0, a / 255.0))
    return f"rgba({r}, {g}, {b}, {alpha:.3g})"


def html_escape(x: Any) -> str:
    return html.escape(str(x), quote=True)


def split_paragraphs(text: str) -> List[str]:
    raw = [p.strip() for p in re.split(r"(?:\n+|(?<=[。.]))", text) if p.strip()]
    return raw or [text.strip() or "No additional explanation is available for this step."]


def write_step_explanation_html(
    path: Path,
    *,
    source_name: str,
    model_id: str,
    full_file: str,
    items: List[Dict[str, Any]],
) -> None:
    """Output HTML: only step titles use the step color; all other explanation text is uniformly black."""
    title_text = f"{source_name} Detailed Step Explanation"
    lines: List[str] = [
        "<!doctype html>",
        "<html lang=\"zh-CN\">",
        "<head>",
        "  <meta charset=\"utf-8\">",
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"  <title>{html_escape(title_text)}</title>",
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif; line-height: 1.8; margin: 32px; color: #222; background: #fff; }",
        "    h1 { margin: 0 0 16px; font-size: 28px; }",
        "    .meta { margin: 0 0 24px; padding: 12px 16px; background: #f7f7f7; border-radius: 10px; color: #333; }",
        "    .step { margin: 24px 0; padding: 18px 20px; border: 1px solid #e8e8e8; border-radius: 14px; background: #fff; }",
        "    .step h2 { margin: 0 0 12px; font-size: 22px; }",
        "    .item { margin: 6px 0; color: #222; }",
        "    .explanation { margin-top: 14px; font-size: 16px; color: #222; }",
        "    .explanation p { margin: 0 0 10px; color: #222; }",
        "    code { padding: 2px 6px; border-radius: 6px; background: rgba(0, 0, 0, 0.06); font-family: Consolas, Monaco, 'Courier New', monospace; color: #222; }",
        "    .swatch { display: inline-block; width: 0.9em; height: 0.9em; border-radius: 50%; vertical-align: -0.08em; margin-right: 0.35em; border: 1px solid rgba(0,0,0,0.18); }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{html_escape(title_text)}</h1>",
        "  <div class=\"meta\">",
        f"    <div class=\"item\">Model ID: <code>{html_escape(model_id)}</code></div>",
        f"    <div class=\"item\">Full model JSON: <code>{html_escape(full_file)}</code></div>",
        "    <div class=\"item\">Note: only each step title uses its corresponding color; all other text is displayed in black.</div>",
        "  </div>",
    ]

    for item in items:
        idx = int(item.get("step_index", 1))
        title = str(item.get("step_title") or f"Step {idx}").strip()
        file_name = str(item.get("file_name") or "").strip()
        color = item.get("color_rgba") or FULL_MODEL_COLOR
        css = rgba_to_css(color)
        color_json = json.dumps(color, ensure_ascii=False)
        explanation = str(item.get("explanation") or "").strip()
        candidate_indices = item.get("candidate_indices", [])

        # Do not set color on the whole section to avoid coloring the entire block of text
        lines.append("  <section class=\"step\">")

        # Color only the title
        lines.append(f"    <h2 style=\"color: {css};\">Step {idx}: {html_escape(title)}</h2>")

        # All other content stays black
        if file_name:
            lines.append(f"    <div class=\"item\">Corresponding model JSON: <code>{html_escape(file_name)}</code></div>")

        lines.append(
            f"    <div class=\"item\">"
            f"<span class=\"swatch\" style=\"background: {css};\"></span>"
            f"Model color RGBA: <code>{html_escape(color_json)}</code>"
            f"</div>"
        )

        if candidate_indices:
            lines.append(
                f"    <div class=\"item\">Formed from candidate segments: "
                f"<code>{html_escape(json.dumps(candidate_indices, ensure_ascii=False))}</code> (merged).</div>"
            )

        lines.append("    <div class=\"explanation\">")
        for para in split_paragraphs(explanation):
            lines.append(f"      <p>{html_escape(para)}</p>")
        lines.append("    </div>")
        lines.append("  </section>")

    lines.extend(["</body>", "</html>"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def cleanup_legacy_outputs(out_dir: Path) -> None:
    """Remove leftover .md or Description.txt files from older versions so the directory contains only files needed by this version."""
    for name in ["Detailed_Step_Explanation.md", "Description.txt"]:
        p = out_dir / name
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass


def split_file(
    src: Path,
    output_root: Path,
    *,
    model: str,
    ollama_url: str,
    use_ollama: bool = True,
) -> Dict[str, Any]:
    """
    Single-file workflow:
    1. Generate candidate segments with the original split_tree rules.
    2. When candidate segments exist, first ask the configured teaching model to merge them into steps.
    3. Save per-step JSON according to the merge result; each step JSON is the merge of the candidate segment JSONs.
    4. After the per-step JSON files are saved, call the configured teaching model on each per-step JSON to generate the teaching explanation.
    5. When there is no split result, call the configured teaching model directly on the complete-model JSON for a complete modeling-process explanation.
    6. Output the detailed step-explanation HTML, per-step JSON, and complete-model JSON; do not output md/manifest/Description.txt.
    """
    mid = model_id_from_name(src)
    out_dir = output_root / mid
    if out_dir.exists() and OVERWRITE_MODEL_FOLDER:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cleanup_legacy_outputs(out_dir)

    root = load_json(src)
    parts = dedupe(split_tree(root, root))

    used_names: Dict[str, int] = {}
    used_colors = {color_key(FULL_MODEL_COLOR)}
    html_items: List[Dict[str, Any]] = []
    step_json_outputs: List[Dict[str, Any]] = []

    # Complete-model JSON: always output, keep the original file name.
    full_name = src.name
    full_tree = apply_color(root, FULL_MODEL_COLOR)
    full_path = out_dir / full_name
    save_json(full_path, full_tree)

    if parts:
        try:
            if use_ollama:
                plan = call_ollama_for_step_plan(
                    mid,
                    src.name,
                    parts,
                    model=model,
                    ollama_url=ollama_url,
                )
            else:
                plan = fallback_steps(parts, reason="Ollama step merging is disabled")
        except Exception as e:
            if not ALLOW_RULE_FALLBACK:
                raise
            plan = fallback_steps(parts, reason=repr(e))

        steps = normalize_steps(plan, parts)

        for i, step in enumerate(steps, start=1):
            color = pick_part_color(i - 1, used_colors)
            title = str(step.get("step_title") or f"Step {i}").strip()
            candidate_indices = [int(x) for x in step.get("candidate_indices", [])]
            merged_tree = merge_parts_as_step(root, parts, candidate_indices)
            colored_tree = apply_color(merged_tree, color)

            stem = unique_file_stem(title, used_names)
            fname = f"{i:02d}_{stem}.json"
            fpath = out_dir / fname
            save_json(fpath, colored_tree)

            candidate_names = [parts[ix - 1].name for ix in candidate_indices]
            split_reasons = [parts[ix - 1].reason for ix in candidate_indices]
            explanation = call_ollama_for_step_explanation(
                model_id=mid,
                source_name=src.name,
                step_index=i,
                step_title=title,
                step_json=colored_tree,
                color=color,
                candidate_names=candidate_names,
                split_reasons=split_reasons,
                is_full_model=False,
                model=model,
                ollama_url=ollama_url,
                use_ollama=use_ollama,
            )
            time.sleep(SLEEP_BETWEEN)

            item = {
                "step_index": i,
                "step_title": title,
                "candidate_indices": candidate_indices,
                "file_name": fname,
                "color_rgba": color,
                "explanation": explanation,
            }
            html_items.append(item)
            step_json_outputs.append({k: item[k] for k in ["step_index", "step_title", "candidate_indices", "file_name", "color_rgba"]})

    else:
        # No split result: do not force per-step JSON; generate a complete modeling-process explanation directly from the complete-model JSON.
        explanation = call_ollama_for_step_explanation(
            model_id=mid,
            source_name=src.name,
            step_index=1,
            step_title="Complete modeling process",
            step_json=full_tree,
            color=FULL_MODEL_COLOR,
            candidate_names=[],
            split_reasons=[],
            is_full_model=True,
            model=model,
            ollama_url=ollama_url,
            use_ollama=use_ollama,
        )
        html_items.append(
            {
                "step_index": 1,
                "step_title": "Complete modeling process",
                "candidate_indices": [],
                "file_name": full_name,
                "color_rgba": FULL_MODEL_COLOR,
                "explanation": explanation,
            }
        )

    explanation_path = out_dir / STEP_EXPLANATION_FILE
    write_step_explanation_html(
        explanation_path,
        source_name=src.name,
        model_id=mid,
        full_file=full_name,
        items=html_items,
    )

    return {
        "model_id": mid,
        "source_file": str(src),
        "output_dir": str(out_dir),
        "step_explanation_html": str(explanation_path),
        "full_model_json": str(full_path),
        "step_json_files": step_json_outputs,
        "used_full_model_when_no_split": not bool(parts),
    }


def is_under(p: Path, parent: Path) -> bool:
    try:
        p.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split CAD JSON with the original rules and generate colored HTML step explanations using Ollama.")
    parser.add_argument("--input_dir", default=INPUT_DIR, help="Input directory to search recursively for *_lite.json")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--ollama_url", default=OLLAMA_URL, help="Ollama /api/generate endpoint")
    parser.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--no_ollama", action="store_true", help="Do not call Ollama; use rule-based fallback explanations only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_root.is_dir():
        raise NotADirectoryError(f"INPUT_DIR is not a folder or does not exist: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p
        for p in input_root.rglob("*_lite.json")
        if p.is_file() and not is_under(p, output_root)
    )

    print(f"Found *_lite.json files: {len(files)}")
    print(f"Ollama: {'disabled' if args.no_ollama else args.ollama_url + ' / ' + args.model}")
    print(f"Step explanation output format: HTML ({STEP_EXPLANATION_FILE})")

    failed: List[tuple[str, str]] = []
    for idx, p in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {p}")
        try:
            result = split_file(
                p,
                output_root,
                model=args.model,
                ollama_url=args.ollama_url,
                use_ollama=not args.no_ollama,
            )
            print(f"  Output directory: {result['output_dir']}")
            print(f"  Step JSON files: {len(result['step_json_files'])}")
            print(f"  HTML explanation: {result['step_explanation_html']}")
            print(f"  Complete model: {result['full_model_json']}")
        except Exception as e:
            failed.append((str(p), repr(e)))
            print(f"  failed: {repr(e)}")

    if failed:
        failed_path = output_root / "failed_files.txt"
        with failed_path.open("w", encoding="utf-8") as f:
            for p, err in failed:
                f.write(f"{p}\t{err}\n")
        print(f"Failed count: {len(failed)}, see failed_files.txt")

    print("Done")
    print(f"Input model count: {len(files)}")
    print(f"Output directory: {output_root}")


if __name__ == "__main__":
    main()
