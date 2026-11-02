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
#python split_cad_json_dataset.py


from __future__ import annotations

import csv
import json
import math
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# Configure only the settings below.
# ============================================================

INPUT_DIR = r"data/31data"
OUTPUT_DIR = r"outputs/split_dataset"

# Whether to overwrite an existing eight-digit model folder in the output directory
OVERWRITE_MODEL_FOLDER = True

# Whether to merge duplicate segments that share the same build step into a single step.
# When enabled, arrayed holes, repeated cut cylinders, repeated extrusions, etc. emit only one representative JSON.
MERGE_SAME_BUILD_STEPS = True

# When judging whether build steps are identical, whether to ignore coordinates/position.
# True: segments with the same profile and parameters but different positions are merged.
# False: different positions are kept as different steps.
MERGE_IGNORE_POSITION = True

# When judging whether build steps are identical, whether to ignore concrete dimension values.
# True: steps of the same kind (cylinder extrusion, rectangular extrusion, hole cut, ...) merge into one step even when radius/height differ.
# False: differences in radius, height, thickness, etc. keep steps separate.
MERGE_IGNORE_PARAMETER_VALUES = True

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
    repeat_count: int = 1
    merged_names: List[str] = field(default_factory=list)
    build_step_signature: str = ""


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
    Note: this handles only exact JSON duplicates.
    Looser same-build-step merging is performed by merge_same_build_steps().
    """

    seen = set()
    res = []

    for p in parts:
        key = json.dumps(p.tree, ensure_ascii=False, sort_keys=True)

        if key not in seen:
            seen.add(key)
            res.append(p)

    return res


# Position/display fields that are ignored when judging whether build steps are identical.
# These fields usually only determine placement or display color, not the build step itself.
SIGNATURE_IGNORE_KEYS = {
    "color",
    "global_color",
    "coordinates",
    "coordinate",
    "coords",
    "center",
    "origin",
    "position",
    "location",
    "placement",
    "translation",
    "transform",
    "matrix",
    "id",
    "uuid",
    "guid",
    "name",
    "label",
}


def normalize_part_name_for_merge(name: str) -> str:
    """
    Normalize part names for merging duplicate steps.

    Examples:
    - cut tool body 1_cylinder / cut tool body 2_cylinder -> cut tool body_cylinder
    - Submodel 1_extruded body / Submodel 2_extruded body -> Submodel_extruded body
    """

    s = name or ""
    s = re.sub(r"(切除工具体|工具体|子模型|轮廓|子体|cut target body|cut tool body|child body|submodel|profile)[_\- ]*(\d+)", r"\1", s, flags=re.I)
    s = re.sub(r"(contour|part|body|tool|child)[_\- ]*\d+", r"\1", s, flags=re.I)
    s = re.sub(r"\d+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "Unnamed part"


def normalize_value_for_signature(v: Any) -> Any:
    """
    Normalize parameter values so that 1 vs 1.0 or 0.7500000001 vs 0.75 do not cause false mismatches.
    """

    if isinstance(v, bool) or v is None:
        return v

    if isinstance(v, int):
        return v

    if isinstance(v, float):
        return round(v, 8)

    if isinstance(v, str):
        text = v.strip()
        try:
            return round(float(text), 8)
        except Exception:
            return text

    if isinstance(v, list):
        return [normalize_value_for_signature(x) for x in v]

    if isinstance(v, tuple):
        return [normalize_value_for_signature(x) for x in v]

    if isinstance(v, dict):
        return {
            str(k): normalize_value_for_signature(v[k])
            for k in sorted(v.keys(), key=lambda x: str(x))
            if not (MERGE_IGNORE_POSITION and str(k).lower() in SIGNATURE_IGNORE_KEYS)
        }

    return str(v)


def parameters_signature(node: JSONDict) -> Dict[str, Any]:
    """
    Extract a parameter signature that represents the modeling features.
    Coordinates, colors, and names do not participate; dimension parameters such as radius, height, width, and thickness do.
    """

    params = get_parameters(node)
    sig: Dict[str, Any] = {}

    for k in sorted(params.keys(), key=lambda x: str(x)):
        key = str(k)

        if key.lower() in SIGNATURE_IGNORE_KEYS:
            continue

        if MERGE_IGNORE_PARAMETER_VALUES:
            # Keep only the parameter name; do not compare concrete dimension values.
            # This way "create a cylinder" is not split into many steps just because radius/height differ.
            sig[key] = "<value>"
        else:
            sig[key] = normalize_value_for_signature(params[k])

    return sig


def tree_build_signature(x: Any) -> Any:
    """
    Generate a structural signature of the modeling tree.

    The signature captures:
    1. node modeling types, e.g. Prism, Circle, Cut, Fuse;
    2. modeling parameters, e.g. radius, height, thickness;
    3. the topology of child features.

    The signature ignores by default:
    1. coordinate positions;
    2. colors;
    3. display/management fields such as names and IDs.
    """

    if isinstance(x, dict):
        node_type = get_node_type(x)

        if node_type:
            return {
                "node_type": node_type,
                "parameters": parameters_signature(x),
                "features": [tree_build_signature(c) for c in features(x)],
            }

        return {
            str(k): tree_build_signature(v)
            for k, v in sorted(x.items(), key=lambda item: str(item[0]))
            if not (MERGE_IGNORE_POSITION and str(k).lower() in SIGNATURE_IGNORE_KEYS)
        }

    if isinstance(x, list):
        return [tree_build_signature(v) for v in x]

    return normalize_value_for_signature(x)


def build_part_step_signature(part: Part) -> str:
    """
    Generate the build-step signature of a single segment.
    role prevents different semantic roles (base body, cut tool body, outer cylinder, ...) from being merged incorrectly.
    """

    payload = {
        "role": normalize_part_name_for_merge(part.name),
        "reason": part.reason,
        "tree": tree_build_signature(part.tree),
    }

    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def merge_same_build_steps(parts: List[Part]) -> List[Part]:
    """
    Merge segments with identical build steps into one step, keeping only the first representative JSON.

    Typical scenarios:
    - a part with tens or hundreds of identical circular holes;
    - several cut cylinders with the same radius and height that differ only in position;
    - repeated extrusions produced by arrays.
    """

    if not MERGE_SAME_BUILD_STEPS:
        return parts

    seen: Dict[str, Part] = {}
    merged: List[Part] = []

    for part in parts:
        sig = build_part_step_signature(part)

        if sig not in seen:
            part.build_step_signature = sig
            part.repeat_count = 1
            part.merged_names = [part.name]
            seen[sig] = part
            merged.append(part)
            continue

        first = seen[sig]
        first.repeat_count += 1
        first.merged_names.append(part.name)

    return merged


def split_file(src: Path, output_root: Path) -> List[Dict[str, Any]]:
    """
    Split a single *_lite.json file.
    """

    mid = model_id_from_name(src)
    out_dir = output_root / mid

    if out_dir.exists() and OVERWRITE_MODEL_FOLDER:
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    root = load_json(src)

    raw_parts = split_tree(root, root)
    parts = merge_same_build_steps(dedupe(raw_parts))

    used_names: Dict[str, int] = {}
    used_colors = {color_key(FULL_MODEL_COLOR)}

    rows: List[Dict[str, Any]] = []

    # Output split parts first
    for i, part in enumerate(parts, start=1):
        color = pick_part_color(i - 1, used_colors)

        stem = unique_file_stem(part.name, used_names)
        fname = f"{i:02d}_{stem}.json"
        fpath = out_dir / fname

        save_json(fpath, apply_color(part.tree, color))

        reason = part.reason

        if part.repeat_count > 1:
            reason = f"{reason}; merged {part.repeat_count} identical build steps"

        rows.append(
            {
                "model_id": mid,
                "source_file": str(src),
                "output_file": str(fpath),
                "step_index": i,
                "part_name": part.name,
                "file_name": fname,
                "color_rgba": json.dumps(color, ensure_ascii=False),
                "is_full_model": False,
                "repeat_count": part.repeat_count,
                "merged_part_names": json.dumps(part.merged_names, ensure_ascii=False),
                "build_step_signature": part.build_step_signature,
                "reason": reason,
            }
        )

    # Output the complete model last
    # Keep the file name exactly the same as the original input JSON
    full_i = len(parts) + 1
    full_name = src.name
    full_path = out_dir / full_name

    save_json(full_path, apply_color(root, FULL_MODEL_COLOR))

    rows.append(
        {
            "model_id": mid,
            "source_file": str(src),
            "output_file": str(full_path),
            "step_index": full_i,
            "part_name": "Complete model",
            "file_name": full_name,
            "color_rgba": json.dumps(FULL_MODEL_COLOR, ensure_ascii=False),
            "is_full_model": True,
            "repeat_count": 1,
            "merged_part_names": json.dumps(["Complete model"], ensure_ascii=False),
            "build_step_signature": "",
            "reason": "Original complete modeling tree",
        }
    )

    return rows


def is_under(p: Path, parent: Path) -> bool:
    """
    Return whether p lies under parent, used to avoid re-scanning outputs when OUTPUT_DIR is inside INPUT_DIR.
    """

    try:
        p.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def write_manifest(output_root: Path, rows: List[Dict[str, Any]]) -> None:
    """
    manifest.csv and manifest.json are written only at the top of the output directory.
    """

    fieldnames = [
        "model_id",
        "source_file",
        "output_file",
        "step_index",
        "part_name",
        "file_name",
        "color_rgba",
        "is_full_model",
        "repeat_count",
        "merged_part_names",
        "build_step_signature",
        "reason",
    ]

    csv_path = output_root / "manifest.csv"
    json_path = output_root / "manifest.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    save_json(json_path, rows)


def main() -> None:
    input_root = Path(INPUT_DIR)
    output_root = Path(OUTPUT_DIR)

    if not input_root.is_dir():
        raise NotADirectoryError(f"INPUT_DIR is not a folder or does not exist: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p
        for p in input_root.rglob("*_lite.json")
        if p.is_file() and not is_under(p, output_root)
    )

    print(f"Found *_lite.json files: {len(files)}")

    all_rows: List[Dict[str, Any]] = []
    failed: List[tuple[str, str]] = []

    for idx, p in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {p}")

        try:
            rows = split_file(p, output_root)
            all_rows.extend(rows)

        except Exception as e:
            failed.append((str(p), repr(e)))
            print(f"  failed: {repr(e)}")

    write_manifest(output_root, all_rows)

    if failed:
        failed_path = output_root / "failed_files.txt"

        with failed_path.open("w", encoding="utf-8") as f:
            for p, err in failed:
                f.write(f"{p}\t{err}\n")

    print("Done")
    print(f"Input model count: {len(files)}")
    print(f"Output JSON count: {len(all_rows)}")
    print(f"Output directory: {output_root}")

    if failed:
        print(f"Failed count: {len(failed)}, see failed_files.txt")


if __name__ == "__main__":
    main()