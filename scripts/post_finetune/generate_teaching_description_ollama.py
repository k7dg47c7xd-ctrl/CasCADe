# -*- coding: utf-8 -*-
"""
generate_description_ollama.py

Purpose:
1. Read CAD *_lite.json files;
2. Reuse the step splitting and color assignment logic of split_cad_json_dataset.py;
3. Call a large language model interface to generate a more professional, teaching-style "Step 1/Step 2/..." modeling process description;
4. Output three description files: plain text, colored HTML, and structured JSON.

Recommended outputs:
- Description.txt          plain-text step description in the form: Step 1: ...; Step 2: ...; ...
- Description_colored.html  step description with font colors; colors match the color field in the split JSON
- Description_steps.json    structured steps, colors, and descriptions for later programmatic use

Notes:
- Font colors cannot be reliably written to plain txt, so the truly colored result is written to HTML.
- HTML uses RGB colors and does not treat the fourth RGBA component as opacity, avoiding [R,G,B,0] becoming invisible on the web page.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# =====================
# Default configuration
# =====================
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
import os

ROOT_DIR = Path(os.getenv("TEACHING_DATA_DIR", "data/teaching_dataset"))

MAX_JSON_CHARS = 12000
MAX_PART_JSON_CHARS = 700
MAX_RETRY = 2
SLEEP_BETWEEN = 0.5

# Default color table consistent with split_cad_json_dataset.py.
# If split_cad_json_dataset.py can be imported from the same directory, the colors and split functions in that file take precedence.
FULL_MODEL_COLOR_FALLBACK = [117, 152, 176, 0]
PART_COLORS_FALLBACK = [
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

BOOL_TYPES = {"Fuse", "Cut", "Common", "Intersect", "Intersection", "Union", "Subtract"}

TYPE_CN_NAME = {
    "Group": "Feature group",
    "Prism": "Extrusion feature",
    "Face": "Profile face",
    "Wire": "Closed profile wire",
    "Edge": "Straight edge",
    "Arc": "Arc edge",
    "Circle": "Circular profile",
    "Fuse": "Boolean union",
    "Union": "Boolean union",
    "Cut": "Boolean difference",
    "Subtract": "Boolean difference",
    "Common": "Boolean intersection",
    "Intersect": "Boolean intersection",
    "Intersection": "Boolean intersection",
    "FeatureCollection": "Feature collection",
}

COLOR_CN_NAME = {
    (231, 76, 60): "Red",
    (46, 204, 113): "Green",
    (241, 196, 15): "Yellow",
    (155, 89, 182): "Purple",
    (230, 126, 34): "Orange",
    (26, 188, 156): "Teal",
    (52, 73, 94): "Dark blue gray",
    (255, 107, 154): "Pink",
    (127, 140, 141): "Gray",
    (22, 160, 133): "Dark green",
    (211, 84, 0): "Dark orange",
    (142, 68, 173): "Dark purple",
    (192, 57, 43): "Dark red",
    (39, 174, 96): "Deep green",
    (41, 128, 185): "Blue",
    (243, 156, 18): "Golden orange",
    (44, 62, 80): "Dark gray blue",
    (117, 152, 176): "Complete model blue gray",
}

# Allow the script to import split_cad_json_dataset.py from the same directory.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import split_cad_json_dataset as split_cad  # type: ignore

    SPLIT_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - runtime environment fallback only
    split_cad = None  # type: ignore
    SPLIT_IMPORT_ERROR = exc


@dataclass
class StepInfo:
    step_index: int
    title: str
    part_name: str
    reason: str
    color_rgba: List[int]
    is_full_model: bool
    feature_summary: str
    json_excerpt: str


# =====================
# Basic utility functions
# =====================
def pretty_json(obj: Any, max_chars: int = MAX_JSON_CHARS) -> str:
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(s) > max_chars:
        s = s[:max_chars] + "\n...(truncated)"
    return s


def encode_image_base64(img_path: Path) -> str:
    with img_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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


def model_id_from_name(p: Path) -> str:
    m = re.match(r"(.+?)_lite\.json$", p.name, flags=re.I)
    raw = m.group(1) if m else p.stem
    m8 = re.search(r"\d{8}", raw)
    return m8.group(0) if m8 else raw


def cn_number(n: int) -> str:
    units = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    if n < 10:
        return units[n]
    if n == 10:
        return "十"
    if n < 20:
        return "十" + units[n % 10]
    if n < 100:
        ten, one = divmod(n, 10)
        return units[ten] + "十" + (units[one] if one else "")
    return str(n)


def step_title(i: int) -> str:
    return f"Step {i}"


def clean_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = re.sub(r"^(步骤[一二三四五六七八九十百\d]+\s*[：:]|Step\s*\d+\s*[：:])", "", text).strip()
    text = text.strip("；;。 ")
    text = re.sub(r"[，,]{2,}", ",", text)
    return text


def normalize_rgba(color: Any) -> List[int]:
    if isinstance(color, list) and len(color) >= 3:
        rgba = [int(x) for x in color[:4]]
        if len(rgba) == 3:
            rgba.append(0)
        return rgba
    return [0, 0, 0, 0]


def rgba_to_css_rgb(color: List[int]) -> str:
    rgba = normalize_rgba(color)
    return f"rgb({rgba[0]}, {rgba[1]}, {rgba[2]})"


def color_name(color: List[int]) -> str:
    rgba = normalize_rgba(color)
    rgb = tuple(rgba[:3])
    return COLOR_CN_NAME.get(rgb, f"RGB({rgba[0]}, {rgba[1]}, {rgba[2]})")


def color_key(color: List[int]) -> Tuple[int, int, int, int]:
    rgba = normalize_rgba(color)
    return tuple(rgba[:4])  # type: ignore[return-value]


def fallback_pick_color(i: int, used: set) -> List[int]:
    if i < len(PART_COLORS_FALLBACK):
        c = PART_COLORS_FALLBACK[i]
    else:
        # Golden-ratio hue fallback, consistent with the split script.
        import colorsys

        hue = (0.61803398875 * (i + 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.62, 0.88)
        c = [int(round(r * 255)), int(round(g * 255)), int(round(b * 255)), 0]

    while color_key(c) in used or color_key(c) == color_key(FULL_MODEL_COLOR_FALLBACK):
        i += 1
        c = fallback_pick_color(i, used=set())

    used.add(color_key(c))
    return c


# =====================
# JSON analysis and step splitting
# =====================
def get_node_type(node: Any) -> str:
    if not isinstance(node, dict):
        return ""
    if node.get("type") == "Feature":
        geo = node.get("geometry", {})
        if isinstance(geo, dict):
            return str(geo.get("creator_type", ""))
    return str(node.get("type", ""))


def iter_nodes(x: Any):
    if isinstance(x, dict):
        if "type" in x:
            yield x
        for v in x.values():
            yield from iter_nodes(v)
    elif isinstance(x, list):
        for item in x:
            yield from iter_nodes(item)


def node_type_counts(tree: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for node in iter_nodes(tree):
        t = get_node_type(node)
        if t:
            counts[t] = counts.get(t, 0) + 1
    return counts


def collect_parameter_examples(tree: Any, limit: int = 10) -> List[str]:
    out: List[str] = []
    seen = set()

    for node in iter_nodes(tree):
        params = node.get("parameters", {}) if isinstance(node, dict) else {}
        geo = node.get("geometry", {}) if isinstance(node, dict) else {}
        if isinstance(geo, dict) and isinstance(geo.get("parameters"), dict):
            params = geo.get("parameters", {})

        if not isinstance(params, dict):
            continue

        for k, v in params.items():
            if k in {"r", "height", "alpha1", "alpha2", "use_height", "wire_type", "mode_flag"}:
                key = (k, str(v))
                if key not in seen:
                    seen.add(key)
                    out.append(f"{k}={v}")
                    if len(out) >= limit:
                        return out

    return out


def summarize_tree_for_prompt(tree: Any) -> str:
    counts = node_type_counts(tree)
    if not counts:
        return "No clear CAD feature nodes were recognized."

    ordered_types = [
        "Prism", "Face", "Wire", "Circle", "Arc", "Edge",
        "Fuse", "Cut", "Union", "Subtract", "Common", "Group",
    ]
    type_parts = []
    for t in ordered_types:
        if counts.get(t):
            type_parts.append(f"{TYPE_CN_NAME.get(t, t)}×{counts[t]}")

    rest = [f"{TYPE_CN_NAME.get(t, t)}×{v}" for t, v in counts.items() if t not in ordered_types]
    type_text = ", ".join(type_parts + rest[:6])

    param_examples = collect_parameter_examples(tree)
    param_text = "; key parameter examples: " + ", ".join(param_examples) if param_examples else ""

    operations = []
    if counts.get("Prism"):
        operations.append("contains extrusion features")
    if counts.get("Face") or counts.get("Wire"):
        operations.append("contains profile definitions")
    if counts.get("Circle"):
        operations.append("contains circular profiles")
    if counts.get("Arc"):
        operations.append("contains arc edges")
    if counts.get("Fuse") or counts.get("Union"):
        operations.append("contains boolean union")
    if counts.get("Cut") or counts.get("Subtract"):
        operations.append("contains boolean difference cuts")

    op_text = "；" + "，".join(operations) if operations else ""
    return f"Node statistics: {type_text}{op_text}{param_text}."


def fallback_split_parts(root: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
    """
    Simplified splitting used when split_cad_json_dataset.py cannot be imported:
    Prefer splitting out boolean child bodies; otherwise split the child features of the top-level Group.
    """
    parts: List[Tuple[str, Dict[str, Any], str]] = []

    def features(node: Any) -> List[Dict[str, Any]]:
        fs = node.get("features", []) if isinstance(node, dict) else []
        return fs if isinstance(fs, list) else []

    def shape_name(node: Dict[str, Any]) -> str:
        t = get_node_type(node)
        return TYPE_CN_NAME.get(t, t or "submodel")

    def walk(node: Dict[str, Any], prefix: str = "") -> None:
        t = get_node_type(node)
        fs = features(node)
        if t in BOOL_TYPES and fs:
            for i, child in enumerate(fs, start=1):
                nm = f"{prefix}_{t} child body {i}_{shape_name(child)}" if prefix else f"{t} child body {i}_{shape_name(child)}"
                parts.append((nm, child, f"{t} boolean primitive body"))
                walk(child, nm)
        elif t == "Group" and fs:
            for i, child in enumerate(fs, start=1):
                nm = f"{prefix}_Submodel {i}_{shape_name(child)}" if prefix else f"Submodel {i}_{shape_name(child)}"
                parts.append((nm, child, "Group submodel split"))
                walk(child, nm)

    walk(root)

    # Deduplicate so the same subtree does not appear more than once.
    seen = set()
    deduped = []
    for name, tree, reason in parts:
        key = json.dumps(tree, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append((name, tree, reason))

    return deduped


def build_step_plan(json_data: Dict[str, Any], include_full_model_step: bool = True) -> List[StepInfo]:
    steps: List[StepInfo] = []

    if split_cad is not None:
        # Reuse the step splitting, naming, and color assignment logic of split_cad_json_dataset.py.
        parts = split_cad.dedupe(split_cad.split_tree(json_data, json_data))
        full_color = normalize_rgba(getattr(split_cad, "FULL_MODEL_COLOR", FULL_MODEL_COLOR_FALLBACK))
        used_colors = {split_cad.color_key(full_color)}

        for i, part in enumerate(parts, start=1):
            color = normalize_rgba(split_cad.pick_part_color(i - 1, used_colors))
            tree = part.tree
            steps.append(
                StepInfo(
                    step_index=i,
                    title=step_title(i),
                    part_name=str(part.name),
                    reason=str(part.reason),
                    color_rgba=color,
                    is_full_model=False,
                    feature_summary=summarize_tree_for_prompt(tree),
                    json_excerpt=pretty_json(tree, MAX_PART_JSON_CHARS),
                )
            )
    else:
        full_color = FULL_MODEL_COLOR_FALLBACK
        used_colors = {color_key(full_color)}
        parts = fallback_split_parts(json_data)

        for i, (part_name, tree, reason) in enumerate(parts, start=1):
            color = fallback_pick_color(i - 1, used_colors)
            steps.append(
                StepInfo(
                    step_index=i,
                    title=step_title(i),
                    part_name=part_name,
                    reason=reason,
                    color_rgba=color,
                    is_full_model=False,
                    feature_summary=summarize_tree_for_prompt(tree),
                    json_excerpt=pretty_json(tree, MAX_PART_JSON_CHARS),
                )
            )

    if not steps:
        steps.append(
            StepInfo(
                step_index=1,
                title=step_title(1),
                part_name="Base solid",
                reason="No independent child bodies were split out; generate a single-step description from the complete modeling tree",
                color_rgba=PART_COLORS_FALLBACK[0],
                is_full_model=False,
                feature_summary=summarize_tree_for_prompt(json_data),
                json_excerpt=pretty_json(json_data, MAX_PART_JSON_CHARS),
            )
        )

    if include_full_model_step:
        i = len(steps) + 1
        steps.append(
            StepInfo(
                step_index=i,
                title=step_title(i),
                part_name="Complete model",
                reason="Original complete modeling tree",
                color_rgba=full_color,
                is_full_model=True,
                feature_summary=summarize_tree_for_prompt(json_data),
                json_excerpt=pretty_json(json_data, MAX_PART_JSON_CHARS),
            )
        )

    # Recompute titles so numbering stays consecutive after adding the complete-model step.
    for i, s in enumerate(steps, start=1):
        s.step_index = i
        s.title = step_title(i)

    return steps


# =====================
# Prompt and Ollama calls
# =====================
def build_professional_step_prompt(json_data: Dict[str, Any], steps: List[StepInfo]) -> str:
    step_plan = []
    include_part_excerpt = len(steps) <= 25
    for s in steps:
        item = {
            "step_index": s.step_index,
            "step_title": s.title,
            "part_name": s.part_name,
            "reason": s.reason,
            "color_rgba": s.color_rgba,
            "color_name": color_name(s.color_rgba),
            "is_full_model": s.is_full_model,
            "feature_summary": s.feature_summary,
        }
        # When there are too many steps, do not put every part JSON excerpt into the prompt to avoid exceeding the Ollama context length.
        if include_part_excerpt:
            item["part_json_excerpt"] = s.json_excerpt
        step_plan.append(item)

    full_summary = summarize_tree_for_prompt(json_data)
    full_json_text = pretty_json(json_data, MAX_JSON_CHARS)

    return f"""
You are a CAD parametric modeling course instructor. Based on the CAD modeling JSON, the split step plan, and multi-view screenshots of the model, write a professional, clear, teaching-style modeling process description in English.

Task goals:
- Produce a modeling process in the form "Step 1: ...; Step 2: ...; ...".
- Every step must correspond to one part_name in the step plan below. Do not add, remove, or merge steps.
- The text color of each step is rendered automatically by the program from color_rgba; you only write the professional description of that step. Do not output HTML, Markdown, CSS, or color tags.

Technical terminology reference:
- datum plane, cross-section sketch, closed profile, outer profile, inner profile, circular profile, arc edge, straight edge, profile constraints, profile normal, extrusion feature, solid formation, boolean union, boolean difference, cut tool body, cut target body, coaxial relation, through hole, stepped structure, feature tree, model definition, overall review.

Strict requirements:
1. Output only a JSON array. Do not output explanatory text or code fences.
2. The JSON array length must equal {len(steps)}.
3. Every object must have the format: {{"step_index": 1, "description": "..."}}.
4. description must not contain numbering such as "Step 1:"; the program adds it automatically.
5. description should read like classroom teaching, use professional terminology, and state the modeling action this step builds/defines/generates/executes/completes.
6. Do not write a plain appearance description; describe the modeling process, e.g. "Establish the closed profile and extrude it along the normal into a solid" or "Use the tool body for a boolean difference cut".
7. Do not invent structures, materials, purposes, assembly relations, or color meanings that are absent from the JSON.
8. If the step is the complete model, only describe feature-tree integration, boolean relationship review, and final model definition; do not invent new operations.
9. Keep the language concise; each description should be about 25 to 55 words.
10. The final rendering should match: Step 1: ...; Step 2: ...; Step 3: ....

Complete model structure summary:
{full_summary}

Step plan and colors:
{json.dumps(step_plan, ensure_ascii=False, indent=2)}

Complete CAD JSON reference:
{full_json_text}
""".strip()


def call_ollama_step_desc(
    json_data: Dict[str, Any],
    steps: List[StepInfo],
    images: List[Path],
    model: str = "",
    ollama_url: str = OLLAMA_URL,
    retry: int = MAX_RETRY,
    timeout: int = 180,
) -> Optional[str]:
    prompt = build_professional_step_prompt(json_data, steps)

    image_b64_list = []
    for img in images:
        try:
            image_b64_list.append(encode_image_base64(img))
        except Exception as e:
            print(f"[IMAGE ERROR] {img.name}: {e}")

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
        },
    }

    if image_b64_list:
        payload["images"] = image_b64_list

    for attempt in range(1, retry + 1):
        try:
            resp = requests.post(ollama_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            text = data.get("response", "").strip()
            if text:
                return text

            print(f"[OLLAMA ERROR] attempt={attempt}, empty response, raw={data}")

        except Exception as e:
            print(f"[OLLAMA EXCEPTION] attempt={attempt}, err={e}")

        time.sleep(1)

    return None


def extract_json_array(text: str) -> Optional[List[Any]]:
    if not text:
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Prefer direct parsing.
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # Also handle models that emit a small amount of text before/after the JSON.
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start >= 0 and end > start:
        fragment = cleaned[start:end + 1]
        try:
            data = json.loads(fragment)
            if isinstance(data, list):
                return data
        except Exception:
            return None

    return None


def parse_model_steps(text: Optional[str], steps: List[StepInfo]) -> Optional[List[str]]:
    if not text:
        return None

    data = extract_json_array(text)
    if not data or len(data) != len(steps):
        return None

    by_index: Dict[int, str] = {}
    sequential: List[str] = []

    for i, item in enumerate(data, start=1):
        if isinstance(item, dict):
            idx = item.get("step_index", i)
            desc = item.get("description", "")
            try:
                idx = int(idx)
            except Exception:
                idx = i
            desc = clean_sentence(desc)
            if desc:
                by_index[idx] = desc
        elif isinstance(item, str):
            desc = clean_sentence(item)
            if desc:
                sequential.append(desc)

    if by_index:
        result = []
        for s in steps:
            desc = by_index.get(s.step_index, "")
            if not desc:
                return None
            result.append(desc)
        return result

    if len(sequential) == len(steps):
        return sequential

    return None


# =====================
# Fallback step generation and rendering
# =====================
def build_rule_based_descriptions(steps: List[StepInfo]) -> List[str]:
    descs: List[str] = []

    for s in steps:
        name = s.part_name
        reason = s.reason
        summary = s.feature_summary

        if s.is_full_model:
            desc = "Integrate the full feature tree, verify the extrusion, union, and difference relationships, and finalize the complete model definition"
        elif "cut target body" in name:
            desc = f"Build the base solid of {name}, preserving its base profile and extrusion relationships"
        elif "cut tool body" in name or "Cut" in reason or "Subtract" in reason:
            desc = f"Define {name} as the reference tool body for the boolean difference used to establish the cut region"
        elif "Outer" in name and "cylinder" in name:
            desc = f"Build {name} by extruding the outer profile along its normal to form the main solid"
        elif "Inner" in name:
            desc = f"Build {name}, defining the inner profile or bore reference that provides the geometric basis for the subsequent cut"
        elif "cylinder" in name or "Cylinder" in name:
            desc = f"Build {name} as a cylinder-like solid feature using a circular profile and the extrusion height"
        elif "Profile" in name or "extruded body" in name or "extrusion" in name:
            desc = f"Build {name} by first constraining a closed profile and then extruding it along the profile normal"
        elif "group" in name or "Submodel" in name or "submodel" in name:
            desc = f"Define {name}, organizing the local features and preserving their hierarchy in the overall feature tree"
        else:
            desc = f"Build {name} according to the geometric nodes of this part to complete the local model definition"

        # Let the fallback sentences also reflect the key modeling actions that were recognized; do not append redundant operations to complete-model steps.
        if not s.is_full_model:
            if "boolean union" in summary and "boolean" not in desc:
                desc += ", preserving the union relationship with the adjacent solids"
            if "boolean difference" in summary and "difference" not in desc:
                desc += ", preserving the semantics of the difference cut"

        descs.append(clean_sentence(desc))

    return descs


def render_plain_text(steps: List[StepInfo], descriptions: List[str]) -> str:
    parts = []
    for s, desc in zip(steps, descriptions):
        parts.append(f"{s.title}: {clean_sentence(desc)}")
    return "; ".join(parts) + "."


def render_colored_html(steps: List[StepInfo], descriptions: List[str], source_name: str) -> str:
    rows = []
    for s, desc in zip(steps, descriptions):
        css = rgba_to_css_rgb(s.color_rgba)
        rgba_text = html.escape(json.dumps(s.color_rgba, ensure_ascii=False))
        rows.append(
            "    <p>"
            f"<span class=\"step-text\" data-rgba=\"{rgba_text}\" "
            f"style=\"color: {css};\">"
            f"<strong>{html.escape(s.title)}: </strong>{html.escape(clean_sentence(desc))}"
            "</span>"
            "</p>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(source_name)} - CAD Modeling Step Explanation</title>
  <style>
    body {{
      font-family: Arial, "Segoe UI", sans-serif;
      line-height: 1.8;
      padding: 24px;
      background: #ffffff;
    }}
    .card {{
      max-width: 980px;
      margin: 0 auto;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 20px 24px;
    }}
    h1 {{
      font-size: 20px;
      margin: 0 0 16px;
      color: #111827;
    }}
    p {{
      margin: 8px 0;
      font-size: 16px;
    }}
    .hint {{
      margin-top: 18px;
      color: #6b7280;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>CAD Modeling Step Explanation: {html.escape(source_name)}</h1>
{chr(10).join(rows)}
    <div class="hint">Text colors come from the color_rgba values produced by split_cad_json_dataset.py; the HTML rendering uses RGB so that the fourth RGBA component does not make text transparent.</div>
  </div>
</body>
</html>
""".strip() + "\n"


def build_steps_json(steps: List[StepInfo], descriptions: List[str]) -> List[Dict[str, Any]]:
    out = []
    for s, desc in zip(steps, descriptions):
        out.append(
            {
                "step_index": s.step_index,
                "step_title": s.title,
                "part_name": s.part_name,
                "description": clean_sentence(desc),
                "color_rgba": s.color_rgba,
                "css_color": rgba_to_css_rgb(s.color_rgba),
                "color_name": color_name(s.color_rgba),
                "is_full_model": s.is_full_model,
                "reason": s.reason,
                "feature_summary": s.feature_summary,
            }
        )
    return out


# =====================
# Batch processing flow
# =====================
def discover_json_files(root_dir: Path) -> List[Path]:
    if root_dir.is_file() and root_dir.name.lower().endswith("_lite.json"):
        return [root_dir]
    return sorted(p for p in root_dir.rglob("*_lite.json") if p.is_file())


def output_dir_for_json(src: Path, root_dir: Path, output_root: Optional[Path]) -> Path:
    mid = model_id_from_name(src)
    if output_root is not None:
        return output_root / mid

    # When a directory contains multiple *_lite.json files (for example json_file/ after unzipping), create a folder per model so Description.txt files do not overwrite each other.
    sibling_jsons = list(src.parent.glob("*_lite.json"))
    if len(sibling_jsons) > 1:
        return src.parent / mid

    # Common original-dataset structure: one folder per model; write the description directly into that model folder.
    return src.parent


def process_json_file(
    json_path: Path,
    root_dir: Path,
    output_root: Optional[Path],
    use_ollama: bool,
    model: str,
    ollama_url: str,
    include_full_model_step: bool,
) -> None:
    print(f"processing: {json_path}")

    try:
        raw_json_text = json_path.read_text(encoding="utf-8")
        json_data = json.loads(raw_json_text)
    except Exception as e:
        print(f"error: {json_path}, failed to read JSON: {e}")
        return

    out_dir = output_dir_for_json(json_path, root_dir, output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = build_step_plan(json_data, include_full_model_step=include_full_model_step)
    images = select_images(json_path.parent, max_images=2)
    if not images:
        print(f"warn: {json_path.name}, no png images found; JSON only will be used")

    descriptions: Optional[List[str]] = None

    if use_ollama:
        model_text = call_ollama_step_desc(
            json_data=json_data,
            steps=steps,
            images=images,
            model=model,
            ollama_url=ollama_url,
        )
        descriptions = parse_model_steps(model_text, steps)
        if descriptions is None:
            print(f"warn: {json_path.name}, Ollama output could not be parsed; falling back to rule templates")

    if descriptions is None:
        descriptions = build_rule_based_descriptions(steps)

    plain_text = render_plain_text(steps, descriptions)
    html_text = render_colored_html(steps, descriptions, json_path.name)
    steps_json = build_steps_json(steps, descriptions)

    (out_dir / "Description.txt").write_text(plain_text, encoding="utf-8")
    (out_dir / "Description_colored.html").write_text(html_text, encoding="utf-8")
    (out_dir / "Description_steps.json").write_text(
        json.dumps(steps_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preview = plain_text if len(plain_text) <= 220 else plain_text[:220] + "..."
    print(f"done: {json_path.name} -> {out_dir}，steps={len(steps)}")
    print(f"preview: {preview}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        default=str(ROOT_DIR),
        help="Dataset root directory; supports either one folder per model or a single directory that directly contains multiple *_lite.json",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional. Unified output directory; when not specified, write into the model folder or automatically create a model-id subfolder",
    )
    parser.add_argument(
        "--use_ollama",
        action="store_true",
        help="Whether to use a local Ollama model; when disabled, rule templates generate usable step descriptions",
    )
    parser.add_argument(
        "--ollama_url",
        default=OLLAMA_URL,
        help="Ollama generate endpoint",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("TEACHING_MODEL", ""),
        help="Model ID (set via the TEACHING_MODEL environment variable or --model); takes effect only with --use_ollama",
    )
    parser.add_argument(
        "--no_full_model_step",
        action="store_true",
        help="Do not output the final complete-model review step",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional. Process only the first N JSON files for testing; 0 means process all",
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_root = Path(args.output_dir) if args.output_dir else None

    if not root_dir.exists():
        raise FileNotFoundError(f"ROOT_DIR does not exist: {root_dir}")

    temp_ctx: Optional[tempfile.TemporaryDirectory[str]] = None
    scan_root = root_dir

    # A zip file can be passed directly; the default output directory is <zip_stem>_descriptions next to the zip.
    if root_dir.is_file() and root_dir.suffix.lower() == ".zip":
        if output_root is None:
            output_root = root_dir.with_name(root_dir.stem + "_descriptions")
        temp_ctx = tempfile.TemporaryDirectory()
        scan_root = Path(temp_ctx.name)
        with zipfile.ZipFile(root_dir, "r") as zf:
            zf.extractall(scan_root)
        print(f"info: extracted zip to a temporary directory: {scan_root}")

    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    if split_cad is None:
        print(f"warn: could not import split_cad_json_dataset.py; the simplified splitting logic will be used. Reason: {SPLIT_IMPORT_ERROR}")
    else:
        print("info: imported split_cad_json_dataset.py; step splitting and color assignment will match that file")

    files = discover_json_files(scan_root)
    if args.limit and args.limit > 0:
        files = files[:args.limit]
    print("total:", len(files))

    try:
        for json_path in files:
            try:
                process_json_file(
                    json_path=json_path,
                    root_dir=scan_root,
                    output_root=output_root,
                    use_ollama=args.use_ollama,
                    model=args.model,
                    ollama_url=args.ollama_url,
                    include_full_model_step=not args.no_full_model_step,
                )
                time.sleep(SLEEP_BETWEEN)

            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                print(f"fatal error: {json_path}, {e}")
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    main()
