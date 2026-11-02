import json
import re
from typing import Any, Dict, Iterable, Optional, Set

# ------------------------------ Cadmium-lite Prompt Design ------------------------------
# This prompt template is fitted to the uploaded Cadmium-lite dataset JSON structure.
# The dataset uses a recursive operation tree, not a "parts/part_1/sketch" schema.
#
# Generation flow:
#   Group(root) -> optional Fuse/Cut boolean tree -> Prism -> Face -> Wire/Group -> Edge/Arc/Circle
#
# All generated JSON nodes must use the same five fields:
#   type, features, group, coordinates, parameters


# ------------------------------ Dataset-derived node templates ------------------------------

ROOT_GROUP_TEMPLATE: Dict[str, Any] = {
    "type": "Group",
    "features": ["<Prism | Fuse | Cut nodes>"],
    "group": "operator",
    "coordinates": [],
    "parameters": {
        "use_result_filter": False,
        "exclude": False,
        "return_index_list": []
    }
}

BOOLEAN_FUSE_TEMPLATE: Dict[str, Any] = {
    "type": "Fuse",
    "features": ["<solid_or_boolean_node_1>", "<solid_or_boolean_node_2>"],
    "group": "operator",
    "coordinates": [],
    "parameters": {}
}

BOOLEAN_CUT_TEMPLATE: Dict[str, Any] = {
    "type": "Cut",
    "features": ["<base_solid_or_boolean_node>", "<cutter_solid_or_boolean_node>"],
    "group": "operator",
    "coordinates": [],
    "parameters": {}
}

PRISM_TEMPLATE: Dict[str, Any] = {
    "type": "Prism",
    "features": ["<Face node>"],
    "group": "operator",
    "coordinates": [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ],
    "parameters": {
        "height": 0.0,
        "use_height": True,
        "auto_normal_vector": True,
        "combin": False
    }
}

FACE_TEMPLATE: Dict[str, Any] = {
    "type": "Face",
    "features": ["<Wire node and/or circle Group nodes>"],
    "group": "operator",
    "coordinates": [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
    ],
    "parameters": {
        "combin": False,
        "mode_flag": 3,
        "uv_list": []
    }
}

WIRE_TEMPLATE: Dict[str, Any] = {
    "type": "Wire",
    "features": ["<ordered Edge and/or Arc nodes>"],
    "group": "operator",
    "coordinates": [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
    ],
    "parameters": {
        "vertex_input": False,
        "edge_sort": False,
        "linear_to_lerance": 0.000001
    }
}

EDGE_TEMPLATE: Dict[str, Any] = {
    "type": "Edge",
    "features": [],
    "group": "creator",
    "coordinates": [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ],
    "parameters": {
        "pline": False,
        "combin": False
    }
}

ARC_TEMPLATE: Dict[str, Any] = {
    "type": "Arc",
    "features": [],
    "group": "creator",
    "coordinates": [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [1.0, 0.0, 0.0]
    ],
    "parameters": {
        "r": 0.0,
        "alpha1": 0.0,
        "alpha2": 0.0,
        "circle_mode": "0p",
        "arc_mode": "3p",
        "sense": True
    }
}

CIRCLE_GROUP_TEMPLATE: Dict[str, Any] = {
    "type": "Group",
    "features": ["<one Circle node>"],
    "group": "operator",
    "coordinates": [],
    "parameters": {
        "use_result_filter": False,
        "exclude": False,
        "return_index_list": []
    }
}

CIRCLE_TEMPLATE: Dict[str, Any] = {
    "type": "Circle",
    "features": [],
    "group": "creator",
    "coordinates": [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0]
    ],
    "parameters": {
        "r": 0.0,
        "wire_type": True
    }
}

NODE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "RootGroup": ROOT_GROUP_TEMPLATE,
    "Fuse": BOOLEAN_FUSE_TEMPLATE,
    "Cut": BOOLEAN_CUT_TEMPLATE,
    "Prism": PRISM_TEMPLATE,
    "Face": FACE_TEMPLATE,
    "Wire": WIRE_TEMPLATE,
    "Edge": EDGE_TEMPLATE,
    "Arc": ARC_TEMPLATE,
    "CircleGroup": CIRCLE_GROUP_TEMPLATE,
    "Circle": CIRCLE_TEMPLATE,
}


def _compact_template(obj: Any) -> str:
    """Return a compact JSON-like template string for prompt injection."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


CADMIUM_SCHEMA_REFERENCE = "\n".join(
    f"{name}: {_compact_template(template)}"
    for name, template in NODE_TEMPLATES.items()
)


# ------------------------------ Software command semantic reference ------------------------------
# These commands come from the CAD software operation vocabulary. They are NOT the output schema.
# They are embedded into the prompt as modeling intent guidance so that Qwen chooses better
# Cadmium-lite nodes while the backend/converter can still consume the existing recursive tree.

SOFTWARE_COMMAND_SEMANTIC_REFERENCE = """
CAD SOFTWARE COMMAND SEMANTICS, FOR MODELING INTENT ONLY:
- New_Cut means boolean difference. In Cadmium-lite output, express it as a Cut node. The first child is the base solid; the remaining child nodes are cutter solids.
- New_Compound means boolean union/compound. In Cadmium-lite output, express it as a Fuse node or a root Group containing additive solids.
- New_Vertex means a point. In Cadmium-lite output, do not create standalone Vertex nodes because Vertex is not an allowed type. Use 3D points inside Edge/Arc/Circle/Face/Prism coordinates.
- New_Edge means a straight line segment. In Cadmium-lite output, express it as an Edge node with exactly two 3D coordinate points.
- New_Circle means a circle. In Cadmium-lite output, express it as a Circle node wrapped by an operator Group when used inside a Face.
- New_Box means a rectangular box/cuboid. In Cadmium-lite output, express it as a Prism whose Face is a closed rectangular Wire with four Edge nodes. x_size maps to X length, y_size maps to Y width, z_size maps to Prism.parameters.height.
- New_Cylinder means a cylinder. In Cadmium-lite output, express it as a Prism whose Face contains a circle Group. radius maps to Circle.parameters.r, height maps to Prism.parameters.height.
- New_Sphere and New_Cone exist in the CAD software command vocabulary, but Sphere and Cone are NOT allowed Cadmium-lite node types in this project. Do not output type="Sphere", type="Cone", name="New_Sphere", or name="New_Cone" unless the output schema is explicitly changed later.
- All software command fields such as name, rebuild_features_immediately, use_vertexs_coordinates, context, color, feature_properties, model_properties, and model_global_param are reference-only. NEVER output these fields in Cadmium-lite JSON.
""".strip()


# ------------------------------ Fixed base prompt ------------------------------

BASE_SYSTEM_MESSAGE_TEMPLATE = """Generate Cadmium-lite CAD model JSON that matches the uploaded dataset format.

The dataset format is a recursive operation tree. Use ONLY these node templates:
{CADMIUM_SCHEMA_REFERENCE}

The following CAD software commands are provided as semantic guidance only.
They help you choose the correct Cadmium-lite nodes, but they are NOT the output format:
{SOFTWARE_COMMAND_SEMANTIC_REFERENCE}

STRICT OUTPUT RULES:
- OUTPUT ONLY RAW JSON. Do not include markdown fences, comments, explanation, titles, or summaries.
- The top-level object MUST be a root Group node.
- Every node MUST contain exactly these five fields in this order: type, features, group, coordinates, parameters.
- Allowed type values: Group, Fuse, Cut, Prism, Face, Wire, Edge, Arc, Circle.
- Allowed group values: "operator" for Group/Fuse/Cut/Prism/Face/Wire; "creator" for Edge/Arc/Circle.
- Do NOT output the old schema fields: parts, part_1, coordinate_system, description, extrusion, sketch, loop_1, line_1, circle_1, arc_1.
- Do NOT output CAD software command fields: name, rebuild_features_immediately, use_vertexs_coordinates, context, color, feature_properties, model_properties, model_global_param.
- Do NOT output software command names such as New_Box, New_Cylinder, New_Circle, New_Edge, New_Cut, New_Compound, New_Sphere, or New_Cone.
- Do NOT output null for missing geometry. Use an empty list only where the template uses [].
- The angle-bracket strings inside template features, such as "<Face node>", are placeholders only. NEVER output placeholder strings; actual features must contain child node objects.
- All numeric coordinates and dimensions must be JSON numbers. Prefer floats for measured values.
- Keep fixed parameter keys exactly as shown in the templates. Do not add, remove, rename, or translate fields.
- Use 3D coordinate vectors [x,y,z] even for 2D sketch geometry. For planar sketches, z is usually 0.0 unless the sketch plane is offset.
"""


# ------------------------------ Optional instruction blocks ------------------------------
# These blocks are added only when needed. This keeps the fixed schema stable while allowing
# the prompt to become stricter for arcs, circles, booleans, and complex multi-prism models.

INSTRUCTION_BLOCKS: Dict[str, str] = {
    "prism_face": """
PRISM/FACE RULES:
- A Prism creates one extruded solid from exactly one Face child.
- Prism.coordinates is [base_point, extrusion_direction]. Use [0.0,0.0,0.0] and [0.0,0.0,1.0] for standard vertical extrusion.
- Prism.parameters.height is the signed extrusion distance. Positive and negative heights are both allowed.
- A Face contains one or more Wire nodes and/or circle Group nodes.
- Face.coordinates is [plane_origin, plane_normal, x_axis_direction]. Use [[0.0,0.0,0.0],[0.0,0.0,1.0],[1.0,0.0,0.0]] for the default XY plane.
""",
    "wire_edge": """
WIRE/EDGE RULES:
- A Wire is an ordered closed or open chain of Edge and Arc nodes.
- Use Edge for straight segments. Edge.coordinates is [start_point, end_point].
- For a closed profile, the final primitive endpoint must return to the first primitive start point.
- Preserve geometric order around the profile; do not randomly reorder edges/arcs.
""",
    "arc": """
ARC RULES:
- Use Arc only for curved/rounded boundaries.
- Arc.coordinates is exactly [start_point, mid_point, end_point] in 3D coordinates.
- Arc.parameters.r is the radius; alpha1 is usually 0.0; alpha2 is the included angle in radians when known.
- Keep circle_mode="0p", arc_mode="3p", and sense=true.
""",
    "circle": """
CIRCLE RULES:
- A Circle must be wrapped by an operator Group when it is used inside a Face.
- Circle.coordinates is [center_point, [0,0,0], normal_vector, [0,0,0]].
- Circle.parameters.r is the radius and wire_type must be true.
- For a circular boss, create a Prism from a Face containing a circle Group.
- For a circular hole or pocket, create a circular Prism cutter and subtract it with Cut.
""",
    "boolean": """
BOOLEAN COMPOSITION RULES:
- Use Fuse to join two or more solids/features.
- Use Cut to subtract cutter solids from a base solid. The first child is the base; later children are cutters.
- Boolean nodes have group="operator", coordinates=[], and parameters={}.
- Nested Fuse/Cut trees are allowed when the object has multiple additive and subtractive operations.
""",
    "multi_prism": """
MULTI-PRISM RULES:
- Use multiple Prism nodes when different features have different profiles, heights, offsets, or boolean roles.
- Combine additive Prisms with Fuse before applying subtractive Cut operations when this matches the design intent.
- Do not merge unrelated sketches into one Face if they represent separate construction operations.
""",
    "software_command_mapping": """
SOFTWARE COMMAND TO CADMIUM-LITE MAPPING RULES:
- Box/cube/cuboid/rectangular block/长方体/立方体/矩形块: create a Prism from a rectangular Face. Use four ordered Edge nodes in a closed Wire: (0,0,0)->(x,0,0)->(x,y,0)->(0,y,0)->(0,0,0). Use Prism.parameters.height for z.
- Cylinder/圆柱/圆柱体: create a Prism from a Face containing a circle Group. Use Circle.parameters.r as radius and Prism.parameters.height as height.
- Circle/圆/圆形轮廓: use Circle only as a sketch/profile primitive. Wrap Circle with Group when it is inside a Face.
- Straight line/线段/边: use Edge with two 3D points.
- Boolean difference/cut/切除/减去/孔/挖孔: use Cut. Base solid first, cutter solid second or later.
- Boolean union/compound/fuse/合并/并集/组合: use Fuse for additive solids.
- Never create Vertex/Sphere/Cone as Cadmium-lite node types; they are software commands but not allowed output nodes here.
""",
    "primitive_recipe": """
PRIMITIVE MODELING RECIPES:
- For a simple box, prefer one Prism with a rectangular Wire. Do not build it as multiple separate faces.
- For two separate boxes, create two Prism nodes and combine them with Fuse or place them under the root Group according to the design intent. Give them separated coordinates instead of overlapping at the origin.
- For a cylinder, use one circular profile Prism rather than approximating the circle with many Edges.
- For a circular hole through a box, create the box Prism as the Cut base, create a cylinder Prism as the cutter, then wrap them in Cut.
- For multiple holes, use one Cut node whose first child is the base and whose later children are all cutter Prisms.
- If the user gives sizes like length/width/height or radius/diameter/depth, directly map those values to coordinates and parameters. If only relative size is described, choose simple proportional dimensions such as 10, 20, 30.
- Keep models centered or laid out clearly: when creating multiple separate solids, separate them along X or Y by at least one object width unless the user asks them to touch or fuse.
""",
    "quality_check": """
FINAL SELF-CHECK BEFORE OUTPUT:
- Root is a Group.
- No comments or markdown are present.
- All nodes use exactly the dataset keys: type/features/group/coordinates/parameters.
- All primitive coordinates use 3D points.
- Edge has 2 coordinate points; Arc has 3; Circle has 4.
- Prism has exactly one Face child; Face contains Wire and/or circle Group children.
"""
}


DEFAULT_BLOCKS = ("prism_face", "wire_edge", "software_command_mapping", "primitive_recipe", "quality_check")

KEYWORD_BLOCK_HINTS = {
    "primitive_recipe": [
        r"\bbox\b", r"\bcube\b", r"\bcuboid\b", r"\bblock\b", r"\brectangular\b",
        r"\bcylinder\b", r"\bcircle\b", r"\bsphere\b", r"\bcone\b",
        r"长方体", r"立方体", r"矩形块", r"方块", r"盒子", r"圆柱", r"圆", r"圆形", r"球", r"圆锥"
    ],
    "arc": [
        r"\barc\b", r"\bround(ed)?\b", r"\bfillet\b", r"\bcurved?\b",
        r"圆弧", r"圆角", r"倒圆", r"弧"
    ],
    "circle": [
        r"\bcircle\b", r"\bcircular\b", r"\bcylinder\b", r"\bhole\b", r"\bbore\b",
        r"圆形", r"圆盘", r"圆孔", r"孔", r"圆柱", r"通孔", r"盲孔"
    ],
    "boolean": [
        r"\bcut\b", r"\bsubtract\b", r"\bremove\b", r"\bfuse\b", r"\bjoin\b", r"\bunion\b",
        r"切除", r"减去", r"布尔", r"合并", r"相加", r"挖", r"孔", r"圆孔", r"通孔", r"盲孔", r"中空"
    ],
    "multi_prism": [
        r"\bmultiple\b", r"\bseparate\b", r"\bstep\b", r"\bboss\b", r"\bpocket\b", r"\bslot\b",
        r"多个", r"多级", r"凸台", r"台阶", r"凹槽", r"槽", r"口袋"
    ],
}

def strip_negated_feature_keywords(text: str) -> str:
    """Remove geometry feature words from negated contexts to avoid falsely triggering arc / boolean / circle rules.

    Examples:
    - "Does not contain holes, chamfers, fillets" should not trigger Cut or Arc;
    - "A cylinder without a circular hole" should keep "cylinder" but drop "without a circular hole".
    """
    if not text:
        return ""

    feature_alt = (
        r"(?:圆孔|通孔|盲孔|孔|中空|切除|挖|倒角|圆角|圆弧|弧形|弧|"
        r"缺口|槽|凹槽|口袋|"
        r"hole|bore|cut|subtract|remove|fillet|rounded|arc|slot|pocket)"
    )

    negated_feature_list = (
        rf"(?:不包含|不含|没有|无|不存在|不要|无需|not|no|without)\s*"
        rf"{feature_alt}(?:[、,，/和及\s]*{feature_alt})*"
    )

    text = re.sub(negated_feature_list, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"除上述几何外没有其他结构", " ", text)
    text = re.sub(r"除上述结构外没有其他结构", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _walk_json_types(obj: Any) -> Iterable[str]:
    """Yield all Cadmium node type names found in a JSON-like object."""
    if isinstance(obj, dict):
        type_name = obj.get("type")
        if isinstance(type_name, str):
            yield type_name
        for child in obj.get("features", []):
            yield from _walk_json_types(child)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_json_types(item)


def collect_node_types(obj: Any) -> list[str]:
    """Return all Cadmium node type names in traversal order."""
    return list(_walk_json_types(obj))


def infer_expected_types_from_json(json_desc: Any) -> list[str]:
    """Infer expected type labels from a reference Cadmium-lite JSON object or JSON string."""
    if isinstance(json_desc, str):
        try:
            json_desc = json.loads(json_desc)
        except json.JSONDecodeError:
            return []
    return sorted(set(collect_node_types(json_desc)))


def infer_instruction_blocks(
    json_desc: Optional[Any] = None,
    text_desc: Optional[str] = None,
    extra_blocks: Optional[Iterable[str]] = None,
) -> Set[str]:
    """Infer which optional instruction blocks should be added.

    Revision:
    - When a reference JSON is available, prefer adding rule blocks from the actual types present in it;
    - Before matching natural-language keywords, remove feature words from negated contexts;
    - Avoid "does not contain holes, chamfers, fillets" falsely triggering boolean / arc.
    """
    blocks: Set[str] = set(DEFAULT_BLOCKS)

    if isinstance(json_desc, str):
        try:
            json_desc = json.loads(json_desc)
        except json.JSONDecodeError:
            json_desc = None

    if json_desc is not None:
        types = set(_walk_json_types(json_desc))
        if {"Fuse", "Cut"} & types:
            blocks.add("boolean")
        if "Circle" in types:
            blocks.add("circle")
        if "Arc" in types:
            blocks.add("arc")
        if sum(1 for t in _walk_json_types(json_desc) if t == "Prism") > 1:
            blocks.add("multi_prism")

    if text_desc:
        cleaned_text = strip_negated_feature_keywords(text_desc)
        lowered = cleaned_text.lower()

        for block_name, patterns in KEYWORD_BLOCK_HINTS.items():
            if any(re.search(pattern, lowered) for pattern in patterns):
                blocks.add(block_name)

    if extra_blocks:
        blocks.update(extra_blocks)

    return blocks

def build_system_message(
    json_desc: Optional[Any] = None,
    text_desc: Optional[str] = None,
    blocks: Optional[Iterable[str]] = None,
    extra_blocks: Optional[Iterable[str]] = None,
) -> str:
    """Build the fixed Cadmium-lite generation prompt.

    - If blocks is provided, it is used exactly.
    - Otherwise blocks are inferred from json_desc/text_desc plus DEFAULT_BLOCKS.
    - extra_blocks can force-add specific instruction blocks.
    """
    if blocks is None:
        selected_blocks = infer_instruction_blocks(
            json_desc=json_desc,
            text_desc=text_desc,
            extra_blocks=extra_blocks,
        )
    else:
        selected_blocks = set(blocks)
        if extra_blocks:
            selected_blocks.update(extra_blocks)

    ordered_block_names = [
        "prism_face",
        "wire_edge",
        "software_command_mapping",
        "primitive_recipe",
        "arc",
        "circle",
        "boolean",
        "multi_prism",
        "quality_check",
    ]

    block_text = "\n".join(
        INSTRUCTION_BLOCKS[name].strip()
        for name in ordered_block_names
        if name in selected_blocks
    )

    return (
        BASE_SYSTEM_MESSAGE_TEMPLATE.format(
            CADMIUM_SCHEMA_REFERENCE=CADMIUM_SCHEMA_REFERENCE,
            SOFTWARE_COMMAND_SEMANTIC_REFERENCE=SOFTWARE_COMMAND_SEMANTIC_REFERENCE,
        ).strip()
        + "\n\n"
        + block_text
    )


SYSTEM_MESSAGE = build_system_message()
