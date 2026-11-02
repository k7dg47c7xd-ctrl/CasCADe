import argparse
import json
import math
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ALLOWED_TYPES = {
    "Group", "Fuse", "Cut", "Prism", "Face", "Wire", "Edge", "Arc", "Circle"
}

REQUIRED_KEY_ORDER = ["type", "features", "group", "coordinates", "parameters"]
REQUIRED_KEYS = set(REQUIRED_KEY_ORDER)

GROUP_RULES = {
    "Group": "operator",
    "Fuse": "operator",
    "Cut": "operator",
    "Prism": "operator",
    "Face": "operator",
    "Wire": "operator",
    "Edge": "creator",
    "Arc": "creator",
    "Circle": "creator"
}

COORD_COUNT_RULES = {
    "Group": 0,
    "Fuse": 0,
    "Cut": 0,
    "Prism": 2,
    "Face": 3,
    "Wire": 3,
    "Edge": 2,
    "Arc": 3,
    "Circle": 4
}

PARAM_RULES = {
    "Group": {"use_result_filter", "exclude", "return_index_list"},
    "Prism": {"height", "use_height", "auto_normal_vector", "combin"},
    "Face": {"combin", "mode_flag", "uv_list"},
    "Wire": {"vertex_input", "edge_sort", "linear_to_lerance"},
    "Edge": {"pline", "combin"},
    "Arc": {"r", "alpha1", "alpha2", "circle_mode", "arc_mode", "sense"},
    "Circle": {"r", "wire_type"},
    "Fuse": set(),
    "Cut": set()
}

CHILD_RULES = {
    "Prism": {"Face"},
    "Face": {"Wire", "Group"},
    "Wire": {"Edge", "Arc"},
    "Edge": set(),
    "Arc": set(),
    "Circle": set()
}

REFERENCE_COMPARE_WEIGHTS = {
    "type_count_score": 25.0,
    "preorder_type_score": 15.0,
    "hierarchy_score": 20.0,
    "coordinate_score": 10.0,  # Keep the geometric coordinate weight unchanged
    "parameter_score": 20.0,
    "node_count_score": 0.0,
    "boolean_operation_score": 10.0
}

# -------------------
# Basic JSON validation and type collection
# -------------------

def extract_json_from_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()
    return text

def load_json_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")

def is_3d_point(x: Any) -> bool:
    return isinstance(x, list) and len(x) == 3 and all(isinstance(v, (int, float)) for v in x)

def collect_types(node: Any, types: Optional[List[str]] = None) -> List[str]:
    if types is None:
        types = []
    if isinstance(node, dict):
        node_type = node.get("type")
        if isinstance(node_type, str):
            types.append(node_type)
        for child in node.get("features", []):
            collect_types(child, types)
    return types

def count_nodes(node: Any) -> int:
    if not isinstance(node, dict):
        return 0
    total = 1
    for child in node.get("features", []):
        total += count_nodes(child)
    return total

def validate_node(node: Any, errors: List[str], path: str = "root") -> None:
    if not isinstance(node, dict):
        errors.append(f"{path}: node is not a dict")
        return
    actual_key_order = list(node.keys())
    keys = set(actual_key_order)
    if keys != REQUIRED_KEYS:
        errors.append(f"{path}: fields must be exactly {REQUIRED_KEY_ORDER}, got {actual_key_order}")
    elif actual_key_order != REQUIRED_KEY_ORDER:
        errors.append(f"{path}: field order must be {REQUIRED_KEY_ORDER}, got {actual_key_order}")
    node_type = node.get("type")
    features = node.get("features")
    group = node.get("group")
    coordinates = node.get("coordinates")
    parameters = node.get("parameters")
    if node_type not in ALLOWED_TYPES:
        errors.append(f"{path}: invalid type: {node_type}")
        return
    if group != GROUP_RULES[node_type]:
        errors.append(f"{path}: {node_type} group should be {GROUP_RULES[node_type]}, got {group}")
    for i, child in enumerate(features):
        validate_node(child, errors, path=f"{path}.features[{i}]")

# -------------------
# Compare against the reference JSON
# -------------------

def _walk_nodes_with_paths(node: Any, path: str = "root", parent_type: str = "") -> List[Tuple[str, str, Dict[str, Any]]]:
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    if not isinstance(node, dict):
        return out
    out.append((path, parent_type, node))
    features = node.get("features", [])
    if isinstance(features, list):
        for i, child in enumerate(features):
            if isinstance(child, dict):
                out.extend(_walk_nodes_with_paths(child, path=f"{path}.features[{i}]", parent_type=str(node.get("type", ""))))
    return out

def _type_sequence(obj: Any) -> List[str]:
    return [node.get("type", "") for _, _, node in _walk_nodes_with_paths(obj)]

def _hierarchy_sequence(obj: Any) -> List[str]:
    edges: List[str] = []
    for _path, parent_type, node in _walk_nodes_with_paths(obj):
        if parent_type:
            edges.append(f"{parent_type}->{node.get('type','')}")
    return edges

def _boolean_operation_sequence(obj: Any) -> List[str]:
    return [t for t in _type_sequence(obj) if t in {"Fuse", "Cut", "Prism"}]

def _flatten_numbers(value: Any) -> List[float]:
    nums: List[float] = []
    if isinstance(value, bool):
        return nums
    if isinstance(value, (int, float)):
        if math.isfinite(value):
            nums.append(float(value))
    elif isinstance(value, dict):
        for key in sorted(value.keys()):
            nums.extend(_flatten_numbers(value[key]))
    elif isinstance(value, (list, tuple)):
        for item in value:
            nums.extend(_flatten_numbers(item))
    return nums

def _numeric_list_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    n = max(len(a), len(b))
    if n == 0:
        return 1.0
    total = 0.0
    for i in range(n):
        if i >= len(a) or i >= len(b):
            total += 0.0
        else:
            total += 1.0 - abs(a[i]-b[i])/max(abs(a[i]),abs(b[i]),0.05)
    return total / n

def _scalar_similarity(a: Any, b: Any) -> float:
    if isinstance(a,bool) or isinstance(b,bool):
        return 1.0 if a is b else 0.0
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        if not math.isfinite(a) or not math.isfinite(b):
            return 0.0
        return max(0.0, 1.0-abs(a-b)/max(abs(a),abs(b),0.05))
    if isinstance(a,str) or isinstance(b,str):
        return 1.0 if str(a)==str(b) else 0.0
    if isinstance(a,list) or isinstance(b,list):
        return _numeric_list_similarity(_flatten_numbers(a), _flatten_numbers(b))
    if isinstance(a,dict) or isinstance(b,dict):
        keys = sorted(set(a.keys() if isinstance(a,dict) else set()) | set(b.keys() if isinstance(b,dict) else set()))
        return sum(_scalar_similarity(a.get(k,0),b.get(k,0)) for k in keys)/max(len(keys),1)
    return 1.0 if a==b else 0.0

def _align_nodes_by_type(reference_nodes: List[Dict[str,Any]], generated_nodes: List[Dict[str,Any]]) -> List[Tuple[Dict[str,Any],Optional[Dict[str,Any]]]]:
    used: Set[int] = set()
    pairs: List[Tuple[Dict[str,Any],Optional[Dict[str,Any]]]] = []
    for ref in reference_nodes:
        ref_type = ref.get("type")
        best_idx: Optional[int] = None
        for i, gen in enumerate(generated_nodes):
            if i in used:
                continue
            if gen.get("type")==ref_type:
                best_idx=i
                break
        if best_idx is None:
            pairs.append((ref,None))
        else:
            used.add(best_idx)
            pairs.append((ref,generated_nodes[best_idx]))
    return pairs

def _coordinates_similarity(reference_obj: Any, generated_obj: Any) -> float:
    ref_nodes = [node for _,_,node in _walk_nodes_with_paths(reference_obj)]
    gen_nodes = [node for _,_,node in _walk_nodes_with_paths(generated_obj)]
    pairs = _align_nodes_by_type(ref_nodes,gen_nodes)
    scores=[]
    for ref,gen in pairs:
        if gen is None:
            scores.append(0.0)
            continue
        ref_nums=_flatten_numbers(ref.get("coordinates",[]))
        gen_nums=_flatten_numbers(gen.get("coordinates",[]))
        scores.append(_numeric_list_similarity(ref_nums,gen_nums))
    return sum(scores)/len(scores) if scores else 1.0

def _parameters_similarity(reference_obj: Any, generated_obj: Any) -> float:
    ref_nodes = [node for _,_,node in _walk_nodes_with_paths(reference_obj)]
    gen_nodes = [node for _,_,node in _walk_nodes_with_paths(generated_obj)]
    pairs = _align_nodes_by_type(ref_nodes,gen_nodes)
    scores=[]
    for ref,gen in pairs:
        if gen is None:
            scores.append(0.0)
            continue
        scores.append(_scalar_similarity(ref.get("parameters",{}),gen.get("parameters",{})))
    return sum(scores)/len(scores) if scores else 1.0

def _counter_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    ca = Counter(a)
    cb = Counter(b)

    if not ca and not cb:
        return 1.0

    keys = set(ca) | set(cb)
    matched = sum(min(ca[k], cb[k]) for k in keys)
    total = sum(max(ca[k], cb[k]) for k in keys)

    return float(matched / total) if total else 1.0

def compare_cadjson_to_reference(generated_json: Any, reference_json: Any) -> Dict[str,Any]:
    gen_types = _type_sequence(generated_json)
    ref_types = _type_sequence(reference_json)
    ref_nodes = [node for _,_,node in _walk_nodes_with_paths(reference_json)]
    gen_nodes = [node for _,_,node in _walk_nodes_with_paths(generated_json)]
    type_count_ratio=_counter_similarity(ref_types,gen_types)
    preorder_ratio=SequenceMatcher(a=ref_types,b=gen_types,autojunk=False).ratio()
    hierarchy_ratio=SequenceMatcher(a=_hierarchy_sequence(reference_json),b=_hierarchy_sequence(generated_json),autojunk=False).ratio()
    coordinate_ratio=_coordinates_similarity(reference_json,generated_json)
    parameter_ratio=_parameters_similarity(reference_json,generated_json)
    boolean_ratio=SequenceMatcher(a=_boolean_operation_sequence(reference_json),b=_boolean_operation_sequence(generated_json),autojunk=False).ratio()
    ref_count=len(ref_nodes)
    gen_count=len(gen_nodes)
    node_count_ratio=1.0 if ref_count==gen_count else min(ref_count,gen_count)/max(ref_count,gen_count) if max(ref_count,gen_count) else 1.0
    scores={
        "type_count_score": round(REFERENCE_COMPARE_WEIGHTS["type_count_score"]*type_count_ratio,2),
        "preorder_type_score": round(REFERENCE_COMPARE_WEIGHTS["preorder_type_score"]*preorder_ratio,2),
        "hierarchy_score": round(REFERENCE_COMPARE_WEIGHTS["hierarchy_score"]*hierarchy_ratio,2),
        "coordinate_score": round(REFERENCE_COMPARE_WEIGHTS["coordinate_score"]*coordinate_ratio,2),
        "parameter_score": round(REFERENCE_COMPARE_WEIGHTS["parameter_score"]*parameter_ratio,2),
        "node_count_score": round(REFERENCE_COMPARE_WEIGHTS["node_count_score"]*node_count_ratio,2),
        "boolean_operation_score": round(REFERENCE_COMPARE_WEIGHTS["boolean_operation_score"]*boolean_ratio,2),
    }
    scores["reference_compare_score"]=round(sum(scores.values()),2)
    return scores

def score_json_output_with_reference(
    output_text: str,
    reference_text: Optional[str] = None,
    reference_path: str | Path = "",
    expected_types: Optional[Sequence[str]] = None,
    requirement: str = "",
    software_open: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Compatible with the call style of chat_modified_qwen.py / legacy chat_modified_siliconflow.py.

    Current scoring policy:
    - With reference_path/reference_text: the final score is dominated by comparing the generated JSON with the reference JSON;
    - requirement, expected_types, and software_open are kept but no longer dominate the score;
    - This avoids large changes in the main script.
    """
    result: Dict[str, Any] = {
        "json_valid": 0,
        "schema_score": 0,
        "semantic_score": 0,
        "software_score": None,
        "total_score_without_software": 0.0,
        "total_score": None,
        "max_score_without_software": 100,
        "max_score": 100,
        "node_count": 0,
        "generated_types": [],
        "errors": [],
        "reference_available": False,
        "reference_error": "",
        "reference_compare_score": 0.0,
        "final_score": 0.0,
    }

    try:
        generated_obj = json.loads(extract_json_from_text(output_text))
        result["json_valid"] = 10
        result["generated_types"] = collect_types(generated_obj)
        result["node_count"] = count_nodes(generated_obj)
    except Exception as e:
        result["errors"].append(f"generated_json_parse_failed: {type(e).__name__}: {e}")
        return result

    try:
        if reference_text:
            reference_obj = json.loads(extract_json_from_text(reference_text))
        elif reference_path:
            reference_obj = json.loads(load_json_text(reference_path))
        else:
            result["reference_error"] = "reference_path_or_reference_text_missing"
            return result
    except Exception as e:
        result["reference_error"] = f"reference_json_parse_failed: {type(e).__name__}: {e}"
        return result

    compare = compare_cadjson_to_reference(generated_obj, reference_obj)

    result.update(compare)
    result["reference_available"] = True
    result["reference_compare_score"] = float(compare.get("reference_compare_score", 0.0) or 0.0)
    result["final_score"] = round(result["reference_compare_score"], 2)
    result["total_score_without_software"] = result["final_score"]
    result["max_score_without_software"] = 100

    if software_open is not None:
        result["software_score"] = 10 if software_open else 0
        result["total_score"] = round(result["final_score"] + result["software_score"], 2)
        result["max_score"] = 110
    else:
        result["total_score"] = None
        result["max_score"] = 100

    return result

def main():
    parser=argparse.ArgumentParser(description="Automatic CAD-JSON scoring (reference-JSON comparison dominated)")
    parser.add_argument("--input", required=True, help="Path to the model-generated JSON file")
    parser.add_argument("--reference", required=True, help="Path to the reference JSON file")
    args=parser.parse_args()
    input_text=load_json_text(args.input)
    result=score_json_output_with_reference(input_text,args.reference)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()