# -*- coding: utf-8 -*-
"""
Reproduce the scoring convention of Teach2CAD and scripts/pre_finetune/llm-selection/chat_qwen/Rating_modified.py.
Standard library only, so the evaluation script behaves identically on different machines.
"""
import json
import math
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

ALLOWED_TYPES = {"Group", "Fuse", "Cut", "Prism", "Face", "Wire", "Edge", "Arc", "Circle"}
GROUP_RULES = {
    "Group": "operator", "Fuse": "operator", "Cut": "operator",
    "Prism": "operator", "Face": "operator", "Wire": "operator",
    "Edge": "creator", "Arc": "creator", "Circle": "creator",
}

# Identical to Text2CAD
REFERENCE_COMPARE_WEIGHTS = {
    "type_count_score": 25.0,
    "preorder_type_score": 15.0,
    "hierarchy_score": 20.0,
    "coordinate_score": 10.0,
    "parameter_score": 20.0,
    "node_count_score": 0.0,
    "boolean_operation_score": 10.0,
}


def extract_json_from_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1].strip() if start != -1 and end != -1 and end > start else text


def _walk(node: Any, path: str = "root", parent_type: str = "") -> List[Tuple[str, str, Dict[str, Any]]]:
    if not isinstance(node, dict):
        return []
    out = [(node.get("type", ""), path, node)]
    for i, child in enumerate(node.get("features", []) or []):
        out += _walk(child, f"{path}.features[{i}]", node.get("type", ""))
    return out


def _type_sequence(obj: Any) -> List[str]:
    return [t for t, _, _ in _walk(obj)]


def _hierarchy_sequence(obj: Any) -> List[str]:
    seq: List[str] = []
    def dfs(node: Any, depth: int = 0):
        if not isinstance(node, dict):
            return
        seq.append(f"{depth}:{node.get('type','')}")
        for c in node.get("features", []) or []:
            dfs(c, depth + 1)
    dfs(obj)
    return seq


def _boolean_operation_sequence(obj: Any) -> List[str]:
    seq: List[str] = []
    def dfs(node: Any):
        if not isinstance(node, dict):
            return
        t = node.get("type", "")
        if t in {"Fuse", "Cut"}:
            children = [c.get("type", "") for c in (node.get("features") or [])]
            seq.append(f"{t}({','.join(children)})")
        for c in node.get("features", []) or []:
            dfs(c)
    dfs(obj)
    return seq


def _flatten_numbers(value: Any) -> List[float]:
    out: List[float] = []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out.append(float(value))
    elif isinstance(value, (list, tuple)):
        for v in value:
            out += _flatten_numbers(v)
    elif isinstance(value, dict):
        for v in value.values():
            out += _flatten_numbers(v)
    return out


def _numeric_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) == len(b):
        sims = []
        for x, y in zip(a, b):
            d = abs(x - y)
            sims.append(math.exp(-d / max(abs(y), 1e-6)))
        return float(sum(sims) / len(sims))
    return float(SequenceMatcher(a=[str(round(x, 6)) for x in a],
                                 b=[str(round(y, 6)) for y in b], autojunk=False).ratio())


def _counter_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    if not keys:
        return 1.0
    return float(sum(min(ca.get(k, 0), cb.get(k, 0)) for k in keys) / max(sum(ca.values()), sum(cb.values()), 1))


def _align_by_type(ref_nodes: List[Dict[str, Any]], gen_nodes: List[Dict[str, Any]]):
    pairs = []
    used = set()
    for r in ref_nodes:
        best, best_d = None, float("inf")
        for j, g in enumerate(gen_nodes):
            if j in used or r.get("type") != g.get("type"):
                continue
            cost = 1.0 - _numeric_similarity(_flatten_numbers(r.get("coordinates")),
                                             _flatten_numbers(g.get("coordinates")))
            if cost < best_d:
                best_d, best = cost, j
        if best is not None:
            used.add(best)
            pairs.append((r, gen_nodes[best]))
    return pairs


def _coordinates_similarity(ref: Any, gen: Any) -> float:
    rn = [n for _, _, n in _walk(ref)]
    gn = [n for _, _, n in _walk(gen)]
    pairs = _align_by_type(rn, gn)
    if not pairs:
        return float(_counter_similarity([str(n.get("coordinates")) for n in rn],
                                         [str(n.get("coordinates")) for n in gn]))
    return float(sum(_numeric_similarity(_flatten_numbers(r.get("coordinates")),
                                         _flatten_numbers(g.get("coordinates"))) for r, g in pairs) / len(pairs))


def _parameters_similarity(ref: Any, gen: Any) -> float:
    rn = [n for _, _, n in _walk(ref)]
    gn = [n for _, _, n in _walk(gen)]
    pairs = _align_by_type(rn, gn)
    if not pairs:
        return float(_counter_similarity([str(n.get("parameters")) for n in rn],
                                         [str(n.get("parameters")) for n in gn]))
    return float(sum(_numeric_similarity(_flatten_numbers(r.get("parameters")),
                                         _flatten_numbers(g.get("parameters"))) for r, g in pairs) / len(pairs))


def compare_cadjson_to_reference(generated: Any, reference: Any) -> Dict[str, Any]:
    gt, rt = _type_sequence(generated), _type_sequence(reference)
    scores = {
        "type_count_score": round(REFERENCE_COMPARE_WEIGHTS["type_count_score"] * _counter_similarity(rt, gt), 2),
        "preorder_type_score": round(REFERENCE_COMPARE_WEIGHTS["preorder_type_score"] * SequenceMatcher(a=rt, b=gt, autojunk=False).ratio(), 2),
        "hierarchy_score": round(REFERENCE_COMPARE_WEIGHTS["hierarchy_score"] * SequenceMatcher(a=_hierarchy_sequence(reference), b=_hierarchy_sequence(generated), autojunk=False).ratio(), 2),
        "coordinate_score": round(REFERENCE_COMPARE_WEIGHTS["coordinate_score"] * _coordinates_similarity(reference, generated), 2),
        "parameter_score": round(REFERENCE_COMPARE_WEIGHTS["parameter_score"] * _parameters_similarity(reference, generated), 2),
        "boolean_operation_score": round(REFERENCE_COMPARE_WEIGHTS["boolean_operation_score"] * SequenceMatcher(a=_boolean_operation_sequence(reference), b=_boolean_operation_sequence(generated), autojunk=False).ratio(), 2),
    }
    rn = len([n for _, _, n in _walk(reference)])
    gn = len([n for _, _, n in _walk(generated)])
    node_count_ratio = 1.0 if rn == gn else (min(rn, gn) / max(rn, gn) if max(rn, gn) else 1.0)
    scores["node_count_score"] = round(REFERENCE_COMPARE_WEIGHTS["node_count_score"] * node_count_ratio, 2)
    scores["reference_compare_score"] = round(sum(scores.values()), 2)
    return scores


def score_text_against_reference(text: str, reference_obj: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {"json_valid": False, "reference_compare_score": 0.0, "errors": []}
    try:
        obj = json.loads(extract_json_from_text(text))
        result["json_valid"] = True
        result.update(compare_cadjson_to_reference(obj, reference_obj))
        result["node_count"] = len([n for _, _, n in _walk(obj)])
        result["generated_types"] = _type_sequence(obj)
    except Exception as e:
        result["errors"].append(f"{type(e).__name__}: {e}")
    return result
