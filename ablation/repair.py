# -*- coding: utf-8 -*-
"""Iterative error-aware CAD-JSON repair.

This module owns the repair loop shown in the paper figure:
    (T, y_t, E_t) -> LLM repair -> validation barrier -> pass/reject

It is deliberately reference-free: ground-truth/reference JSON must never be
passed to this module.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Tuple

from validation import compact_repair_errors

SYSTEM_PROMPT = (
    "You are a 3D CAD JSON generator. Generate a 3D modeling structure that conforms to the CAD-JSON specification based on the user's natural language requirements. "
    "Output only JSON, without Markdown, explanations, or comments. The root node must be a Group. Each node must contain exactly the five fields "
    "type/features/group/coordinates/parameters. The type can only be one of "
    "Group/Fuse/Cut/Prism/Face/Wire/Edge/Arc/Circle."
)


REPAIR_RULES = """Repair requirements:
1. Output only one complete JSON object. Do not output Markdown, explanations, or comments.
2. Prefer minimal changes; preserve the already-correct geometric semantics, node types, coordinates, dimensions, boolean relationships, and hierarchy from the previous round.
3. You must address each path/code/message given in the "validation errors" one by one; do not modify unrelated nodes without justification.
4. CAD-JSON Schema constraints: the root node must be a Group; each node must contain exactly type/features/group/coordinates/parameters; the group field for operator/creator nodes must be correct.
5. Geometric constraints: Wire must be closed; Face must be constructible from valid contours; Prism must form a valid solid with non-zero volume; Fuse/Cut must actually produce a geometric effect.
6. Do not delete Edge/Arc/Circle/Cut/Fuse just to pass validation, or simplify complex contours into template solids without justification.
7. Do not use any reference answers, ground truth, or expected node types; repair only based on the original task, the previous output, and validation errors.
"""


def build_repair_messages(
    requirement: str,
    previous: str,
    validation_report: Dict[str, Any],
    *,
    feedback_mode: str = "structured",
) -> List[Dict[str, str]]:
    """Build the repair context C_{t+1} = (T, y_t, E_t)."""
    errors = compact_repair_errors(validation_report)

    if feedback_mode == "none":
        err_text = "输出未通过验证，请根据原始任务重新检查并修复。"
    elif feedback_mode == "generic":
        stages = sorted({e.get("stage", "unknown") for e in errors})
        err_text = "未通过的验证阶段: " + ", ".join(stages)
    elif feedback_mode == "structured":
        err_text = json.dumps(errors, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unknown feedback_mode: {feedback_mode}")

    content = f"""原始建模需求：
{requirement}

上一轮模型输出：
{previous}

验证错误：
{err_text}

{REPAIR_RULES}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def iterative_repair(
    *,
    requirement: str,
    initial_text: str,
    initial_validation: Dict[str, Any],
    generate_fn: Callable[[List[Dict[str, str]]], str],
    validate_fn: Callable[[str], Dict[str, Any]],
    max_rounds: int = 3,
    feedback_mode: str = "structured",
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], int]:
    """Run iterative repair until validation passes or the budget is exhausted.

    Parameters
    ----------
    requirement:
        Original natural-language task T.
    initial_text:
        Initial model generation y_0.
    initial_validation:
        Validation report for y_0.
    generate_fn:
        Callback that accepts chat messages and returns one model response.
        The caller can bind API/local-model details outside this module.
    validate_fn:
        Callback implementing Syntax -> Schema -> Geometry validation.
    max_rounds:
        Maximum number of repair attempts after the initial generation.
    feedback_mode:
        structured / generic / none, used directly for feedback ablations.

    Returns
    -------
    final_text, final_validation, repair_history, repair_rounds
    """
    current = initial_text
    current_val = initial_validation
    history: List[Dict[str, Any]] = []
    rounds = 0

    # If the first generation already passes the entire barrier, normalize it.
    if current_val.get("valid") is True:
        normalized = current_val.get("normalized_json")
        if normalized:
            current = normalized
        return current, current_val, history, rounds

    for r in range(1, max(0, int(max_rounds)) + 1):
        rounds = r
        before_errors = compact_repair_errors(current_val)
        messages = build_repair_messages(
            requirement,
            current,
            current_val,
            feedback_mode=feedback_mode,
        )

        repaired = generate_fn(messages)
        repaired_val = validate_fn(repaired)

        history.append(
            {
                "round": r,
                "input_errors": before_errors,
                "output_valid": bool(repaired_val.get("valid")),
                "remaining_errors": compact_repair_errors(repaired_val),
            }
        )

        current = repaired
        current_val = repaired_val

        if current_val.get("valid") is True:
            normalized = current_val.get("normalized_json")
            if normalized:
                current = normalized
            break

    return current, current_val, history, rounds
