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
    "你是一个三维 CAD JSON 生成器。根据用户自然语言需求生成符合 CAD-JSON 规范的三维建模结构。"
    "只输出 JSON，不输出 Markdown、解释或注释。根节点必须是 Group。每个节点必须且只能包含 "
    "type/features/group/coordinates/parameters 五个字段。type 只能取 "
    "Group/Fuse/Cut/Prism/Face/Wire/Edge/Arc/Circle。"
)

REPAIR_RULES = """修复要求：
1. 只输出一个完整 JSON 对象，不要输出 Markdown、解释或注释。
2. 优先做最小修改；保留上一轮中已经正确的几何语义、节点类型、坐标、尺寸、布尔关系和层级。
3. 必须逐条处理“验证错误”中给出的 path/code/message；不要凭空改动无关节点。
4. CAD-JSON Schema 约束：根节点为 Group；每个节点恰好包含 type/features/group/coordinates/parameters；operator/creator 的 group 必须正确。
5. 几何约束：Wire 必须闭合；Face 必须能由有效轮廓构造；Prism 必须形成非零体积有效实体；Fuse/Cut 必须真正产生几何效果。
6. 不要为了通过检查而删除 Edge/Arc/Circle/Cut/Fuse，或把复杂轮廓无依据地简化成模板体。
7. 不使用任何参考答案、ground truth 或预期节点类型；只能依据原始任务、上一轮输出和验证错误修复。
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
