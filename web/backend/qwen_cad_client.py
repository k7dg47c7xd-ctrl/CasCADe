# -*- coding: utf-8 -*-
"""
Qwen CAD JSON runtime client.

This file is a trimmed-down rewrite of the original chat_modified_qwen.py:
- Kept: Qwen / DashScope OpenAI-compatible calls, dynamic prompts, JSON extraction, optional normalization, one-shot repair.
- Removed: the Gradio visualization UI, batch evaluation, scoring rules, and fixed dataset reading.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except ImportError:  # Raise a clear error at API call time instead
    OpenAI = None

try:
    from finetuned_model_client import call_finetuned_api
except Exception:
    call_finetuned_api = None

from qwen_prompts import build_system_message

try:
    from cadjson_template_framework import normalize_cadjson_text
except Exception:  # Stay runnable; fall back to JSON extraction only when the import fails
    normalize_cadjson_text = None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
QWEN_TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.1"))
QWEN_TOP_P = float(os.getenv("QWEN_TOP_P", "0.8"))
QWEN_MAX_TOKENS = int(os.getenv("QWEN_MAX_TOKENS", "8192"))
QWEN_ENABLE_THINKING = _env_bool("QWEN_ENABLE_THINKING", False)
QWEN_KEEP_HISTORY = _env_bool("QWEN_KEEP_HISTORY", False)
QWEN_NORMALIZE_OUTPUT = _env_bool("QWEN_NORMALIZE_OUTPUT", True)
QWEN_REPAIR_RETRIES = int(os.getenv("QWEN_REPAIR_RETRIES", "1"))


def get_dashscope_api_key() -> str:
    """Prefer DASHSCOPE_API_KEY, with QWEN_API_KEY as a compatible fallback."""
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY or QWEN_API_KEY is not configured.")
    return api_key


def get_client() -> OpenAI:
    if OpenAI is None:
        raise ImportError("The openai dependency is missing; install it first with: pip install openai")

    return OpenAI(
        api_key=get_dashscope_api_key(),
        base_url=DASHSCOPE_BASE_URL,
        timeout=300.0,
        max_retries=1,
    )


def build_user_prompt(requirement: str) -> str:
    return f"""现在请根据下面的建模需求生成 CAD-JSON：

建模需求：
{requirement}

重要解释：
1. 如果需求中出现“除上述几何外没有其他结构”，它只表示不要添加额外结构。
2. 不要因为否定语境中的词生成 Cut、Arc、孔、槽或其他附加结构。
3. 只有当需求明确要求存在某个结构时，才生成对应节点。

再次强调：
1. 只输出 JSON。
2. 不要输出 Markdown。
3. 不要输出解释、注释或多余文字。
4. 输出必须可以被 Python json.loads() 解析。
"""


def extract_json_from_text(text: str) -> str:
    """Extract the first JSON object from the model response text."""
    if text is None:
        raise ValueError("The model returned empty output.")

    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    if raw.startswith("{") and raw.endswith("}"):
        return raw

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in the model response.")
    return raw[start : end + 1].strip()


def canonicalize_json_text(json_text: str) -> str:
    obj = json.loads(json_text)
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _try_parse_json_text(text: str) -> Optional[Any]:
    if not text or not text.strip():
        return None
    try:
        return json.loads(extract_json_from_text(text))
    except Exception:
        return None


def format_messages_for_log(messages: Sequence[Dict[str, str]]) -> str:
    chunks: List[str] = []
    for i, msg in enumerate(messages, start=1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        chunks.append(f"===== message_{i}: {role} =====\n{content}")
    return "\n\n".join(chunks)


def build_model_messages(
    requirement: str,
    file_content: str = "",
    chat_history: Optional[List[Dict[str, str]]] = None,
    keep_history: Optional[bool] = None,
) -> List[Dict[str, str]]:
    final_requirement = requirement
    json_hint = None

    if file_content:
        json_hint = _try_parse_json_text(file_content)
        final_requirement = f"""用户上传了一个文件，文件内容如下：

{file_content}

用户的建模需求或问题是：

{requirement}
"""

    system_prompt = build_system_message(json_desc=json_hint, text_desc=final_requirement)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    use_history = QWEN_KEEP_HISTORY if keep_history is None else keep_history
    if use_history and chat_history:
        for msg in chat_history[-8:]:
            role = msg.get("role")
            content = msg.get("content")
            if role in {"user", "assistant"} and content:
                messages.append({"role": str(role), "content": str(content)})

    messages.append({"role": "user", "content": build_user_prompt(final_requirement)})
    return messages


def _message_content_to_text(content: Any) -> str:
    """Handle message.content being either a str or a structured list in OpenAI SDK responses."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def call_qwen_api(
    messages: List[Dict[str, str]],
    model_id: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Call the Qwen / DashScope OpenAI-compatible Chat Completions API.

    Prefer stream=True and fall back to non-streaming automatically on failure; the final CAD-JSON uses only content,
    never reasoning_content.
    """
    # When configured, use the same paper fine-tuned checkpoint for CAD-JSON
    # generation and repair. DashScope remains available as an explicit alternative.
    provider = os.getenv("CAD_JSON_LLM_PROVIDER", "finetuned").strip().lower()
    temperature = QWEN_TEMPERATURE if temperature is None else float(temperature)
    top_p = QWEN_TOP_P if top_p is None else float(top_p)
    max_tokens = QWEN_MAX_TOKENS if max_tokens is None else int(max_tokens)

    if provider == "finetuned":
        if call_finetuned_api is None:
            raise RuntimeError(
                "CAD_JSON_LLM_PROVIDER=finetuned but finetuned_model_client "
                "could not be imported."
            )
        return call_finetuned_api(
            messages,
            model_id=os.getenv("FINETUNED_MODEL", "").strip(),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=300.0,
        )

    if provider != "dashscope":
        raise ValueError(
            f"Unsupported CAD_JSON_LLM_PROVIDER={provider!r}; "
            "use finetuned or dashscope."
        )

    client = get_client()
    model_id = model_id or QWEN_MODEL

    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
            extra_body={"enable_thinking": QWEN_ENABLE_THINKING},
        )

        content_parts: List[str] = []
        reasoning_parts: List[str] = []

        for chunk in stream:
            if chunk is None:
                continue
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            content = getattr(delta, "content", None)
            reasoning_content = getattr(delta, "reasoning_content", None)
            if content:
                content_parts.append(_message_content_to_text(content))
            if reasoning_content:
                reasoning_parts.append(_message_content_to_text(reasoning_content))

        final_content = "".join(content_parts).strip()
        if final_content:
            return final_content

        if "".join(reasoning_parts).strip():
            raise ValueError("Qwen returned only reasoning_content and no final content. Disable thinking or switch to a non-reasoning model.")
        raise ValueError("The Qwen stream=True call succeeded, but no content was received.")

    except Exception as stream_error:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=False,
                extra_body={"enable_thinking": QWEN_ENABLE_THINKING},
            )
            choices = getattr(response, "choices", None)
            if not choices:
                raise ValueError(f"Qwen non-streaming response has no choices field: {response}")
            message = choices[0].message
            final_content = _message_content_to_text(getattr(message, "content", None)).strip()
            if final_content:
                return final_content
            reasoning_text = _message_content_to_text(getattr(message, "reasoning_content", None)).strip()
            if reasoning_text:
                raise ValueError("Qwen returned only reasoning_content and no final content. Disable thinking or switch to a non-reasoning model.")
            raise ValueError(f"Qwen non-streaming response has empty content: {response}")
        except Exception as non_stream_error:
            raise RuntimeError(
                "Qwen / DashScope API call failed.\n"
                f"base_url: {DASHSCOPE_BASE_URL}\n"
                f"model: {model_id}\n"
                f"stream_error: {type(stream_error).__name__}: {stream_error}\n"
                f"non_stream_error: {type(non_stream_error).__name__}: {non_stream_error}"
            )


def normalize_or_canonicalize(json_text: str) -> str:
    if QWEN_NORMALIZE_OUTPUT and normalize_cadjson_text is not None:
        return normalize_cadjson_text(json_text)
    return canonicalize_json_text(json_text)


def build_repair_messages(requirement: str, bad_output: str, error: str) -> List[Dict[str, str]]:
    system_prompt = build_system_message(json_desc=None, text_desc=requirement)
    repair_instruction = f"""
你是 CAD-JSON 修复器。

你的任务不是重新设计模型，而是把上一轮输出修复成合法、可解析、符合 schema 的 CAD-JSON。

必须遵守：
1. 只输出一个 JSON 对象。
2. 不要输出 Markdown。
3. 不要输出解释文字。
4. 尽量保留上一轮输出中的几何语义、节点类型、坐标、尺寸、布尔关系和层级关系。
5. 不要把复杂轮廓简化成长方体、圆柱体或其他模板化结构。
6. 只修复 JSON 语法、字段、group、coordinates 数量、parameters 键等结构性问题。

建模需求：
{requirement}

解析/规范化错误：
{error}

上一轮模型输出：
{bad_output}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": repair_instruction},
    ]


def generate_cad_json(
    requirement: str,
    history: Optional[List[Dict[str, str]]] = None,
    file_content: str = "",
) -> Tuple[Dict[str, Any], str, str, str]:
    """
    Generate a saveable CAD JSON document.

    Returns:
        cad_json_obj, json_text, raw_output, prompt_log
    """
    messages = build_model_messages(requirement, file_content=file_content, chat_history=history)
    prompt_log = format_messages_for_log(messages)

    raw_output = call_qwen_api(messages)
    current_output = raw_output
    last_error = ""

    for attempt in range(QWEN_REPAIR_RETRIES + 1):
        try:
            json_text = extract_json_from_text(current_output)
            json_text = normalize_or_canonicalize(json_text)
            cad_json_obj = json.loads(json_text)
            if not isinstance(cad_json_obj, dict):
                raise ValueError("The CAD JSON root node must be an object.")
            return cad_json_obj, json_text, raw_output, prompt_log
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= QWEN_REPAIR_RETRIES:
                raise ValueError(f"Qwen output could not be parsed as CAD JSON: {last_error}")
            repair_messages = build_repair_messages(requirement, current_output, last_error)
            current_output = call_qwen_api(
                repair_messages,
                temperature=0.0,
                top_p=0.1,
                max_tokens=QWEN_MAX_TOKENS,
            )
