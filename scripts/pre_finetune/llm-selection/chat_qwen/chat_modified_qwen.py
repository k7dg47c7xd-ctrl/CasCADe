"""
chat_modified_qwen.py

Qwen / Alibaba Cloud Model Studio (DashScope) CAD-JSON generation and batch evaluation tool.

Features:
1. Single-turn chat debugging;
3. Extract the model output into generated.json and save it to a fixed path;
4. Call Rating_modified.py for automatic scoring;
5. Batch evaluation over the simple / medium / complex folders;
6. Batch evaluation over dataset.jsonl;
7. Aggregate results.csv, results.jsonl, summary.csv, and summary.json.
= New version
= Added negated-context guards to avoid false-positive structures
= Accepts a single simple / medium / complex directory directly
= Auto-matches *_description.txt with *_lite.json, *.json, and *_standard.json
= Batch evaluation does not use few-shot by default
= Better suited to the current dataset file-naming convention
= Removed usage examples
= API switched to the Qwen / DashScope OpenAI-compatible API

Run:
python chat_modified_qwen.py

Dependencies:
pip install openai gradio

Required environment variables:
DASHSCOPE_API_KEY=<your-model-studio-api-key>

Optional environment variables:
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_ENABLE_THINKING=false

Notes:
- This version uses the OpenAI-compatible Chat Completions API of Alibaba Cloud Model Studio (DashScope).
- The default model is qwen3.7-max; it can be changed to another available model ID in the Gradio dropdown.
- Thinking mode is disabled by default for stable plain CAD-JSON returns; set QWEN_ENABLE_THINKING=true to enable it.
"""

from __future__ import annotations
import csv
import json
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import gradio as gr
except ImportError:  # Allow importing this file for offline testing/scoring only
    gr = None

try:
    from openai import OpenAI
except ImportError:  # Give a clear installation hint at API call time
    OpenAI = None

from Rating_modified import extract_json_from_text, score_json_output_with_reference, collect_types
from Rating_raw_structure import score_raw_structured_output

try:
    from cadjson_template_framework import normalize_cadjson_text
except ImportError:
    normalize_cadjson_text = None

try:
    from dataset_prompts import (
        SYSTEM_MESSAGE as DATASET_SYSTEM_MESSAGE,
        build_system_message,
        infer_expected_types_from_json,
    )
except ImportError:
    DATASET_SYSTEM_MESSAGE = None

    def build_system_message(*args, **kwargs):
        return LEGACY_SYSTEM_PROMPT

    def infer_expected_types_from_json(json_desc):
        return []


# =========================
# 1. Qwen / DashScope API configuration
# =========================

DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
REFERENCE_JSON_ROOT = Path(os.getenv("REFERENCE_JSON_ROOT", r"data/simple"))

# Model Studio model IDs change over time; the list below only contains common examples.
# The Gradio dropdown sets allow_custom_value=True so that the actual model ID from the Model Studio console can be entered manually.
DEFAULT_MODELS = [
    "qwen3.7-max",
    "qwen3.6-plus",
    "qwen3.6-27b",
    "qwen2.5-7b",
]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


QWEN_ENABLE_THINKING = _env_bool("QWEN_ENABLE_THINKING", False)


def get_dashscope_api_key() -> str:
    """
    Prefer the official Model Studio environment variable DASHSCOPE_API_KEY.
    QWEN_API_KEY is also supported so the key can be configured separately per project name.
    """
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        raise ValueError(
            "No Qwen / DashScope API key detected. Set the environment variable first:\n"
            "Windows PowerShell:\n"
            "  $env:DASHSCOPE_API_KEY=\"<your-model-studio-api-key>\"\n\n"
            "Windows CMD:\n"
            "  set DASHSCOPE_API_KEY=<your-model-studio-api-key>\n\n"
            "macOS / Linux:\n"
            "  export DASHSCOPE_API_KEY=\"<your-model-studio-api-key>\""
        )
    return api_key


def get_client() -> OpenAI:
    if OpenAI is None:
        raise ImportError("The openai package is missing; run: pip install openai")
    return OpenAI(
        api_key=get_dashscope_api_key(),
        base_url=DASHSCOPE_BASE_URL,
    )


LEGACY_SYSTEM_PROMPT = """你是一个三维 CAD JSON 生成器。

你的任务是根据用户的自然语言建模需求，生成符合 CAD-JSON 规范的三维建模结构。

必须严格遵守以下规则：

1. 只输出 JSON，不要输出 Markdown，不要输出解释文字。
2. 根节点必须是 Group。
3. 每一个节点都必须包含且只能包含以下 5 个字段：
   - type
   - features
   - group
   - coordinates
   - parameters
4. type 只能从以下集合中选择：
   ["Group", "Fuse", "Cut", "Prism", "Face", "Wire", "Edge", "Arc", "Circle"]
5. group 规则：
   - Edge、Arc、Circle 的 group 必须是 "creator"
   - Group、Fuse、Cut、Prism、Face、Wire 的 group 必须是 "operator"
6. coordinates 规则：
   - Group、Fuse、Cut：coordinates 必须为 []
   - Prism：coordinates 必须包含 2 个三维坐标
   - Face：coordinates 必须包含 3 个三维坐标
   - Wire：coordinates 必须包含 3 个三维坐标
   - Edge：coordinates 必须包含 2 个三维坐标
   - Arc：coordinates 必须包含 3 个三维坐标
   - Circle：coordinates 必须包含 4 个三维坐标
7. features 层级规则：
   - Prism 的 features 中只能包含 Face
   - Face 的 features 中可以包含 Wire 或 Group
   - Wire 的 features 中只能包含 Edge 或 Arc
   - Edge、Arc、Circle 的 features 必须为空数组
   - Fuse 和 Cut 的 features 至少包含 2 个子实体
8. parameters 规则：
   - Group: {"use_result_filter": false, "exclude": false, "return_index_list": []}
   - Prism: {"height": number, "use_height": true, "auto_normal_vector": true, "combin": false}
   - Face: {"combin": false, "mode_flag": 3, "uv_list": []}
   - Wire: {"vertex_input": false, "edge_sort": false, "linear_to_lerance": 0.000001}
   - Edge: {"pline": false, "combin": false}
   - Arc: {"r": number, "alpha1": number, "alpha2": number, "circle_mode": "0p", "arc_mode": "3p", "sense": true}
   - Circle: {"r": number, "wire_type": true}
   - Fuse 和 Cut: {}

如果建模需求涉及“合并、融合、并集”，请优先使用 Fuse。
如果建模需求涉及“切除、挖孔、中空、圆孔”，请优先使用 Cut。
如果建模需求涉及“圆柱体、圆孔、圆形”，请使用 Circle 与 Prism 表示圆形拉伸体。
"""

# Prefer the dataset templates in dataset_prompts.py; fall back to the legacy hard-coded prompt only when the import fails.
SYSTEM_PROMPT = DATASET_SYSTEM_MESSAGE or LEGACY_SYSTEM_PROMPT


# =========================
# 2. General utility functions
# =========================


def read_text_file(path: str | Path, max_chars: int = 20000) -> str:
    path = Path(path)
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="gbk")

    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[File content too long; first {max_chars} characters kept]"
    return content


def read_uploaded_file(file: Any) -> str:
    if file is None:
        return ""

    file_path = file.name if hasattr(file, "name") else file
    ext = os.path.splitext(file_path)[-1].lower()
    allowed_exts = [
        ".txt", ".md", ".py", ".json", ".jsonl", ".csv",
        ".log", ".xml", ".html", ".css", ".js"
    ]
    if ext not in allowed_exts:
        return f"Only text-like files are supported; {ext} files are not supported."

    try:
        return read_text_file(file_path, max_chars=12000)
    except Exception as e:
        return f"Failed to read the file: {e}"


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", name.replace("/", "_"))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def _try_parse_json_text(text: str) -> Optional[Any]:
    """Try to parse a text blob as JSON; return None when it is not JSON."""
    if not text or not text.strip():
        return None
    try:
        return json.loads(extract_json_from_text(text))
    except Exception:
        return None


def format_messages_for_log(messages: Sequence[Dict[str, str]]) -> str:
    """Save both system and user prompts so experiments are reproducible."""
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
    keep_history: bool = False,
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

    # Key change: the system prompt no longer uses the hard-coded rules in chat_modified_qwen.py;
    # dataset_prompts.py adds the circle / arc / boolean / multi_prism instruction blocks automatically based on the requirement.
    system_prompt = build_system_message(
        json_desc=json_hint,
        text_desc=final_requirement,
    )

    messages = [{"role": "system", "content": system_prompt}]

    if keep_history and chat_history:
        for msg in chat_history:
            if msg.get("role") in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

    messages.append({
        "role": "user",
        "content": build_user_prompt(final_requirement),
    })
    return messages


# =========================
# 3. Qwen / DashScope API calls
# =========================


def _message_content_to_text(content: Any) -> str:
    """Handle message.content being either a str or a structured list as returned by the OpenAI SDK."""
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
    model_id: str,
    temperature: float = 0.1,
    top_p: float = 0.8,
    max_tokens: int = 8192
) -> str:
    """
    Call the Qwen / DashScope OpenAI-compatible Chat Completions API.

    Logic:
    1. Prefer stream=True to accommodate long reasoning-model outputs;
    2. Fall back to non-streaming automatically if the streaming call fails;
    3. Handle reasoning_content for compatibility, but take the final CAD-JSON only from content.
    """
    client = get_client()

    try:
        request_kwargs = {
            "model": model_id,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "stream": True,
            # enable_thinking is a DashScope parameter outside the OpenAI standard; it must be passed through extra_body.
            # Disabled by default to prevent the model from placing the CAD-JSON inside reasoning_content; set QWEN_ENABLE_THINKING=true to enable it.
            "extra_body": {"enable_thinking": QWEN_ENABLE_THINKING},
        }
        stream = client.chat.completions.create(**request_kwargs)

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

        reasoning_text = "".join(reasoning_parts).strip()
        if reasoning_text:
            raise ValueError(
                "The Qwen model returned only reasoning_content and no final content."
                "Switch to a non-reasoning model, or check whether this model outputs JSON in its final answer."
            )

        raise ValueError("The Qwen / DashScope stream=True call succeeded but no content was received.")

    except Exception as stream_error:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
                stream=False,
                extra_body={"enable_thinking": QWEN_ENABLE_THINKING},
            )

            if response is None:
                raise ValueError("The Qwen / DashScope API returned an empty response.")

            choices = getattr(response, "choices", None)
            if not choices:
                response_text = ""
                try:
                    response_text = response.model_dump_json(indent=2)
                except Exception:
                    response_text = str(response)

                raise ValueError(
                    "The Qwen / DashScope API non-streaming response has no choices field.\n"
                    f"stream_error: {type(stream_error).__name__}: {stream_error}\n"
                    f"non_stream_response: {response_text}"
                )

            message = choices[0].message
            content = getattr(message, "content", None)
            reasoning_content = getattr(message, "reasoning_content", None)

            final_content = _message_content_to_text(content).strip()
            if final_content:
                return final_content

            reasoning_text = _message_content_to_text(reasoning_content).strip()
            if reasoning_text:
                raise ValueError(
                    "The Qwen model returned only reasoning_content and no final content."
                    "Switch to a non-reasoning model, or check whether this model outputs JSON in its final answer."
                )

            raise ValueError(f"The Qwen / DashScope API returned empty content: {response}")

        except Exception as non_stream_error:
            raise ValueError(
                "The Qwen / DashScope API call failed.\n"
                f"base_url: {DASHSCOPE_BASE_URL}\n"
                f"model: {model_id}\n"
                f"stream_error: {type(stream_error).__name__}: {stream_error}\n"
                f"non_stream_error: {type(non_stream_error).__name__}: {non_stream_error}"
            )


# Keep this alias for compatibility with the original file naming.
def call_siliconflow_api(
    messages: List[Dict[str, str]],
    model_id: str,
    temperature: float = 0.1,
    top_p: float = 0.8,
    max_tokens: int = 8192,
) -> str:
    return call_qwen_api(
        messages=messages,
        model_id=model_id,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def call_modelscope_api(
    messages: List[Dict[str, str]],
    model_id: str,
    temperature: float = 0.1,
    top_p: float = 0.8,
    max_tokens: int = 8192,
) -> str:
    return call_qwen_api(
        messages=messages,
        model_id=model_id,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def parse_expected_types(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in re.split(r"[,，\s]+", text) if x.strip()]

def canonicalize_json_text(json_text: str) -> str:
    """
    Re-dump parseable JSON into standard JSON text.
    Purpose:
    1. Remove formatting instability from model outputs;
    2. Guarantee that the saved generated.json is always standard JSON;
    3. Avoid indentation, escaping, or trailing-content issues in later scoring or CAD software.
    """
    obj = json.loads(json_text)
    return json.dumps(obj, ensure_ascii=False, indent=2)


def score_raw_output(
    raw_output: str,
    requirement: str,
    expected_types: Optional[List[str]] = None,
    reference_path: str = "",
) -> tuple[str, Dict[str, Any]]:
    """
    Compute two reference-based scores at the same time:

    1. raw_reference_score：
       The first raw model output (raw_output) is compared directly, without normalize or repair,
       against the original reference CAD-JSON pointed to by reference_path.

    2. optimized_reference_score：
       The model output first goes through JSON extraction and normalize_cadjson_text() normalization,
       then it is compared against the original reference CAD-JSON pointed to by reference_path.

    Returns:
        json_text: normalized JSON text used to save generated.json
        score: contains both scores and the full details
    """
    expected_types = expected_types or []

    # A. New scoring: compare the raw model output directly with the original reference JSON.
    # Note: raw_output must be used here, not the normalized json_text.
    raw_reference_detail = score_raw_structured_output(
        output_text=raw_output,
        requirement=requirement,
        expected_types=expected_types,
        reference_path=reference_path,
    )
    raw_reference_score = raw_reference_detail.get("raw_reference_score", raw_reference_detail.get("total_score_without_software", 0))

    # B. Legacy flow: extract JSON, normalize it, then compare with the original reference JSON.
    try:
        json_text = extract_json_from_text(raw_output)
    except Exception:
        json_text = raw_output

    fixed_template_normalized = False
    normalize_error = ""

    if normalize_cadjson_text is not None:
        try:
            json_text = normalize_cadjson_text(json_text)
            fixed_template_normalized = True
        except Exception as e:
            normalize_error = f"{type(e).__name__}: {e}"
            try:
                json_text = canonicalize_json_text(json_text)
            except Exception:
                pass
    else:
        normalize_error = "cadjson_template_framework.normalize_cadjson_text not imported"
        try:
            json_text = canonicalize_json_text(json_text)
        except Exception:
            pass

    optimized_reference_detail = score_json_output_with_reference(
        output_text=json_text,
        requirement=requirement,
        expected_types=expected_types,
        reference_path=reference_path,
    )
    optimized_reference_score = optimized_reference_detail.get("total_score_without_software", 0)

    # C. Keep legacy score compatibility: total_score_without_software still means the "optimized score".
    score = dict(optimized_reference_detail)
    score["total_score_without_software"] = optimized_reference_score
    score["final_score"] = optimized_reference_score

    # D. The two final primary scores.
    score["optimized_reference_score"] = optimized_reference_score
    score["raw_reference_score"] = raw_reference_score

    # E. Details are kept in score_detail.json / results.jsonl.
    score["optimized_reference_detail"] = optimized_reference_detail
    score["raw_reference_detail"] = raw_reference_detail
    score["raw_reference_source"] = "current_raw_output"

    score["fixed_template_normalized"] = fixed_template_normalized
    score["normalize_error"] = normalize_error

    if normalize_error and score.get("json_valid", 0) == 10:
        errors = score.setdefault("errors", [])
        errors.append(f"fixed_template_normalize_failed: {normalize_error}")

    return json_text, score

def build_repair_messages(
    requirement: str,
    bad_output: str,
    score: Dict[str, Any],
    expected_types: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    expected_types = expected_types or []
    errors = score.get("errors", [])

    system_prompt = build_system_message(
        json_desc=None,
        text_desc=requirement,
    )

    repair_instruction = f"""
你是 CAD-JSON 修复器。

你的任务不是重新设计模型，而是把上一轮输出修复成合法、可解析、符合 schema 的 CAD-JSON。

必须遵守：
1. 只输出一个 JSON 对象。
2. 不要输出 Markdown。
3. 不要输出解释文字。
4. 尽量保留上一轮输出中的几何语义、节点类型、坐标、尺寸、布尔关系和层级关系。
5. 不要把复杂轮廓简化成长方体、圆柱体或其他模板化结构。
6. 只修复以下问题：
   - JSON 语法错误；
   - 缺失逗号、括号、引号；
   - 多余 Markdown 包裹；
   - Python 风格 True/False/None；
   - 缺失必需字段 type/features/group/coordinates/parameters；
   - parameters 缺失必要键；
   - group 字段不符合 creator/operator 规则；
   - coordinates 数量明显不符合节点类型要求。
7. 如果上一轮已经表达了 Edge、Arc、Circle、Cut、Fuse，不要随意删除这些节点。
8. 如果预期 type 中包含某类节点，应尽量保留或补足该类节点。

建模需求：
{requirement}

预期节点类型：
{",".join(expected_types) if expected_types else "未显式提供"}

上一轮评分错误：
{json.dumps(errors, ensure_ascii=False, indent=2)}

上一轮模型输出：
{bad_output}
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": repair_instruction},
    ]

def is_json_acceptable(score: Dict[str, Any]) -> bool:
    """
    第一目标：JSON 合法。
    不在这里强行要求 schema 满分，避免修复过程过度改动语义。
    """
    return score.get("json_valid", 0) == 10

def generate_valid_cadjson(
    messages: List[Dict[str, str]],
    requirement: str,
    expected_types: Optional[List[str]],
    model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repair_retries: int = 2,
    reference_path: str = "",
) -> tuple[str, str, Dict[str, Any], int]:
    """
    Returns:
    raw_output: model output finally used for generated.json; after repair, it is the repaired output.
    json_text: extracted and normalized JSON text.
    score: final score; raw_reference_score always comes from the first raw model output.
    repair_count: actual number of repairs.
    """
    raw_output = call_qwen_api(
        messages=messages,
        model_id=model_id,
        temperature=temperature,
        top_p=top_p,
        max_tokens=int(max_tokens),
    )

    json_text, score = score_raw_output(
        raw_output=raw_output,
        requirement=requirement,
        expected_types=expected_types,
        reference_path=reference_path,
    )

    # Save the comparison score between the first raw model output and the reference model.
    # Later repairs may update optimized_reference_score but must never overwrite raw_reference_score.
    initial_raw_reference_score = score.get("raw_reference_score", 0)
    initial_raw_reference_detail = score.get("raw_reference_detail", {})
    initial_raw_output = raw_output
    score["_initial_raw_output"] = initial_raw_output
    score["raw_reference_source"] = "initial_model_output_before_repair"

    if is_json_acceptable(score):
        return raw_output, json_text, score, 0

    current_output = raw_output

    for repair_idx in range(1, repair_retries + 1):
        repair_messages = build_repair_messages(
            requirement=requirement,
            bad_output=current_output,
            score=score,
            expected_types=expected_types,
        )

        repaired_output = call_qwen_api(
            messages=repair_messages,
            model_id=model_id,
            temperature=0.0,
            top_p=0.1,
            max_tokens=int(max_tokens),
        )

        json_text, score = score_raw_output(
            raw_output=repaired_output,
            requirement=requirement,
            expected_types=expected_types,
            reference_path=reference_path,
        )

        # optimized_reference_score may be updated after repair,
        # but raw_reference_score must keep the first raw output's score.
        score["raw_reference_score"] = initial_raw_reference_score
        score["raw_reference_detail"] = initial_raw_reference_detail
        score["raw_reference_source"] = "initial_model_output_before_repair"
        score["_initial_raw_output"] = initial_raw_output

        current_output = repaired_output

        if is_json_acceptable(score):
            return repaired_output, json_text, score, repair_idx

    return current_output, json_text, score, repair_retries

def save_generation_files(
    base_dir: str | Path,
    prompt: str,
    raw_output: str,
    requirement: str,
    score: Dict[str, Any],
    json_text: str,
) -> Dict[str, str]:
    base_dir = ensure_dir(base_dir)

    prompt_path = base_dir / "prompt.txt"
    raw_path = base_dir / "raw_output.txt"
    initial_raw_path = base_dir / "initial_raw_output.txt"
    json_path = base_dir / "generated.json"
    score_path = base_dir / "score.json"
    score_detail_path = base_dir / "score_detail.json"
    requirement_path = base_dir / "requirement.txt"

    prompt_path.write_text(prompt, encoding="utf-8")
    raw_path.write_text(raw_output, encoding="utf-8")
    requirement_path.write_text(requirement, encoding="utf-8")

    # If repair happened, raw_output.txt stores the output finally used for generated.json;
    # initial_raw_output.txt stores the first raw model output to verify raw_reference_score.
    initial_raw_output = score.get("_initial_raw_output", raw_output)
    initial_raw_path.write_text(str(initial_raw_output), encoding="utf-8")

    # score.json stores only the two final readable core scores.
    visible_score = {
        "optimized_reference_score": score.get(
            "optimized_reference_score",
            score.get("total_score_without_software", 0),
        ),
        "raw_reference_score": score.get("raw_reference_score", 0),
    }
    score_path.write_text(
        json.dumps(visible_score, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # The full scoring details are saved separately.
    score_detail = dict(score)
    score_detail.pop("_initial_raw_output", None)
    score_detail_path.write_text(
        json.dumps(score_detail, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        parsed = json.loads(json_text)
        json_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # Save the extracted result even when it is not valid JSON, for troubleshooting
        json_path.write_text(json_text, encoding="utf-8")

    return {
        "prompt_path": str(prompt_path),
        "raw_output_path": str(raw_path),
        "initial_raw_output_path": str(initial_raw_path),
        "json_path": str(json_path),
        "score_path": str(score_path),
        "score_detail_path": str(score_detail_path),
        "requirement_path": str(requirement_path),
    }

# =========================
# 4. Dataset loading
# =========================


def _extract_reference_json_object(data: Dict[str, Any]) -> Optional[Any]:
    """Extract embedded reference CAD JSON from a dataset record when available."""
    for key in ["output", "reference", "answer", "target", "json", "cad_json"]:
        value = data.get(key)
        if not value:
            continue
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            parsed = _try_parse_json_text(value)
            if parsed is not None:
                return parsed
    return None


def _infer_expected_types_from_reference_path(reference_path: str) -> List[str]:
    if not reference_path:
        return []
    try:
        ref_text = read_text_file(reference_path)
        ref_obj = _try_parse_json_text(ref_text)
        if ref_obj is None:
            return []
        return infer_expected_types_from_json(ref_obj)
    except Exception:
        return []


def _normalize_expected_types(expected_types: Any, reference_obj: Optional[Any] = None, reference_path: str = "") -> List[str]:
    if isinstance(expected_types, str):
        parsed = parse_expected_types(expected_types)
    elif isinstance(expected_types, list):
        parsed = [str(x).strip() for x in expected_types if str(x).strip()]
    else:
        parsed = []

    if parsed:
        return parsed

    if reference_obj is not None:
        inferred = infer_expected_types_from_json(reference_obj)
        if inferred:
            return inferred

    return _infer_expected_types_from_reference_path(reference_path)


def read_case_from_json_file(path: Path) -> Dict[str, Any]:
    data = json.loads(read_text_file(path))
    requirement = (
        data.get("input")
        or data.get("instruction")
        or data.get("requirement")
        or data.get("prompt")
        or data.get("question")
        or ""
    )

    reference_obj = _extract_reference_json_object(data)
    reference_path = data.get("reference_path", "")
    if not reference_path and reference_obj is not None:
        reference_path = str(path)

    expected_types = _normalize_expected_types(
        data.get("expected_types") or [],
        reference_obj=reference_obj,
        reference_path=reference_path,
    )

    return {
        "id": data.get("id") or path.stem,
        "difficulty": data.get("difficulty") or path.parent.name,
        "requirement": requirement,
        "input_path": str(path),
        "reference_path": reference_path,
        "expected_types": expected_types,
    }


def read_dataset_jsonl(jsonl_path: str | Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    jsonl_path = Path(jsonl_path)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            requirement = data.get("input") or data.get("requirement") or data.get("instruction") or ""
            input_path = data.get("input_path", "")
            if not requirement and input_path:
                requirement = read_text_file(input_path)

            reference_obj = _extract_reference_json_object(data)
            reference_path = data.get("reference_path", "")
            expected_types = _normalize_expected_types(
                data.get("expected_types") or [],
                reference_obj=reference_obj,
                reference_path=reference_path,
            )

            case_id = data.get("id") or f"case_{line_no:04d}"
            difficulty = data.get("difficulty") or "unknown"

            cases.append({
                "id": case_id,
                "difficulty": difficulty,
                "requirement": requirement.strip(),
                "input_path": input_path,
                "reference_path": reference_path,
                "expected_types": expected_types,
            })

    return cases

def find_reference_json_for_description(path: Path, reference_root: Optional[Path] = None) -> str:
    stem = path.stem
    base_stems = [stem]
    if stem.endswith("_description"):
        base_stems.insert(0, stem[: -len("_description")])

    candidate_names: List[str] = []
    for base in base_stems:
        candidate_names.extend([
            base + "_lite.json",
            base + ".json",
        ])

    # Actual lookup root: prefer the passed reference_root, otherwise use the global variable
    root = reference_root if reference_root is not None else REFERENCE_JSON_ROOT

    # 1. Look directly under the root directory first
    if root and root.exists():
        for name in candidate_names:
            candidate = root / name
            if candidate.exists():
                return str(candidate)
        # 2. Search recursively
        for name in candidate_names:
            matches = sorted(root.rglob(name))
            if matches:
                return str(matches[0])

    # 3. Fall back to searching next to the description file
    for name in candidate_names:
        candidate = path.with_name(name)
        if candidate.exists():
            return str(candidate)

    return ""

def build_text_case_from_file(path: Path, difficulty: str, idx: int, reference_root: Optional[Path] = None) -> Dict[str, Any]:
    requirement = read_text_file(path).strip()
    reference_path = find_reference_json_for_description(path, reference_root)

    return {
        "id": f"{difficulty}_{path.stem}",
        "difficulty": difficulty,
        "requirement": requirement,
        "input_path": str(path),
        "reference_path": reference_path,
        "expected_types": _normalize_expected_types([], reference_path=reference_path),
    }


def discover_cases(dataset_path: str | Path, reference_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Three data input modes are supported:

    Mode 1: dataset.jsonl
      Pass the dataset.jsonl path directly.

    Mode 2: dataset root directory
      dataset/
        simple/*_description.txt
        medium/*_description.txt
        complex/*_description.txt

    Mode 3: a single difficulty directory
      Pass directly:
        data/simple

      All *_description.txt files under that directory are then read.

    reference_root: optional root directory used to locate reference JSONs.
                   When provided, this directory is searched first; otherwise it falls back to the global REFERENCE_JSON_ROOT.
    """
    dataset_path = Path(dataset_path)

    if dataset_path.is_file() and dataset_path.suffix.lower() == ".jsonl":
        return read_dataset_jsonl(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    candidate_jsonl = dataset_path / "dataset.jsonl"
    if candidate_jsonl.exists():
        return read_dataset_jsonl(candidate_jsonl)

    cases: List[Dict[str, Any]] = []
    difficulties = ["simple", "medium", "complex"]

    # Case A: the user passes data/simple directly
    if dataset_path.name.lower() in difficulties:
        difficulty = dataset_path.name.lower()

        files = sorted(dataset_path.glob("*.txt"))

        # If no *_description.txt exists, fall back to reading plain txt/md files.
        if not files:
            files = sorted(
                list(dataset_path.glob("*.txt"))
                + list(dataset_path.glob("*.md"))
            )

        for idx, path in enumerate(files, start=1):
            case = build_text_case_from_file(path, difficulty, idx, reference_root)
            if case["requirement"]:
                cases.append(case)

    # Case B: the user passes data
    else:
        for difficulty in difficulties:
            folder = dataset_path / difficulty
            if not folder.exists():
                continue

            files = sorted(folder.glob("*_description.txt"))

            # Legacy compatibility: read plain txt/md/json when no *_description.txt is present.
            if not files:
                files = sorted(
                    list(folder.glob("*.txt"))
                    + list(folder.glob("*.md"))
                    + list(folder.glob("*.json"))
                )

            for idx, path in enumerate(files, start=1):
                if path.suffix.lower() == ".json":
                    try:
                        case = read_case_from_json_file(path)
                        if not case["requirement"]:
                            continue
                    except Exception:
                        continue
                else:
                    case = build_text_case_from_file(path, difficulty, idx, reference_root)

                if not case["id"].startswith(difficulty):
                    case["id"] = f"{difficulty}_{idx:03d}"

                case["difficulty"] = difficulty

                if case["requirement"]:
                    cases.append(case)

    if not cases:
        raise ValueError(
            f"No evaluable samples were found. Current path: {dataset_path}."
            "To read a single difficulty directory, make sure it contains *_description.txt files,"
            "for example 00000007_description.txt."
        )

    print(f"[discover_cases] dataset_path={dataset_path}")
    print(f"[discover_cases] found_cases={len(cases)}")
    for case in cases[:10]:
        print(f"[case] {case['id']} -> {case['input_path']}")

    return cases


# =========================
# 5. Single-sample debugging
# =========================


def single_chat(
    user_input: str,
    file: Any,
    chat_history: Optional[List[Dict[str, str]]],
    model_id: str,
    save_dir: str,
    expected_types_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
):
    if chat_history is None:
        chat_history = []

    if not user_input or not user_input.strip():
        return chat_history, "", "Enter the modeling requirement.", {}

    file_content = read_uploaded_file(file)
    messages = build_model_messages(
        requirement=user_input,
        file_content=file_content,
        chat_history=chat_history,
        keep_history=False,
    )

    prompt_text = format_messages_for_log(messages)

    try:
        expected_types = parse_expected_types(expected_types_text)

        raw_output, json_text, score, repair_count = generate_valid_cadjson(
            messages=messages,
            requirement=user_input,
            expected_types=expected_types,
            model_id=model_id,
            temperature=temperature,
            top_p=top_p,
            max_tokens=int(max_tokens),
            repair_retries=2,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = safe_name(model_id)
        case_dir = Path(save_dir) / model_name / "single" / timestamp
        paths = save_generation_files(
            base_dir=case_dir,
            prompt=prompt_text,
            raw_output=raw_output,
            requirement=user_input,
            score=score,
            json_text=json_text,
        )

        status = (
            f"Generation finished.\n"
            f"Repair count: {repair_count}\n"
            f"Fixed-template normalization: {'success' if score.get('fixed_template_normalized') else 'failed or not run'}\n"
            f"Normalization error: {score.get('normalize_error') or 'none'}\n"
            f"generated.json: {paths['json_path']}\n"
            f"score.json: {paths['score_path']}\n"
            f"Current score: {score.get('total_score_without_software')}/{score.get('max_score_without_software')}\n"\
            f"Reference comparison score: {score.get('reference_compare_score', 'no reference JSON')}\n"\
            f"Prerequisite score: {score.get('prerequisite_total_score_without_software', score.get('total_score_without_software'))}"
        )
        response = raw_output

    except Exception as e:
        traceback.print_exc()
        response = f"Call or save failed: {type(e).__name__}: {e}"
        status = response
        score = {"error": response}

    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response})

    return chat_history, "", status, score


def clear_history():
    return [], "", {}


# =========================
# 6. Batch evaluation
# =========================


def batch_evaluate(
    dataset_path: str,
    output_root: str,
    model_ids: Sequence[str],
    temperature: float,
    top_p: float,
    max_tokens: int,
    sleep_seconds: float,
    reference_json_root: str = "",          # New: reference JSON root directory (from the UI)
):
    """
    Batch evaluation logic:
    1. Generate one JSON per sample;
    2. Score every sample separately and record it in results.csv / results.jsonl;
    3. Finally aggregate mean scores by model + difficulty (simple/medium/complex);
    4. Invalid JSON, API failures, and scoring failures keep their records and count as 0 in the overall mean.

    reference_json_root: when provided, this directory is preferred for auto-matching reference JSONs.
    """
    if not dataset_path or not dataset_path.strip():
        return "Fill in the dataset path.", None

    if not output_root or not output_root.strip():
        return "Fill in the output path.", None

    if not model_ids:
        return "Select at least one model.", None

    # Convert the user-provided reference root into a Path (None for an empty string)
    ref_root = Path(reference_json_root) if reference_json_root.strip() else None

    try:
        cases = discover_cases(dataset_path, reference_root=ref_root)
    except Exception as e:
        return f"Failed to load the dataset: {type(e).__name__}: {e}", None

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root_path = ensure_dir(Path(output_root) / f"run_{run_id}")

    all_rows: List[Dict[str, Any]] = []
    results_jsonl_path = output_root_path / "results.jsonl"
    results_csv_path = output_root_path / "results.csv"
    summary_path = output_root_path / "summary.json"
    summary_csv_path = output_root_path / "summary.csv"

    total_tasks = len(model_ids) * len(cases)
    done = 0

    fieldnames = [
        "model",
        "case_id",
        "difficulty",

        # Status fields
        "api_success",
        "score_success",
        "json_valid_bool",
        "success",

        # Score fields
        "json_valid",
        "schema_score",
        "semantic_score",
        "total_score_without_software",
        "optimized_reference_score",
        "raw_reference_score",
        "max_score_without_software",
        "final_score",
        "reference_available",
        "reference_compare_score",
        "prerequisite_total_score_without_software",
        "prerequisite_rate",
        "type_count_score",
        "preorder_type_score",
        "hierarchy_score",
        "coordinate_score",
        "parameter_score",
        "node_count_score",
        "boolean_operation_score",
        "reference_error",

        # Structure and error information
        "node_count",
        "error_count",
        "errors",
        "repair_count",
        "fixed_template_normalized",
        "normalize_error",

        # Input and file paths
        "expected_types",
        "input_path",
        "reference_path",
        "json_path",
        "score_path",
        "score_detail_path",
        "initial_raw_output_path",
        "raw_output_path",
        "prompt_path",
        "error",
    ]

    with results_jsonl_path.open("w", encoding="utf-8") as jsonl_f:
        for model_id in model_ids:
            model_name = safe_name(model_id)

            for case in cases:
                done += 1
                case_id = case["id"]
                difficulty = case.get("difficulty", "unknown")
                requirement = case["requirement"]
                expected_types = case.get("expected_types", [])

                case_dir = output_root_path / model_name / difficulty / case_id
                ensure_dir(case_dir)

                row: Dict[str, Any] = {
                    "model": model_id,
                    "case_id": case_id,
                    "difficulty": difficulty,
                    "input_path": case.get("input_path", ""),
                    "reference_path": case.get("reference_path", ""),
                    "requirement": requirement,
                    "expected_types": ",".join(expected_types) if expected_types else "",

                    "api_success": False,
                    "score_success": False,
                    "json_valid_bool": False,
                    "success": False,

                    "json_valid": 0,
                    "schema_score": 0,
                    "semantic_score": 0,
                    "total_score_without_software": 0,
                    "optimized_reference_score": 0,
                    "raw_reference_score": 0,
                    "max_score_without_software": 100,
                    "final_score": 0,
                    "reference_available": False,
                    "reference_compare_score": 0,
                    "prerequisite_total_score_without_software": 0,
                    "prerequisite_rate": 0,
                    "type_count_score": 0,
                    "preorder_type_score": 0,
                    "hierarchy_score": 0,
                    "coordinate_score": 0,
                    "parameter_score": 0,
                    "node_count_score": 0,
                    "boolean_operation_score": 0,
                    "reference_error": "",

                    "node_count": 0,
                    "error_count": 0,
                    "errors": "",
                    "repair_count": 0,
                    "fixed_template_normalized": False,
                    "normalize_error": "",

                    "json_path": "",
                    "score_path": "",
                    "score_detail_path": "",
                    "initial_raw_output_path": "",
                    "raw_output_path": "",
                    "prompt_path": "",
                    "error": "",
                }

                api_success = False
                score_success = False

                try:
                    messages = build_model_messages(
                        requirement=requirement,
                        file_content="",
                        chat_history=None,
                        keep_history=False,
                    )
                    prompt_text = format_messages_for_log(messages)

                    raw_output, json_text, score, repair_count = generate_valid_cadjson(
                        messages=messages,
                        requirement=requirement,
                        expected_types=expected_types,
                        model_id=model_id,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=int(max_tokens),
                        repair_retries=2,
                        reference_path=case.get("reference_path", ""),
                    )

                    api_success = True
                    score_success = True

                    paths = save_generation_files(
                        base_dir=case_dir,
                        prompt=prompt_text,
                        raw_output=raw_output,
                        requirement=requirement,
                        score=score,
                        json_text=json_text,
                    )

                    errors = score.get("errors", [])
                    json_valid_bool = score.get("json_valid", 0) == 10

                    row.update({
                        "api_success": api_success,
                        "score_success": score_success,
                        "json_valid_bool": json_valid_bool,
                        "success": json_valid_bool,

                        "json_valid": score.get("json_valid", 0),
                        "schema_score": score.get("schema_score", 0),
                        "semantic_score": score.get("semantic_score", 0),
                        "total_score_without_software": score.get("total_score_without_software", 0),
                        "optimized_reference_score": score.get("optimized_reference_score", score.get("total_score_without_software", 0)),
                        "raw_reference_score": score.get("raw_reference_score", 0),
                        "max_score_without_software": score.get("max_score_without_software", 100),
                        "final_score": score.get("final_score", score.get("total_score_without_software", 0)),
                        "reference_available": score.get("reference_available", False),
                        "reference_compare_score": score.get("reference_compare_score", 0),
                        "prerequisite_total_score_without_software": score.get("prerequisite_total_score_without_software", score.get("total_score_without_software", 0)),
                        "prerequisite_rate": score.get("prerequisite_rate", 0),
                        "type_count_score": score.get("type_count_score", 0),
                        "preorder_type_score": score.get("preorder_type_score", 0),
                        "hierarchy_score": score.get("hierarchy_score", 0),
                        "coordinate_score": score.get("coordinate_score", 0),
                        "parameter_score": score.get("parameter_score", 0),
                        "node_count_score": score.get("node_count_score", 0),
                        "boolean_operation_score": score.get("boolean_operation_score", 0),
                        "reference_error": score.get("reference_error", ""),

                        "node_count": score.get("node_count", 0),
                        "error_count": len(errors),
                        "errors": " | ".join(str(x) for x in errors),
                        "repair_count": repair_count,
                        "fixed_template_normalized": score.get("fixed_template_normalized", False),
                        "normalize_error": score.get("normalize_error", ""),

                        "json_path": paths["json_path"],
                        "score_path": paths["score_path"],
                        "score_detail_path": paths.get("score_detail_path", ""),
                        "initial_raw_output_path": paths.get("initial_raw_output_path", ""),
                        "raw_output_path": paths["raw_output_path"],
                        "prompt_path": paths["prompt_path"],
                        "error": "",
                    })

                except Exception as e:
                    error_text = f"{type(e).__name__}: {e}"
                    traceback.print_exc()

                    row.update({
                        "api_success": api_success,
                        "score_success": score_success,
                        "json_valid_bool": False,
                        "success": False,

                        "json_valid": 0,
                        "schema_score": 0,
                        "semantic_score": 0,
                        "total_score_without_software": 0,
                        "optimized_reference_score": 0,
                        "raw_reference_score": 0,
                        "max_score_without_software": 100,
                        "final_score": 0,
                        "reference_available": False,
                        "reference_compare_score": 0,
                        "prerequisite_total_score_without_software": 0,
                        "prerequisite_rate": 0,
                        "type_count_score": 0,
                        "preorder_type_score": 0,
                        "hierarchy_score": 0,
                        "coordinate_score": 0,
                        "parameter_score": 0,
                        "node_count_score": 0,
                        "boolean_operation_score": 0,
                        "reference_error": error_text,

                        "node_count": 0,
                        "error_count": 1,
                        "errors": error_text,
                        "repair_count": 0,
                        "fixed_template_normalized": False,
                        "normalize_error": "",
                        "error": error_text,
                    })

                    (case_dir / "error.txt").write_text(error_text, encoding="utf-8")

                all_rows.append(row)
                jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                jsonl_f.flush()

                if sleep_seconds > 0:
                    time.sleep(float(sleep_seconds))

    with results_csv_path.open("w", encoding="utf-8-sig", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary = summarize_results(all_rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(summary, summary_csv_path)

    status = (
        f"Batch evaluation finished.\n"
        f"Samples: {len(cases)}\n"
        f"Models: {len(model_ids)}\n"
        f"Total tasks: {total_tasks}\n"
        f"Output directory: {output_root_path}\n"
        f"Per-sample scoring CSV: {results_csv_path}\n"
        f"Per-sample scoring JSONL: {results_jsonl_path}\n"
        f"Per-difficulty mean summary.json: {summary_path}\n"
        f"Per-difficulty mean summary.csv: {summary_csv_path}"
    )

    return status, str(results_csv_path)


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    The final summary.json keeps only two core scores:

    1. avg_optimized_reference_score：
       Mean comparison score between the optimized JSON and the original reference model JSON.

    2. avg_raw_reference_score：
       Mean comparison score between the model's first raw output and the original reference model JSON.

    Other detailed scores are kept in results.jsonl / results.csv / score_detail.json.
    """
    summary: Dict[str, Any] = {}

    for row in rows:
        model = row.get("model", "unknown")
        difficulty = row.get("difficulty", "unknown")

        summary.setdefault(model, {})
        summary[model].setdefault(difficulty, {
            "count": 0,
            "sum_optimized_reference_score": 0.0,
            "sum_raw_reference_score": 0.0,
        })

        item = summary[model][difficulty]
        item["count"] += 1

        optimized_score = float(
            row.get("optimized_reference_score")
            or row.get("total_score_without_software")
            or 0
        )
        raw_score = float(row.get("raw_reference_score") or 0)

        item["sum_optimized_reference_score"] += optimized_score
        item["sum_raw_reference_score"] += raw_score

    for model, diff_map in summary.items():
        for difficulty, item in diff_map.items():
            count = max(item["count"], 1)

            item["avg_optimized_reference_score"] = round(
                item["sum_optimized_reference_score"] / count,
                2,
            )
            item["avg_raw_reference_score"] = round(
                item["sum_raw_reference_score"] / count,
                2,
            )

            del item["sum_optimized_reference_score"]
            del item["sum_raw_reference_score"]

    return summary

def write_summary_csv(summary: Dict[str, Any], summary_csv_path: str | Path) -> None:
    """
    Export the simplified summary.json to summary.csv as well.
    Only the two final scores are shown.
    """
    summary_csv_path = Path(summary_csv_path)

    fieldnames = [
        "model",
        "difficulty",
        "count",
        "avg_optimized_reference_score",
        "avg_raw_reference_score",
    ]

    with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()

        for model, diff_map in summary.items():
            for difficulty, item in diff_map.items():
                row = {"model": model, "difficulty": difficulty}
                row.update(item)
                writer.writerow({k: row.get(k, "") for k in fieldnames})

# =========================
# 7. Gradio page
# =========================


def build_demo():
    if gr is None:
        raise ImportError("The gradio package is missing; run: pip install gradio")

    with gr.Blocks(title="Qwen / DashScope CAD-JSON Generation and Batch Evaluation Tool") as demo:
        gr.Markdown("# Qwen / DashScope CAD-JSON Generation and Batch Evaluation Tool")
        gr.Markdown(
            "Supports single-sample debugging, generated-JSON saving, automatic scoring, and simple/medium/complex dataset batch evaluation."
        )
        gr.Markdown(
            f"Current Base URL: `{DASHSCOPE_BASE_URL}`  \n"
            "Set the `DASHSCOPE_API_KEY` environment variable first. If the model you want to test is not in the list, enter the Model Studio model ID manually in the dropdown."
        )

        with gr.Tab("Single-sample debug"):
            chatbot = gr.Chatbot(label="Chat window", height=500,type="messages")

            user_input = gr.Textbox(
                label="Modeling requirement",
                placeholder="Example: generate a cuboid part with a circular hole, cuboid size 10x6x3, hole at the center with radius 1.",
                lines=4,
            )

            file_input = gr.File(
                label="Upload a file (optional)",
                file_types=[
                    ".txt", ".md", ".py", ".json", ".jsonl", ".csv",
                    ".log", ".xml", ".html", ".css", ".js",
                ],
                type="filepath",
            )

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=DEFAULT_MODELS,
                    value=DEFAULT_MODELS[0],
                    label="Select or enter the Qwen model ID",
                    allow_custom_value=True,
                )

            with gr.Row():
                temperature = gr.Number(value=0.1, label="temperature")
                top_p = gr.Number(value=0.8, label="top_p")
                max_tokens = gr.Number(value=8192, label="max_tokens", precision=0)

            expected_types_text = gr.Textbox(
                label="Optional: expected types, comma-separated",
                placeholder="Example: Group,Prism,Circle,Cut. Leave empty to infer roughly from the natural-language requirement.",
                lines=1,
            )

            save_dir = gr.Textbox(
                label="Root directory for single-sample results",
                value=r"outputs/descriptions",
                lines=1,
            )

            with gr.Row():
                submit_btn = gr.Button("Generate and score", variant="primary")
                clear_btn = gr.Button("Clear chat")

            status_box = gr.Textbox(label="Save and scoring status", lines=6)
            score_box = gr.JSON(label="Scoring result")

            submit_btn.click(
                fn=single_chat,
                inputs=[
                    user_input, file_input, chatbot, model_dropdown,
                    save_dir, expected_types_text, temperature, top_p, max_tokens,
                ],
                outputs=[chatbot, user_input, status_box, score_box],
            )

            user_input.submit(
                fn=single_chat,
                inputs=[
                    user_input, file_input, chatbot, model_dropdown,
                    save_dir, expected_types_text, temperature, top_p, max_tokens,
                ],
                outputs=[chatbot, user_input, status_box, score_box],
            )

            clear_btn.click(
                fn=clear_history,
                inputs=[],
                outputs=[chatbot, status_box, score_box],
            )

        with gr.Tab("Batch evaluation"):
            gr.Markdown(
                """
                The dataset supports two forms:

                1. Enter a `dataset.jsonl` path directly;
                2. Enter a dataset folder path; the folder contains:
                   - `simple/*.txt`
                   - `medium/*.txt`
                   - `complex/*.txt`

                The content of each txt/md file is one natural-language modeling requirement.
                """
            )

            dataset_path = gr.Textbox(
                label="Dataset path",
                placeholder=r"Example: data/cad_dataset or data/cad_dataset/dataset.jsonl",
                lines=1,
            )

            output_root = gr.Textbox(
                label="Batch output root directory",
                value=r"outputs/descriptions",
                lines=1,
            )

            # New: reference JSON root directory (former REFERENCE_JSON_ROOT)
            reference_root_input = gr.Textbox(
                label="Reference JSON root (for auto-matching the original JSON of each *_description.txt)",
                value=r"data/simple",
                placeholder=r"Example: data/simple. Leave empty to search only next to each description file",
                lines=1,
            )

            model_checkboxes = gr.CheckboxGroup(
                choices=DEFAULT_MODELS,
                value=[DEFAULT_MODELS[0]],
                label="Select models for batch testing. To add more models, extend DEFAULT_MODELS first.",
            )

            with gr.Row():
                batch_temperature = gr.Number(value=0.1, label="temperature")
                batch_top_p = gr.Number(value=0.8, label="top_p")
                batch_max_tokens = gr.Number(value=8192, label="max_tokens", precision=0)
                sleep_seconds = gr.Number(value=0.2, label="Seconds between requests")

            start_batch_btn = gr.Button("Start batch evaluation", variant="primary")
            batch_status = gr.Textbox(label="Batch evaluation status", lines=10)
            result_file = gr.File(label="results.csv")

            start_batch_btn.click(
                fn=batch_evaluate,
                inputs=[
                    dataset_path,
                    output_root,
                    model_checkboxes,
                    batch_temperature,
                    batch_top_p,
                    batch_max_tokens,
                    sleep_seconds,
                    reference_root_input,          # New: pass the reference root directory
                ],
                outputs=[batch_status, result_file],
            )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
