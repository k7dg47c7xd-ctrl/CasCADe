# -*- coding: utf-8 -*-
"""OpenAI-compatible client for the fine-tuned model used for teaching generation."""
from __future__ import annotations
import os
from openai import OpenAI

def call_finetuned_api(
    messages: list[dict[str, str]],
    *,
    model_id: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    timeout: float = 300.0,
) -> str:
    base_url = os.getenv("FINETUNED_API_BASE_URL", "http://127.0.0.1:8001/v1").strip().rstrip("/")
    api_key = os.getenv("FINETUNED_API_KEY", "EMPTY").strip() or "EMPTY"
    model_id = model_id or os.getenv("FINETUNED_MODEL", "").strip()
    if not model_id:
        raise RuntimeError(
            "FINETUNED_MODEL is not set. Set it to the model ID exposed by "
            "the inference server loading the paper's fine-tuned checkpoint."
        )
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Fine-tuned model returned empty content.")
    return content.strip()
