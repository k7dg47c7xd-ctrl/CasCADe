
# utils/finetuned_model_client.py
import os
import requests


def call_finetuned_model(
    prompt: str,
    *,
    expect_json: bool = False,
    max_tokens: int = 2048,
    timeout: int = 300,
) -> str:
    """
    Call the fine-tuned Qwen model exposed through an OpenAI-compatible API.
    The inference service must be running and must load the fine-tuned weights
    used in the paper.
    """
    base_url = os.environ.get(
        "CASCADE_API_BASE_URL",
        "http://127.0.0.1:8001/v1",
    ).rstrip("/")
    model = os.environ.get("CASCADE_MODEL")
    api_key = os.environ.get("CASCADE_API_KEY", "EMPTY")

    if not model:
        raise RuntimeError(
            "Set CASCADE_MODEL to the model ID actually exposed by the inference service"
        )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the fine-tuned CAD instructional assistant. "
                    "Follow the requested output format exactly. "
                    "Base all modeling explanations on the supplied CAD data. "
                    "Do not invent unsupported operations or parameters."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }

    if expect_json:
        payload["response_format"] = {"type": "json_object"}

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The fine-tuned model returned empty text")

    return content.strip()
