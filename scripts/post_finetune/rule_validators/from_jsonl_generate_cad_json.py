#Use an already trained/merged model to batch-generate CAD data in JSON format from an input prompt file.
#Load the specified model and tokenizer
#Read the prompt file (each line is JSON containing messages or prompt fields)
#Generate for each prompt (greedy decoding, up to 4096 input tokens + 2048 output tokens)
#Output a JSONL file with the generated text, truncation flag, source prompt line number, etc.

import argparse
import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_dtype(dtype: str):
    dtype = dtype.lower()
    if dtype in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if dtype in ["fp16", "float16"]:
        return torch.float16
    if dtype in ["fp32", "float32"]:
        return torch.float32
    if dtype == "auto":
        return "auto"
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_line_no"] = i
            rows.append(item)
    return rows


def build_prompt(tokenizer, item: Dict[str, Any]) -> str:
    if "messages" in item and isinstance(item["messages"], list):
        return tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )

    if "prompt" in item:
        return str(item["prompt"])

    raise ValueError(f"Each row must contain messages or prompt. Got keys: {item.keys()}")


def get_input_device(model):
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            device = emb.weight.device
            if device.type != "meta":
                return device
    except Exception:
        pass

    for _, p in model.named_parameters():
        if p.device.type != "meta":
            return p.device

    raise RuntimeError("Cannot find real device for model inputs.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--output_file", required=True)

    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    dtype = parse_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    rows = load_jsonl(args.prompts_file)

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(rows):
            prompt_text = build_prompt(tokenizer, item)

            full_inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=False,
            )
            full_prompt_len = full_inputs["input_ids"].shape[-1]
            was_truncated = full_prompt_len > args.max_input_tokens

            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            )

            input_device = get_input_device(model)
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            prompt_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            new_token_ids = generated[0][prompt_len:].detach().cpu().tolist()
            generated_text = tokenizer.decode(
                new_token_ids,
                skip_special_tokens=True,
            )

            out = {
                "index": idx,
                "line_no": item.get("_line_no"),
                "source_description": item.get("source_description"),
                "reference_json": item.get("reference_json"),
                "full_prompt_len": int(full_prompt_len),
                "prompt_len": int(prompt_len),
                "was_truncated": bool(was_truncated),
                "generated_text": generated_text,
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

            print(
                f"[{idx + 1}/{len(rows)}] "
                f"line={item.get('_line_no')} "
                f"prompt_len={prompt_len} "
                f"full_prompt_len={full_prompt_len} "
                f"truncated={was_truncated}"
            )

    print("saved to:", args.output_file)


if __name__ == "__main__":
    main()
