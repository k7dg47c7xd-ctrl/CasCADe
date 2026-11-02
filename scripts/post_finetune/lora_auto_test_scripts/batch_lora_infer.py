#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_lora_infer.py

Revised responsibilities:
1. Read prompts_all.jsonl;
2. Call the fine-tuned model / LoRA model;
3. Extract the JSON from the model reply;
4. Reorganize the extracted JSON into tree-form JSON;
5. Hand the tree-form JSON to cadjson_quality_check.py for checking and normalization immediately;
6. Emit only the normalized JSON / JSONL at the end.

In --eval_jsonl mode, keep every prediction record (including failures); optional --raw_output saves raw replies.
Logs and errors go to stderr only, to keep the final JSON clean.

Single-sample stdout:
  python batch_lora_infer.py ... --limit 1

Batch write to file:
  python batch_lora_infer.py ... --output normalized.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def strip_markdown_and_prefix(text: str) -> str:
    text = ANSI_RE.sub("", text or "")
    text = text.strip()
    for prefix in ["Assistant:", "assistant:", "ASSISTANT:", "模型输出：", "模型输出:", "输出：", "输出:"]:  # Chinese prefixes are kept for compatibility with legacy model logs
        if prefix in text:
            text = text.rsplit(prefix, 1)[1].strip()
    text = re.sub(r"```(?:json|JSON|text|plaintext)?", "", text)
    text = text.replace("```", "")
    return text.strip()


def extract_first_json(text: str) -> str:
    """Return only a *complete* root Group CAD-JSON, never an incidental fragment.

    `enable_thinking=False` is set in the chat template below, but this still
    defends against explanatory text, markdown and nested JSON objects.
    """
    text = strip_markdown_and_prefix(text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, end = decoder.raw_decode(text, match.start())
        except (ValueError, RecursionError):
            continue
        if isinstance(obj, dict) and obj.get("type") == "Group":
            return text[match.start():end]
    raise ValueError("No complete CAD-JSON with a type=Group root was found; the output may be truncated or the root node is wrong")


def parse_json_from_text(text: str) -> Any:
    json_text = extract_first_json(text)
    obj = json.loads(json_text)
    if isinstance(obj, str):
        obj = json.loads(obj)
    return obj


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path} line {line_no} is not valid JSONL: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path} line {line_no} must be a JSON object")
            records.append(obj)
    return records


def get_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    if isinstance(record.get("messages"), list):
        msgs: List[Dict[str, str]] = []
        for m in record["messages"]:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "user"))
            content = str(m.get("content", ""))
            if content:
                msgs.append({"role": role, "content": content})
        if msgs:
            return msgs

    if "instruction" in record:
        content = str(record.get("instruction", ""))
        if record.get("input"):
            content += "\n" + str(record.get("input"))
        return [{"role": "user", "content": content}]

    if "prompt" in record:
        return [{"role": "user", "content": str(record["prompt"])}]

    if "query" in record:
        return [{"role": "user", "content": str(record["query"])}]

    raise ValueError("The sample has no messages / instruction / prompt / query field; cannot build the input")


def build_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        # Qwen3: suppress hidden reasoning text that can consume the output budget.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    chunks = []
    for m in messages:
        chunks.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype: Any = "auto"
    if args.torch_dtype == "bf16":
        dtype = torch.bfloat16
    elif args.torch_dtype == "fp16":
        dtype = torch.float16
    elif args.torch_dtype == "fp32":
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    if args.adapter_name_or_path:
        if PeftModel is None:
            raise RuntimeError("peft is not installed. Run: pip install -U peft")
        model = PeftModel.from_pretrained(model, args.adapter_name_or_path)
        if args.merge_lora:
            model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def get_eos_token_ids(tokenizer) -> List[int]:
    ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in ids:
            ids.append(im_end_id)
    except Exception:
        pass
    return ids


def chunked(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def infer_batch(model, tokenizer, messages_batch: List[List[Dict[str, str]]], args) -> List[str]:
    prompt_texts = [build_prompt(tokenizer, messages) for messages in messages_batch]
    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": get_eos_token_ids(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
    else:
        gen_kwargs.update({"do_sample": False})

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return [r.strip() for r in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]


def checker_path_from_arg(path: str) -> Path:
    if path:
        return Path(path)
    return Path(__file__).resolve().parent / "cadjson_quality_check.py"


def normalize_by_checker(tree_obj: Any, checker_path: Path, indent: int = 0) -> str:
    tree_json = json.dumps(tree_obj, ensure_ascii=False, indent=2)
    cmd = [sys.executable, str(checker_path), "--indent", str(indent)]
    proc = subprocess.run(
        cmd,
        input=tree_json,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "cadjson_quality_check.py failed")
    return proc.stdout.strip()


def open_output(path: str):
    if not path:
        return None
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.open("w", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run LoRA inference, pass tree JSON to cadjson_quality_check.py, output only normalized JSON.")
    parser.add_argument("--model_name_or_path", required=True, help="Base model path or name")
    parser.add_argument("--adapter_name_or_path", default="", help="LoRA adapter path; leave empty when the LoRA is already merged")
    parser.add_argument("--prompts_path", required=True, help="Path to prompts_all.jsonl")
    parser.add_argument("--output", default="", help="Final normalized JSONL output file; stdout when omitted")
    parser.add_argument("--checker_path", default="", help="Path to cadjson_quality_check.py; defaults to the copy next to this script")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--torch_dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--merge_lora", action="store_true")
    parser.add_argument("--start", type=int, default=1, help="1-based index of the first sample to process")
    parser.add_argument("--limit", type=int, default=0, help="How many samples to process; 0 means all")
    parser.add_argument("--indent", type=int, default=0, help="Final output indentation passed to cadjson_quality_check.py; 0 recommended for batches")
    parser.add_argument("--eval_jsonl", action="store_true", help="Emit one {id,description,output_json,inference_error} record per sample and keep failed samples, for geometric evaluation")
    parser.add_argument("--raw_output", default="", help="Optional: save raw model replies per sample as JSONL for troubleshooting truncation and similar failures")
    args = parser.parse_args(argv)

    prompts_path = Path(args.prompts_path)
    checker_path = checker_path_from_arg(args.checker_path)
    if not checker_path.exists():
        raise FileNotFoundError(f"cadjson_quality_check.py not found: {checker_path}")

    records = read_jsonl(prompts_path)
    indexed_records = list(enumerate(records, start=1))
    indexed_records = [(i, r) for i, r in indexed_records if i >= args.start]
    if args.limit and args.limit > 0:
        indexed_records = indexed_records[:args.limit]

    if not indexed_records:
        log("No samples to process.")
        return 0

    log(f"Samples read: {len(records)}; processed in this run: {len(indexed_records)}")
    model, tokenizer = load_model_and_tokenizer(args)

    out_f = open_output(args.output)
    raw_f = open_output(args.raw_output)
    success = 0
    failed = 0

    def emit_raw_record(record, idx, response=None, error=None):
        if raw_f is None:
            return
        ref = str(record.get("reference_json", ""))
        match = re.search(r"([0-9]{8})_lite\.json$", ref)
        item = {"index": idx, "id": match.group(1) if match else None,
                "raw_response": response, "error": error}
        raw_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        raw_f.flush()

    def emit_eval_record(record, output_json=None, inference_error=None):
        ref = str(record.get("reference_json", ""))
        match = re.search(r"([0-9]{8})_lite\.json$", ref)
        if not match:
            raise ValueError(f"The input sample has no extractable reference_json ID: {ref!r}")
        msg = next((m.get("content", "") for m in reversed(record.get("messages", []))
                    if isinstance(m, dict) and m.get("role") == "user"), "")
        item = {"id": match.group(1), "description": msg,
                "output_json": output_json, "inference_error": inference_error}
        line = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if out_f is not None:
            out_f.write(line + "\n")
            out_f.flush()
        else:
            print(line, flush=True)

    try:
        for batch in chunked(indexed_records, args.batch_size):
            batch_indices: List[int] = []
            batch_records: List[Dict[str, Any]] = []
            batch_messages: List[List[Dict[str, str]]] = []

            for idx, record in batch:
                try:
                    batch_messages.append(get_messages(record))
                    batch_indices.append(idx)
                    batch_records.append(record)
                except Exception as exc:
                    failed += 1
                    emit_raw_record(record, idx, error=f"input_failed: {exc}")
                    if args.eval_jsonl:
                        emit_eval_record(record, inference_error=f"input_failed: {exc}")
                    log(f"[FAIL] {idx}: failed to build the input: {exc}")

            if not batch_messages:
                continue

            try:
                responses = infer_batch(model, tokenizer, batch_messages, args)
            except Exception as exc:
                failed += len(batch_records)
                for idx, record in zip(batch_indices, batch_records):
                    emit_raw_record(record, idx, error=f"batch_infer_failed: {exc}")
                    if args.eval_jsonl:
                        emit_eval_record(record, inference_error=f"batch_infer_failed: {exc}")
                    log(f"[FAIL] {idx}: model inference failed: {exc}")
                continue
            for idx, _record, response in zip(batch_indices, batch_records, responses):
                emit_raw_record(_record, idx, response=response)
                try:
                    tree_obj = parse_json_from_text(response)
                    normalized_json = normalize_by_checker(tree_obj, checker_path=checker_path, indent=args.indent)
                    if args.eval_jsonl:
                        emit_eval_record(_record, output_json=json.loads(normalized_json))
                    elif out_f is not None:
                        out_f.write(normalized_json.replace("\n", "") if args.indent == 0 else normalized_json)
                        out_f.write("\n")
                    else:
                        print(normalized_json, flush=True)
                    success += 1
                    log(f"[OK] {idx}/{len(records)}")
                except Exception as exc:
                    failed += 1
                    if args.eval_jsonl:
                        emit_eval_record(_record, inference_error=f"normalize_failed: {exc}")
                    log(f"[FAIL] {idx}: {exc}")

    finally:
        if out_f is not None:
            out_f.close()
        if raw_f is not None:
            raw_f.close()

    log(f"Finished: {success} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
