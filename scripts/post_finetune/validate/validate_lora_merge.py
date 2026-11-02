#Check whether the tokenizer is identical before and after merging (special tokens, chat_template, etc.)
#Compute next-token logit cosine similarity and max absolute difference item by item
#Compare greedy-generated token sequences and final text
#Write a detailed comparison report (compare_report.jsonl) and summary statistics (summary.json), and give the final verdict (pass/fail).

#Purpose: compare the output consistency of the model loaded in two ways on identical inputs:
#Dynamic loading: base model + LoRA adapter (PeftModel)
#Static loading: the fully merged model with LoRA baked in

import argparse
import gc
import hashlib
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_dtype(dtype: str):
    dtype = dtype.lower()
    if dtype == "auto":
        return "auto"
    if dtype in ["fp16", "float16"]:
        return torch.float16
    if dtype in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if dtype in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def md5_text(text: Optional[str]) -> str:
    if text is None:
        return "NONE"
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def safe_getattr(obj: Any, name: str, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                if isinstance(item, str):
                    item = {"prompt": item}
                elif not isinstance(item, dict):
                    item = {"prompt": str(item)}
            except json.JSONDecodeError:
                item = {"prompt": line}

            item["_line_no"] = i
            rows.append(item)

    if not rows:
        raise ValueError(f"No prompts found in {path}")

    return rows


def load_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        # Only for batch/padding safety; does not change the model weights.
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def tokenizer_snapshot(tokenizer) -> Dict[str, Any]:
    return {
        "vocab_size": len(tokenizer),
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
        "chat_template_md5": md5_text(safe_getattr(tokenizer, "chat_template", None)),
    }


def compare_tokenizers(base_model: str, merged_model: str) -> Dict[str, Any]:
    base_tok = load_tokenizer(base_model)
    merged_tok = load_tokenizer(merged_model)

    base_snap = tokenizer_snapshot(base_tok)
    merged_snap = tokenizer_snapshot(merged_tok)

    diffs = {}
    for k in base_snap:
        if base_snap[k] != merged_snap[k]:
            diffs[k] = {
                "base": base_snap[k],
                "merged": merged_snap[k],
            }

    return {
        "base": base_snap,
        "merged": merged_snap,
        "diffs": diffs,
        "passed": len(diffs) == 0,
    }


def build_prompt_text(tokenizer, item: Dict[str, Any]) -> str:
    """
    Two input forms are supported:
    1. {"messages": [{"role": "user", "content": "..."}]}
    2. {"prompt": "..."}
    """
    if "messages" in item and isinstance(item["messages"], list):
        messages = item["messages"]

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback when no chat_template is present
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    if "prompt" in item:
        return str(item["prompt"])

    raise ValueError(f"Prompt item must contain 'messages' or 'prompt': {item}")


def parse_device_map_arg(device_map):
    """
    argparse only yields strings; convert common single-device spellings into the device_map transformers expects.
    Supported: auto / balanced / balanced_low_0 / sequential / cuda:0 / cpu / none.
    """
    if device_map is None:
        return None

    s = str(device_map).strip()
    lower = s.lower()

    if lower in {"none", "null", "no", "false"}:
        return None

    if s in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return s

    if lower in {"cuda", "gpu"}:
        return {"": "cuda:0"}

    if lower.startswith("cuda:") or lower in {"cpu", "mps"}:
        return {"": lower}

    # Fallback: keep the original value and let transformers handle it.
    return s


def get_input_device(model) -> torch.device:
    """
    Used to place the tokenizer output.
    Cannot simply use next(model.parameters()).device, because under device_map=auto / PEFT / offload
    the first parameter may be a meta tensor. input_ids and attention_mask must go to a real device.
    """
    candidates = [model]

    for obj in list(candidates):
        for attr in ("base_model", "model"):
            child = getattr(obj, attr, None)
            if child is not None and child not in candidates:
                candidates.append(child)

    # Prefer the device of the input embeddings.
    for obj in candidates:
        try:
            emb = obj.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                device = emb.weight.device
                if device.type != "meta":
                    return device
        except Exception:
            pass

    # Fallback: find the first real parameter device.
    for _, p in model.named_parameters():
        if p.device.type != "meta":
            return p.device

    raise RuntimeError(
        "Model parameters are still all on the meta device, meaning the model was not actually loaded onto CPU/GPU."
        "Check device_map, accelerate, and whether VRAM/RAM are sufficient."
    )


def print_device_diagnostics(model, name: str) -> None:
    real_devices = set()
    meta_examples = []
    meta_count = 0

    for n, p in model.named_parameters():
        if p.device.type == "meta":
            meta_count += 1
            if len(meta_examples) < 10:
                meta_examples.append(n)
        else:
            real_devices.add(str(p.device))

    try:
        input_device = get_input_device(model)
    except Exception as e:
        input_device = f"ERROR: {e}"

    print(f"[{name}] input device: {input_device}")
    print(f"[{name}] real parameter devices: {sorted(real_devices)}")
    print(f"[{name}] meta parameter count: {meta_count}")
    if meta_examples:
        print(f"[{name}] meta parameter examples: {meta_examples}")


def load_lora_model(
    base_model: str,
    lora_adapter: str,
    dtype,
    device_map,
    attn_implementation: str,
):
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    model = PeftModel.from_pretrained(
        base,
        lora_adapter,
        torch_dtype=dtype,
        is_trainable=False,
        low_cpu_mem_usage=False,
    )
    model.eval()
    print_device_diagnostics(model, "lora")
    return model


def load_merged_model(
    merged_model: str,
    dtype,
    device_map,
    attn_implementation: str,
):
    model = AutoModelForCausalLM.from_pretrained(
        merged_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    model.eval()
    print_device_diagnostics(model, "merged")
    return model


@torch.inference_mode()
def run_model_once(
    model,
    tokenizer,
    prompt_text: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )

    device = get_input_device(model)
    if device.type == "meta":
        raise RuntimeError("inputs cannot be moved to the meta device; check model loading.")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    prompt_len =  (
        inputs["input_ids"].shape)[-1]

    # 1. next-token logits: for numeric-level comparison
    outputs = model(**inputs)
    next_logits = outputs.logits[0, -1, :].float().detach().cpu()

    top1_id = int(torch.argmax(next_logits).item())
    top1_token = tokenizer.decode([top1_id], skip_special_tokens=False)

    # 2. greedy generation: for output-level comparison
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        return_dict_in_generate=True,
        output_scores=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    sequence = generated.sequences[0].detach().cpu()
    new_token_ids = sequence[prompt_len:].tolist()
    decoded_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    return {
        "prompt_len": int(prompt_len),
        "next_logits": next_logits.numpy().astype(np.float32),
        "top1_id": top1_id,
        "top1_token": top1_token,
        "generated_token_ids": new_token_ids,
        "generated_text": decoded_text,
    }


def run_model_set(
    kind: str,
    base_model: str,
    lora_adapter: str,
    merged_model: str,
    tokenizer_path: str,
    prompts: List[Dict[str, Any]],
    dtype,
    device_map,
    attn_implementation: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    tokenizer = load_tokenizer(tokenizer_path)

    if kind == "lora":
        model = load_lora_model(base_model, lora_adapter, dtype, device_map, attn_implementation)
    elif kind == "merged":
        model = load_merged_model(merged_model, dtype, device_map, attn_implementation)
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    results = []

    for idx, item in enumerate(prompts):
        prompt_text = build_prompt_text(tokenizer, item)

        result = run_model_once(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )

        results.append({
            "index": idx,
            "line_no": item.get("_line_no"),
            "prompt_text": prompt_text,
            "prompt_len": result["prompt_len"],
            "top1_id": result["top1_id"],
            "top1_token": result["top1_token"],
            "generated_token_ids": result["generated_token_ids"],
            "generated_text": result["generated_text"],
            "next_logits": result["next_logits"],
        })

        print(f"[{kind}] finished {idx + 1}/{len(prompts)}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    ta = torch.tensor(a, dtype=torch.float32)
    tb = torch.tensor(b, dtype=torch.float32)
    return float(F.cosine_similarity(ta, tb, dim=0).item())


def compare_results(
    lora_results: List[Dict[str, Any]],
    merged_results: List[Dict[str, Any]],
    tolerance: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    reports = []

    for a, b in zip(lora_results, merged_results):
        logits_a = a["next_logits"]
        logits_b = b["next_logits"]

        diff = np.abs(logits_a - logits_b)

        gen_ids_equal = a["generated_token_ids"] == b["generated_token_ids"]
        text_equal = a["generated_text"] == b["generated_text"]
        top1_equal = a["top1_id"] == b["top1_id"]

        max_abs_diff = float(np.max(diff))
        mean_abs_diff = float(np.mean(diff))
        cosine = cosine_similarity_np(logits_a, logits_b)

        passed = (
            top1_equal
            and max_abs_diff <= tolerance
            and cosine >= 0.99999
        )

        reports.append({
            "index": a["index"],
            "line_no": a["line_no"],
            "prompt_len": a["prompt_len"],
            "top1_equal": top1_equal,
            "lora_top1_id": a["top1_id"],
            "merged_top1_id": b["top1_id"],
            "lora_top1_token": a["top1_token"],
            "merged_top1_token": b["top1_token"],
            "generated_token_ids_equal": gen_ids_equal,
            "generated_text_equal": text_equal,
            "max_abs_logit_diff": max_abs_diff,
            "mean_abs_logit_diff": mean_abs_diff,
            "logit_cosine_similarity": cosine,
            "passed": passed,
            "prompt_text": a["prompt_text"],
            "lora_generated_text": a["generated_text"],
            "merged_generated_text": b["generated_text"],
        })

    total = len(reports)
    passed_count = sum(1 for r in reports if r["passed"])
    gen_token_equal_count = sum(1 for r in reports if r["generated_token_ids_equal"])
    text_equal_count = sum(1 for r in reports if r["generated_text_equal"])
    top1_equal_count = sum(1 for r in reports if r["top1_equal"])

    max_diffs = [r["max_abs_logit_diff"] for r in reports]
    mean_diffs = [r["mean_abs_logit_diff"] for r in reports]
    cosines = [r["logit_cosine_similarity"] for r in reports]

    summary = {
        "total": total,
        "passed_count": passed_count,
        "passed_rate": passed_count / total if total else 0.0,
        "top1_equal_count": top1_equal_count,
        "top1_equal_rate": top1_equal_count / total if total else 0.0,
        "generated_token_ids_equal_count": gen_token_equal_count,
        "generated_token_ids_equal_rate": gen_token_equal_count / total if total else 0.0,
        "generated_text_equal_count": text_equal_count,
        "generated_text_equal_rate": text_equal_count / total if total else 0.0,
        "max_of_max_abs_logit_diff": float(np.max(max_diffs)) if max_diffs else None,
        "mean_of_max_abs_logit_diff": float(np.mean(max_diffs)) if max_diffs else None,
        "mean_of_mean_abs_logit_diff": float(np.mean(mean_diffs)) if mean_diffs else None,
        "min_logit_cosine_similarity": float(np.min(cosines)) if cosines else None,
        "mean_logit_cosine_similarity": float(np.mean(cosines)) if cosines else None,
    }

    return reports, summary


def json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.float64):
        return float(obj)
    if isinstance(obj, np.int64):
        return int(obj)
    return obj


def save_jsonl_without_logits(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            row = dict(row)
            row.pop("next_logits", None)
            f.write(json.dumps(row, ensure_ascii=False, default=json_safe) + "\n")


def save_report(path: str, reports: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in reports:
            f.write(json.dumps(row, ensure_ascii=False, default=json_safe) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", required=True, help="base model path")
    parser.add_argument("--lora_adapter", required=True, help="LoRA adapter path")
    parser.add_argument("--merged_model", required=True, help="merged model path")
    parser.add_argument("--prompts_file", required=True, help="prompts jsonl path")

    parser.add_argument("--tokenizer_path", default=None, help="Defaults to the merged model's tokenizer")
    parser.add_argument("--output_dir", default="./merge_validation_report")

    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_input_tokens", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    # Logit differences under bf16/fp16 usually stay small. For fp32, this can be tightened to 1e-4 or 1e-5.
    parser.add_argument("--tolerance", type=float, default=1e-2)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    device_map = parse_device_map_arg(args.device_map)

    tokenizer_path = args.tokenizer_path or args.merged_model

    print("========== Tokenizer Check ==========")
    tok_report = compare_tokenizers(args.base_model, args.merged_model)
    print(json.dumps(tok_report, ensure_ascii=False, indent=2))

    with open(os.path.join(args.output_dir, "tokenizer_check.json"), "w", encoding="utf-8") as f:
        json.dump(tok_report, f, ensure_ascii=False, indent=2)

    prompts = load_jsonl(args.prompts_file)

    print("\n========== Run base + LoRA adapter ==========")
    lora_results = run_model_set(
        kind="lora",
        base_model=args.base_model,
        lora_adapter=args.lora_adapter,
        merged_model=args.merged_model,
        tokenizer_path=tokenizer_path,
        prompts=prompts,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )

    save_jsonl_without_logits(
        os.path.join(args.output_dir, "lora_outputs.jsonl"),
        lora_results,
    )

    print("\n========== Run merged model ==========")
    merged_results = run_model_set(
        kind="merged",
        base_model=args.base_model,
        lora_adapter=args.lora_adapter,
        merged_model=args.merged_model,
        tokenizer_path=tokenizer_path,
        prompts=prompts,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )

    save_jsonl_without_logits(
        os.path.join(args.output_dir, "merged_outputs.jsonl"),
        merged_results,
    )

    print("\n========== Compare ==========")
    reports, summary = compare_results(
        lora_results=lora_results,
        merged_results=merged_results,
        tolerance=args.tolerance,
    )

    save_report(
        os.path.join(args.output_dir, "compare_report.jsonl"),
        reports,
    )

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n========== Final Judgment ==========")

    if tok_report["passed"]:
        print("[OK] tokenizers are essentially identical.")
    else:
        print("[WARN] tokenizer differs; check tokenizer_config.json, special_tokens_map.json, and chat_template.")

    if summary["top1_equal_rate"] == 1.0 and summary["generated_text_equal_rate"] == 1.0:
        print("[OK] The dynamically loaded LoRA model and the merged model produce identical greedy output.")
    elif summary["top1_equal_rate"] == 1.0 and summary["min_logit_cosine_similarity"] >= 0.99999:
        print("[OK] logits are highly consistent. Minor text differences may come from floating-point precision or nearly tied probabilities.")
    else:
        print("[FAIL] The merged model differs substantially from the dynamically loaded LoRA model; check the merge/export pipeline.")

    print(f"\nReport directory: {args.output_dir}")


if __name__ == "__main__":
    main()
