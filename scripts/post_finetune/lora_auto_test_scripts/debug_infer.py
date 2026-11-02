#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print the raw model output of the first N samples to locate [FAIL] root causes."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_lora_infer import (
    read_jsonl, get_messages, load_model_and_tokenizer,
    infer_batch, parse_json_from_text,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--adapter_name_or_path", default="")
    ap.add_argument("--prompts_path", required=True)
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--max_input_tokens", type=int, default=8192)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--torch_dtype", default="auto")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--merge_lora", action="store_true")
    args = ap.parse_args()

    records = read_jsonl(Path(args.prompts_path))[:args.limit]
    print(f"Read {len(records)} records", file=sys.stderr)
    model, tokenizer = load_model_and_tokenizer(args)

    for idx, rec in enumerate(records, 1):
        msgs = get_messages(rec)
        resp = infer_batch(model, tokenizer, [msgs], args)[0]
        print(f"\n==================== sample {idx} ====================")
        print(f"--- RAW RESPONSE ({len(resp)} chars) ---")
        print(resp)
        try:
            obj = parse_json_from_text(resp)
            print(f"--- EXTRACTED type={type(obj).__name__} ---")
            print(json.dumps(obj, ensure_ascii=False)[:2000])
        except Exception as e:
            print(f"--- EXTRACT FAILED: {e} ---")

if __name__ == "__main__":
    main()