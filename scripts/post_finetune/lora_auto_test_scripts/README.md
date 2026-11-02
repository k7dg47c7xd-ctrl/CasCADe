# LoRA batch-inference pipeline: emit only the final normalized CAD-JSON

This directory runs batch inference with the fine-tuned model and normalizes the output into evaluable CAD-JSON:

```text
batch_lora_infer.py (calls the fine-tuned model)
  -> extracts the JSON from the model reply
  -> reorganizes it into tree-form JSON
  -> hands it to cadjson_quality_check.py
  -> emits only the final normalized JSON
```

## File roles

| File | Role |
|---|---|
| `batch_lora_infer.py` | Calls the fine-tuned model, extracts the reply JSON, reorganizes it into tree-form JSON, and automatically pipes it to `cadjson_quality_check.py` |
| `extract_json_from_chat_log.py` | Extracts tree-form JSON from an existing raw chat/text file; can be piped into `cadjson_quality_check.py` |
| `format_model_json.py` | Re-formats an existing JSON into tree-form JSON; can be piped into `cadjson_quality_check.py` |
| `cadjson_quality_check.py` | Performs detection and normalization only, and emits only the final normalized JSON |
| `debug_infer.py` | Single-sample debug inference (imports `batch_lora_infer` from this directory) |

## One-shot batch run (from the repository root)

```bash
cd scripts/post_finetune/lora_auto_test_scripts
python batch_lora_infer.py     --model_name_or_path <base-model-path>     --adapter_name_or_path <lora-adapter-path>     --prompts_path <prompts_all.jsonl>     --eval_jsonl     --output <normalized-output.jsonl>
```

- `--eval_jsonl`: emit one `{id, description, output_json, inference_error}` record per sample and keep failed samples, for geometric evaluation.
- Each output line is the final normalized CAD-JSON; error information is written only to stderr.

## Pipeline processing of existing model outputs

```bash
python extract_json_from_chat_log.py --input model_output.txt | python cadjson_quality_check.py --output normalized.json
python format_model_json.py --input model.json | python cadjson_quality_check.py --output normalized.json
```
