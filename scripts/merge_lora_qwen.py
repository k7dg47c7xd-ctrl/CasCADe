import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_path = os.getenv("CASCODE_BASE_MODEL", "saves/base/Qwen3.6-27B")
lora_checkpoint_path = os.getenv("CASCODE_LORA_CKPT", "saves/qwen3_6_27b/lora/sft_8654/checkpoint-1947-fixed")
merged_model_path = os.getenv("CASCODE_MERGED_MODEL", "saves/qwen3_6_27b/lora+qwen3_6_27b-fixed")

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    local_files_only=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
    attn_implementation="eager"
)

model = PeftModel.from_pretrained(
    base_model,
    lora_checkpoint_path,
    local_files_only=True
)

merged_model = model.merge_and_unload()

merged_model.save_pretrained(
    merged_model_path,
    safe_serialization=True,
    max_shard_size="4GB"
)

tokenizer.save_pretrained(merged_model_path)

print("Merge finished, saved to:", merged_model_path)