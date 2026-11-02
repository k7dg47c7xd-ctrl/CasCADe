# -*- coding: utf-8 -*-
"""Parse all YAML configs and check key fields (no torch/OCC dependency)."""
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(rel):
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def test_train_lora_config():
    cfg = _load("configs/train_lora_custom/qwen3_6_27b_lora_sft.yaml")
    assert cfg["model_name_or_path"].startswith("saves/")
    assert cfg["stage"] == "sft"
    assert cfg["finetuning_type"] == "lora"
    assert cfg["quantization_bit"] == 4
    assert cfg["lora_rank"] == 8
    assert cfg["lora_alpha"] == 16
    assert cfg["learning_rate"] == 5.0e-5
    assert cfg["num_train_epochs"] == 1.0


def test_ablation_train_configs():
    for name in ["data_4k", "data_10k", "lora_rank16"]:
        cfg = _load(f"ablation/train_configs/{name}.yaml")
        assert "model_name_or_path" in cfg
        assert "output_dir" in cfg


def test_ablation_config():
    cfg = _load("ablation/config.yaml")
    assert set(cfg["test_splits"]) >= {"simple", "complex"}
    assert {"no_repair", "full_repair", "syntax_only_repair", "full_generic_feedback"} <= set(cfg["variants"])
    assert cfg["generation"]["max_repair_rounds"] == 3


def test_default_config_legacy():
    cfg = _load("configs/default.yaml")
    assert cfg["train"]["seed"] == 42
    assert cfg["lora"]["r"] == 8
    assert cfg["llm"]["api_key"] == "EMPTY"
