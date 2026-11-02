# Experiment Notes — Paper-Section ↔ Code ↔ Command ↔ Output Mapping

This file maps the paper sections of CasCADe to the code entry points, configuration files, and outputs shipped in this repository. It also records packaging decisions and open verification items.

> Earlier drafts of this note incorrectly stated that the instructional-reconstruction (Section 4.4) code was excluded from this repository. **The repository includes the instructional-reconstruction scripts** under `scripts/post_finetune/`.

## Section ↔ code mapping

| Paper section / task | Code entry point | Key configuration | Outputs |
| --- | --- | --- | --- |
| Data annotation (3.1) | `scripts/make_train_json.py`, `scripts/make_val_json.py`, `scripts/make_prompts_jsonl.py`, `scripts/pre_finetune/generate_description_qwen.py`, `scripts/generation_description_qwen3_7_api.py` | `configs/`; environment variables documented in the script headers | `configs/cad_train.json`, `configs/val.json`, `configs/prompts.jsonl`, generated `*_description.txt` files |
| Fine-tuning method (3.2) | LLaMA-Factory CLI | `configs/train_lora_custom/qwen3_6_27b_lora_sft.yaml`, `configs/llamafactory/dataset_info.json` | `saves/qwen3_6_27b/lora/*` (not bundled) |
| Validation and error correction (3.3) | `ablation/generate.py` → `ablation/validation.py` / `ablation/schema_check.py` / `ablation/geometry_check.py` / `ablation/repair.py` | `ablation/config.yaml` | `ablation/outputs/<difficulty>/<variant>/rows.jsonl` and `summary.json` |
| Instructional reconstruction (3.4 / 4.4) | `scripts/post_finetune/split_cad_json_dataset.py`, `scripts/post_finetune/generate_teaching_description_ollama.py`, `scripts/post_finetune/split_cad_json_dataset_ollama_html_steps.py` | `INPUT_DIR` / `OUTPUT_DIR` constants in the split script; `TEACHING_MODEL` environment variable; `--ollama_url` / `--model` flags | per-model step JSONs, teaching text and HTML, manifest files |
| Base-model screening (4.2.1) | `scripts/pre_finetune/llm-selection/chat_qwen/chat_modified_qwen.py` plus `Rating_modified.py` / `Rating_raw_structure.py` | `DASHSCOPE_BASE_URL`, `REFERENCE_JSON_ROOT`, `DASHSCOPE_API_KEY` (or `QWEN_API_KEY`) | `scripts/pre_finetune/llm-selection/*/save_output/` |
| Repair-mechanism ablations (4.2.3) | `ablation/run_ablation.py --group repair|trigger|feedback`, `ablation/run_all.sh` | `ablation/config.yaml` variant definitions | `ablation/outputs/<difficulty>/<variant>/summary.json` |
| Training-data size and LoRA rank (4.2.4) | `ablation/train_configs/data_4k.yaml`, `data_10k.yaml`, `lora_rank16.yaml` | LLaMA-Factory CLI | `saves/qwen3_6_27b/lora/*` (not bundled) |
| Overall comparison (4.2.5) | `ablation/main_eval/run_main.py`, `ablation/main_eval/summarize_main.py` | in-script constants; environment overrides `CASCODE_DATASET` / `CASCODE_BASE_MODEL` / `CASCODE_MERGED_MODEL` | `ablation/main_eval/outputs/main_results/` |

## Data conventions

- The final training pipeline uses **English** descriptions with English instruction templates; see `scripts/make_train_json.py` and `scripts/make_val_json.py`.
- `scripts/pre_finetune/generate_description.py` (the Chinese-description generator) is intentionally **not** part of this release package because it is not used by the final English training pipeline. The retained generators are `scripts/pre_finetune/generate_description_qwen.py` (Cadmium-style English descriptions) and `scripts/generation_description_qwen3_7_api.py` (strict LLM-only English descriptions with optional rendered views).
- The 146-sample evaluation set referenced by the original project is produced through the batch-inference workflow (`scripts/post_finetune/lora_auto_test_scripts/`); the split files and prompt files are not bundled. Train/eval overlap-filtering records remain in the original project directory.

## Relationship to the original project

This directory is a packaging subset of the original project directory. The original directory was not modified. Items excluded from this release:

| Excluded item | Reason |
| --- | --- |
| `scripts/chat_gemma_timing.py` | LLM API timing utility unrelated to the release scope |
| `scripts/post_finetune/lora_auto_test_scripts/run_all.sh` | stale example script with machine-specific paths; the supported command is documented in this repository's README |
| `third_party/LLaMA-Factory/`, datasets, weights, outputs | third-party repository and large artifacts; excluded via `.gitignore` |
| `eval/geometry_metrics.py` | geometric-structure evaluation (Section 4.2.2 geometric metrics), intentionally excluded from this subset |
| `cad_compare_starter/` | cross-model comparison (Section 4.3, CADmium-7B), intentionally excluded from this subset |
| `cad-chat-demo-java-qwen-integrated/` | external demo project; its backend contains its own copies of some similarly named scripts |

## Packaged instructional-reconstruction scripts

- `scripts/post_finetune/split_cad_json_dataset.py` — rule-based modeling-tree splitting. It is referenced by the documented workflow and imported by `generate_teaching_description_ollama.py` (which falls back to an internal simplified split when the module is not importable).
- `scripts/post_finetune/generate_teaching_description_ollama.py` — LLM-assisted per-step teaching text with placeholder replacement and numeric backfill.
- `scripts/post_finetune/split_cad_json_dataset_ollama_html_steps.py` — integrated split + explanation + HTML persistence pipeline.

A separate implementation of the splitting workflow exists in the Web demo backend under the same file name. The two copies are **not identical** and are kept independently; do not assume equivalence and do not merge them without comparing contents and call sites. Neither the demo application nor a running inference service is distributed with this repository.

## Open verification items (resolve before public release)

1. **Repair-round setting.** `ablation/config.yaml` ships `generation.max_repair_rounds: 2`, `ablation/generate.py` defaults to 2, and `ablation/main_eval/run_main.py` passes `--max_repair_rounds 2`, while the paper describes up to 3 repair rounds. Confirm which value was actually used for the reported experiments and align config, code, and paper.
2. **Offline test inconsistency.** `tests/test_configs.py::test_ablation_train_configs` expects `ablation/train_configs/data_2164.yaml`, `data_4327.yaml`, and `data_8654.yaml`, but the release ships `data_4k.yaml`, `data_10k.yaml`, and `lora_rank16.yaml`. Update the test or the configs. This is currently the only failing offline check in the snapshot.
3. **Shared initial outputs.** `ablation/config.yaml` sets `reuse_initial_outputs: true` and `ablation/run_ablation.py` passes `--initial_cache`, but `ablation/generate.py` does not define that flag. Wire up the cache implementation or set `reuse_initial_outputs: false` before running the trigger/feedback ablations.
4. **Uniform final evaluation.** Confirm that final per-variant metrics are computed with a uniform three-stage validation independent of the repair-trigger setting, as required by the paper's ablation protocol.
5. **End-to-end verification.** No end-to-end runs (fine-tuning, inference, repair ablations, teaching generation) were executed for this release snapshot. Only offline syntax and configuration checks were run. Re-run every pipeline on the target environment before publishing, and re-verify all paper-reported numbers.
6. **In-file Chinese documentation.** Several bundled files remain in Chinese and were not rewritten in this pass: `ablation/README.md`, `ablation/README_REPAIR.md`, `third_party/README.md`, `data/README.md`, `configs/llamafactory/README.md`, `scripts/post_finetune/lora_auto_test_scripts/README.md`. Only the root `README.md` and this file were converted to English.
7. **Citation and license.** The citation is a placeholder and no license file is included. Finalize both before release.
