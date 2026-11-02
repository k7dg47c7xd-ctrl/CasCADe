# CasCADe: Fine-Tuning, Validation, Repair, and Instructional Reconstruction for Natural-Language CAD Generation

CasCADe is a research codebase for generating hierarchical CAD-JSON modeling sequences from natural-language modeling requirements. It provides experimental pipelines for parameter-efficient fine-tuning, structured output validation, iterative repair, CAD evaluation, and instructional reconstruction.

## Overview

The repository includes the following components:

- **Data preparation:** DeepCAD-derived data layout, English natural-language description generation, and Alpaca-format training, validation, and prompt files.
- **QLoRA fine-tuning:** LLaMA-Factory configuration and LoRA merging for Qwen3.6-27B.
- **Inference and evaluation:** Batch inference, JSON normalization, and output quality checks.
- **Validation and iterative repair:** Reference-free syntax, schema, and geometry validation, together with an error-aware repair loop.
- **Ablation studies:** Experiments on repair mechanisms, repair triggers, feedback strategies, training data size, LoRA rank, and overall model performance.
- **Instructional reconstruction:** Rule-based decomposition of CAD modeling trees into teachable steps, LLM-assisted step explanations, and an integrated HTML generation pipeline.
- **Base-model screening:** Batch chat and rating tools for the Qwen/DashScope channel.

The overall workflow consists of preparing CAD-JSON data and natural-language descriptions, fine-tuning a compatible language model, generating structured outputs, validating and repairing generated sequences, and evaluating the resulting outputs.

> **Scope and availability**
>
> This repository includes instructional-reconstruction scripts in addition to fine-tuning and evaluation tools.
>
> The companion Web demo project, `cad-chat-demo-java-qwen-integrated`, is maintained separately and is not part of this repository. Its backend contains separate implementations of some similarly named scripts, including `split_cad_json_dataset_ollama_html_steps.py` and `finetuned_model_client.py`. Files with identical names should not be assumed to have identical implementations.
>
> This repository does not bundle datasets, pretrained model weights, LoRA checkpoints, or external inference services. The presence of a script does not guarantee that the corresponding end-to-end workflow has been tested in every environment.

## Quick Start

### 1. Create an environment

The original experiments were conducted in a Linux GPU environment with Python 3.10. Start by creating a compatible environment:

```bash
conda create -n cascade python=3.10 -y
conda activate cascade
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
```

Additional dependencies may be required for GPU fine-tuning, geometry validation, and external inference services. See [Environment and Installation](#environment-and-installation).

### 3. Run the offline test suite

From the repository root:

```bash
python -m pytest tests/ -q
```

The current checked snapshot passed 18 offline tests. These tests do not establish that GPU fine-tuning, external API calls, or end-to-end CAD reconstruction will work in every environment.

### 4. Prepare data and model checkpoints

Follow [Data Preparation](#data-preparation) and `data/README.md` to prepare the required CAD-JSON data and natural-language descriptions.

Configure the local base-model path, dataset paths, and any required inference services before running training or inference.

## Repository Structure

```text
CasCADe_github/
├── ablation/                   # Validation, repair, scoring, and ablations
│   ├── config.yaml             # Ablation configuration
│   ├── run_ablation.py         # Repair, trigger, and feedback experiments
│   ├── run_all.sh              # Shell wrapper for repair ablations
│   ├── generate.py             # Generation, repair, and evaluation
│   ├── repair.py               # Iterative repair loop
│   ├── validation.py           # Syntax, schema, and geometry validation
│   ├── schema_check.py         # Reference-free schema validation
│   ├── geometry_check.py       # OCC-based geometry checks
│   ├── scoring.py              # CAD reference scoring
│   ├── analyze.py              # Per-variant result aggregation
│   ├── summarize_all.py        # Cross-variant summary
│   ├── build_dataset.py        # Test-split construction
│   ├── merge_lora.py           # LoRA merge helper
│   ├── main_eval/              # Overall model comparisons
│   └── train_configs/          # Data-size and LoRA-rank configurations
├── configs/
│   ├── default.yaml            # Legacy Qwen2.5-3B LoRA configuration
│   ├── train_lora_custom/      # Main QLoRA training configuration
│   └── llamafactory/           # LLaMA-Factory dataset registration
├── scripts/
│   ├── make_prompts_jsonl.py
│   ├── make_train_json.py
│   ├── make_val_json.py
│   ├── merge_lora_qwen.py
│   ├── build_batch_config.py
│   ├── generation_description_qwen3_7_api.py
│   ├── pre_finetune/
│   │   ├── generate_description_qwen.py
│   │   └── llm-selection/chat_qwen/
│   └── post_finetune/
│       ├── split_cad_json_dataset.py
│       ├── generate_teaching_description_ollama.py
│       ├── split_cad_json_dataset_ollama_html_steps.py
│       ├── lora_auto_test_scripts/
│       ├── rule_validators/
│       └── validate/
├── utils/                      # Shared configuration, device, I/O, and logging helpers
│   └── finetuned_model_client.py
├── tests/                      # Offline tests
├── data/
│   └── README.md               # Dataset layout and preparation notes
├── docs/
│   └── experiment_notes.md     # Experiment-to-code mapping and verification notes
├── third_party/
│   └── README.md               # LLaMA-Factory setup notes
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Environment and Installation

### Tested environment

The original experiments were conducted on a Linux GPU server with the following software components:

- Python 3.10
- LLaMA-Factory 0.9.5.108
- PyTorch and CUDA-compatible training dependencies
- Transformers, PEFT, and bitsandbytes
- pythonocc-core for Open CASCADE-based geometry checks

The actual compatibility of these components depends on the selected hardware, CUDA version, and package versions.

### Basic installation

```bash
conda create -n cascade python=3.10 -y
conda activate cascade
python -m pip install -r requirements.txt
```

### Optional dependencies and services

- **GPU fine-tuning:** Requires a compatible CUDA-enabled PyTorch and bitsandbytes installation, together with LLaMA-Factory.
- **Geometry validation:** Requires a compatible Open CASCADE Python binding, such as `pythonocc-core`.
- **Instructional text generation:** Requires a configured Ollama-compatible endpoint when LLM-assisted generation is enabled.
- **API-based description generation:** Requires access to a compatible inference endpoint and a valid API key.

See `third_party/README.md` for LLaMA-Factory setup notes.

A successful dependency installation does not by itself demonstrate that the full training, inference, and reconstruction pipeline is operational. Verify each stage in the target environment.

## Data Preparation

The underlying geometric and topological information is derived from the public DeepCAD dataset. Obtain the original dataset from its official source and follow the applicable license and terms of use.

**No dataset is bundled with this repository.**

Dataset layout conventions are documented in `data/README.md`. The core data format uses pairs of files:

- `*_description.txt`: Natural-language modeling descriptions.
- `*_lite.json`: Hierarchical CAD-JSON ground truth.

The final training pipeline uses English descriptions and English instruction templates. Relevant scripts include `scripts/make_train_json.py` and `scripts/make_val_json.py`.

### Build prompt and training files

Build `prompts.jsonl` for model screening or batch inference:

```bash
python scripts/make_prompts_jsonl.py \
  --data_root data/prompt_data \
  --output configs/prompts.jsonl
```

Build the Alpaca-format training and validation files:

```bash
python scripts/make_train_json.py
python scripts/make_val_json.py
```

The scripts define their input and output paths internally. Review those paths before running them.

### English description generation

Description generation is needed only when rebuilding the description files.

- `scripts/pre_finetune/generate_description_qwen.py`: Generates English procedural descriptions. Relevant environment variables include `DATASET_ROOT`, `OUTPUT_DIR`, `QWEN_BASE_URL`, `DASHSCOPE_BASE_URL`, `QWEN_MODEL`, `DASHSCOPE_MODEL`, `QWEN_API_KEY`, `DASHSCOPE_API_KEY`, and `USE_IMAGES`.
- `scripts/generation_description_qwen3_7_api.py`: Generates English descriptions using an LLM and can optionally use rendered views. Relevant options include `USE_IMAGES`, `REQUIRE_IMAGES`, and `IMAGE_VIEWS`.

Check the headers of the corresponding scripts for the complete list of supported variables and configuration details.

Do not commit API keys, private data, or generated files containing sensitive information.

## Qwen3.6-27B QLoRA Fine-Tuning

The main training configuration is located at:

```text
configs/train_lora_custom/qwen3_6_27b_lora_sft.yaml
```

The training pipeline consists of dataset registration, fine-tuning, LoRA merging, and merge validation.

### 1. Prepare the dataset registration

After generating the training data, register the dataset with LLaMA-Factory using the configuration in `configs/llamafactory/`.

For example, if the generated dataset file is `configs/cad_train.json`:

```bash
cp configs/cad_train.json configs/llamafactory/cad_train.json
```

Verify that the dataset registration in `dataset_info.json` matches the actual dataset path and format.

### 2. Run fine-tuning

```bash
llamafactory-cli train \
  configs/train_lora_custom/qwen3_6_27b_lora_sft.yaml \
  --dataset_dir configs/llamafactory
```

Update `model_name_or_path` in the training configuration to the local base-model checkpoint before training.

### 3. Merge the LoRA adapter

```bash
python scripts/merge_lora_qwen.py
```

### 4. Validate the merged model

```bash
python scripts/post_finetune/validate/validate_lora_merge.py
```

### Main training configuration

The supplied configuration specifies the following settings:

- Fine-tuning stage: supervised fine-tuning (`sft`)
- Fine-tuning method: LoRA
- Quantization: 4-bit
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- Learning rate: `5.0e-5`
- Training epochs: 1
- Learning-rate schedule: cosine
- Warmup ratio: 0.03
- Per-device training batch size: 1
- Gradient accumulation steps: 4
- Maximum sequence length: 4096
- Maximum samples: 10000
- Attention implementation: FlashAttention-2
- Training framework template: Qwen

These values describe the supplied configuration; they do not imply that the training commands have been executed successfully in the release environment.

## Inference and Evaluation

The batch inference utilities are located in:

```text
scripts/post_finetune/lora_auto_test_scripts/
```

A typical invocation of `batch_lora_infer.py` is:

```bash
cd scripts/post_finetune/lora_auto_test_scripts

python batch_lora_infer.py \
  --model_name_or_path <base-model-path> \
  --adapter_name_or_path <lora-adapter-path> \
  --prompts_path <prompts.jsonl> \
  --output <normalized-output.jsonl>
```

Replace the path placeholders with the actual local paths.

The script supports batch generation and JSON normalization. Run the following command to inspect all available arguments:

```bash
python batch_lora_infer.py --help
```

Optional arguments include settings for LoRA merging, evaluation input, temperature, sample limits, and raw output.

## Validation and Repair Ablations

The validation and repair experiments are implemented in `ablation/`.

### Run the repair ablation groups

From the repository root:

```bash
python ablation/run_ablation.py --group repair --difficulty all
python ablation/run_ablation.py --group trigger --difficulty all
python ablation/run_ablation.py --group feedback --difficulty all

python ablation/summarize_all.py
```

### Run the overall comparison

```bash
python ablation/main_eval/run_main.py
```

### Experiment configuration

`ablation/config.yaml` defines test splits, model paths, and repair variants. The configured groups include:

- `full_repair`
- `no_repair`
- `syntax_only_repair`
- `syntax_schema_repair`
- `full_generic_feedback`
- `full_no_error_feedback`
- `full_one_round`
- `full_three_rounds`
- `full_allow_no_effect_boolean`

The shell wrapper `ablation/run_all.sh` runs the repair, trigger, and feedback groups and the summary step in a Bash-compatible environment.

### Repair mechanism

The repair loop in `ablation/repair.py` is designed to operate without reference CAD-JSON during repair. It feeds the task, the previous output, and validation errors back to the model.

The reference JSON is used for evaluation only and is not included in repair prompts.

The validation pipeline in `ablation/validation.py` combines syntax, schema, and geometry checks. Geometry checks use Open CASCADE functionality where available.

### Model and dataset paths

The overall evaluation script reads the following environment variables:

- `CASCODE_DATASET`
- `CASCODE_BASE_MODEL`
- `CASCODE_MERGED_MODEL`

The LoRA merge helper `scripts/merge_lora_qwen.py` also reads:

- `CASCODE_BASE_MODEL`
- `CASCODE_LORA_CKPT`
- `CASCODE_MERGED_MODEL`

Review the relevant scripts and configuration files to confirm that the paths match your environment before running the experiments.

## Instructional Reconstruction

The instructional-reconstruction pipeline transforms a CAD-JSON modeling tree into smaller steps and generates explanations intended to help users understand the modeling process.

Three scripts implement the main workflows.

### 1. Rule-based step splitting

```text
scripts/post_finetune/split_cad_json_dataset.py
```

This script recursively splits a CAD-JSON modeling tree into separately renderable step JSON files. Its rules include concentric-circle prism splitting, Fuse/Cut decoupling, Group flattening, step color assignment, and manifest generation.

The input and output directories are configured using the `INPUT_DIR` and `OUTPUT_DIR` constants near the top of the script.

### 2. Generate instructional descriptions

```text
scripts/post_finetune/generate_teaching_description_ollama.py
```

This script generates step-by-step teaching text for split modeling steps. It supports options including:

- `--root_dir`
- `--output_dir`
- `--use_ollama`
- `--ollama_url`
- `--model`
- `--no_full_model_step`
- `--limit`

The model can also be specified through the `TEACHING_MODEL` environment variable.

### 3. Integrated splitting, explanation, and HTML generation

```text
scripts/post_finetune/split_cad_json_dataset_ollama_html_steps.py
```

This script combines splitting, LLM-assisted explanations, and HTML output generation. It supports an Ollama `/api/generate` endpoint and a rule-template fallback when `--no_ollama` is used.

Relevant options include:

- `--input_dir`
- `--output_dir`
- `--ollama_url`
- `--model`
- `--no_ollama`

The Web backend of the companion demo project contains a separate implementation of a similarly named splitting workflow. These implementations should be treated as independent until their interfaces and behavior have been compared.

## Base-Model Screening

The Qwen/DashScope screening tools are located in:

```text
scripts/pre_finetune/llm-selection/chat_qwen/
```

Run the Gradio interface from that directory:

```bash
python chat_modified_qwen.py
```

The script reads configuration from environment variables such as:

- `DASHSCOPE_API_KEY` or `QWEN_API_KEY`
- `DASHSCOPE_BASE_URL`
- `REFERENCE_JSON_ROOT`
- `QWEN_ENABLE_THINKING`

The model list in the script contains example model names and can be adjusted through the interface or relevant configuration.

Supporting files include:

- `Rating_modified.py`
- `Rating_raw_structure.py`
- `dataset_prompts.py`
- `cadjson_template_framework.py`

Only the Qwen channel is included in this repository.

## Runtime and Credential Conventions

- Unless otherwise stated, commands are intended to be run from the repository root.
- Scripts under `ablation/` resolve the repository root from their own location. Some scripts rely on same-directory imports, so keep them in their original directories.
- API keys must be supplied through environment variables, such as `DASHSCOPE_API_KEY` and `QWEN_API_KEY`. Never commit real credentials.
- Ollama teaching generation uses `TEACHING_MODEL` or the `--model` option.
- `utils/finetuned_model_client.py` targets an OpenAI-compatible inference endpoint using `CASCADE_API_BASE_URL`, `CASCADE_MODEL`, and `CASCADE_API_KEY`.

The companion Web backend may use different environment variables or client interfaces. Check each implementation independently rather than assuming the two clients are interchangeable.

## Evaluation Metrics

The evaluation scripts report several metrics.

- **JSON validity rate:** The proportion of generated outputs that pass JSON syntax validation.
- **Schema validity rate:** The proportion that passes the configured CAD-JSON schema checks.
- **Geometry validity rate:** The proportion that passes the configured geometry checks.
- **Overall validity rate:** The proportion of outputs that pass all required validation stages.
- **Repair trigger rate:** The proportion of samples that trigger the repair procedure.
- **Repair success rate:** The proportion of triggered repairs that satisfy the configured repair success criterion.
- **Average repair rounds:** The average number of repair rounds used by a variant.
- **CAD reference score:** A 0–100 composite score computed by `ablation/scoring.py` for comparison with reference CAD-JSON.

The repair prompts are reference-free. Reference CAD-JSON is used for evaluation, not as repair input.

Metric definitions and success criteria should be interpreted according to the implementation in the relevant evaluation scripts.

## Reproducibility and Limitations

- **Datasets and checkpoints:** No datasets, pretrained weights, or fine-tuned checkpoints are bundled. Users must obtain permitted data and model checkpoints and configure their local paths.
- **External services:** DashScope, Ollama-compatible endpoints, and any fine-tuned inference service must be configured separately.
- **Environment compatibility:** GPU training and Open CASCADE functionality depend on compatible software and hardware versions.
- **Offline tests versus end-to-end execution:** Passing offline tests does not demonstrate that training, inference, external APIs, or CAD reconstruction will succeed end to end.
- **Experimental results:** Paper-reported results must be reproduced by running the relevant pipelines with the corresponding data, checkpoints, and configurations. The presence of scripts or configuration files alone does not verify those results.
- **Known implementation details:** Consult `docs/experiment_notes.md` for experiment-to-code mappings and remaining verification items.

## Tests

Run the offline test suite from the repository root:

```bash
python -m pytest tests/ -q
```

**Latest checked result for this repository snapshot: 18 tests passed.**

The offline tests cover configuration parsing, entry points, schema validation, scoring, syntax compilation, and shared utilities.

These tests do not verify GPU fine-tuning, external API access, full-model inference, numerical reproduction of paper results, or end-to-end CAD reconstruction.

If the code or test suite changes, rerun the tests and update this section to reflect the actual result.

## Citation

If you use CasCADe in your research, please cite this software repository.

Formal publication information will be added when it becomes available.

## License

The CasCADe source code developed for this project is released under the MIT License. See the [LICENSE] file for details.

Third-party software, datasets, and pretrained models used in this project are subject to their respective licenses and terms of use. The MIT License for this repository does not grant additional rights to third-party materials.