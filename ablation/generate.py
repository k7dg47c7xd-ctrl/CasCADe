# -*- coding: utf-8 -*-
"""Batch generation with Syntax -> Schema -> Geometry -> iterative repair."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from scoring import score_text_against_reference
from validation import validate_for_repair
from repair import SYSTEM_PROMPT, iterative_repair


def load_client(base_url: str, api_key: str):
    try:
        from openai import OpenAI

        return OpenAI(
            api_key=api_key or "EMPTY",
            base_url=base_url,
            timeout=1800,
        )

    except Exception as e:
        raise RuntimeError(
            "API mode requires 'pip install openai'. "
            "For local mode use --local_model, "
            "or --model_type together with --base_model_path."
        ) from e


def call_api(
    client,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    return resp.choices[0].message.content or ""


class LocalModel:
    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )

        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        print(f"[Model] Loading base model: {model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        if adapter_path:
            from peft import PeftModel

            print(f"[Model] Loading LoRA adapter: {adapter_path}")

            model = PeftModel.from_pretrained(
                model,
                adapter_path,
            )

        else:
            print("[Model] No adapter loaded (base model).")

        model.eval()

        self.model = model


    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:

        import torch

        text = self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tok(
            text,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():

            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                top_p=1.0,
            )

        return self.tok.decode(
            out[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )


def _validation_flags(trigger: str):
    """
    Return:

        enable_schema
        enable_geometry

    for repair-trigger ablations.
    """

    return (
        trigger in {"syntax_schema", "full"},
        trigger == "full",
    )


def resolve_local_model_paths(args):
    """
    Decide, based on the command-line arguments:

    1. which base model to load;
    2. whether to load a LoRA adapter.

    Precedence:

    model_type mode
        |
        v
    legacy local_model / adapter mode
    """

    # ----------------------------------
    # New: Main Results mode
    # ----------------------------------

    if args.model_type:

        if args.model_type == "base":

            model_path = (
                args.base_model_path
                or args.local_model
            )

            adapter_path = None


        elif args.model_type == "sft":

            model_path = (
                args.base_model_path
                or args.local_model
            )

            adapter_path = (
                args.sft_adapter_path
                or args.adapter
            )


        else:
            raise ValueError(
                f"Unknown model_type: {args.model_type}"
            )


        if not model_path:

            raise SystemExit(
                "\n[ERROR] --model_type requires a base model path.\n"
                "\n"
                "For example:\n"
                "--base_model_path saves/base/Qwen3.6-27B\n"
            )


        if args.model_type == "sft" and not adapter_path:

            raise SystemExit(
                "\n[ERROR] model_type=sft was given without a LoRA adapter.\n"
                "\n"
                "Provide either:\n"
                "--sft_adapter_path saves/qwen3_6_27b/lora/sft_8654\n"
                "\n"
                "or:\n"
                "--adapter saves/qwen3_6_27b/lora+qwen3_6_27b_fixed\n"
            )


        return model_path, adapter_path


    # ----------------------------------
    # Legacy local-model mode
    # ----------------------------------

    if args.local_model:

        return (
            args.local_model,
            args.adapter or None,
        )


    return None, None


def load_initial_cache(path: Path) -> Dict[str, str]:
    """Load the shared initial-output cache (JSONL entries with id/description/text).

    Returns a mapping from the row cache key (row id, falling back to the
    description text) to the cached first-pass model output.
    """
    cache: Dict[str, str] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = str(entry.get("id") or entry.get("description") or "")
        if key and isinstance(entry.get("text"), str):
            cache[key] = entry["text"]
    return cache


def append_initial_cache_entry(
    path: Path,
    row_id: str,
    description: str,
    text: str,
) -> None:
    """Append one first-pass output to the shared initial-output cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "id": row_id,
        "description": description,
        "text": text,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()


def main():

    ap = argparse.ArgumentParser()


    # =========================================================
    # Dataset / output
    # =========================================================

    ap.add_argument(
        "--dataset",
        required=True,
        help="Evaluation dataset jsonl",
    )


    # Accepts both:
    #
    #   --output_dir xxx
    #
    # and:
    #
    #   --output xxx
    #
    ap.add_argument(
        "--output_dir",
        "--output",
        dest="output_dir",
        required=True,
        help="Output directory",
    )


    # =========================================================
    # Experiment metadata
    # =========================================================

    ap.add_argument(
        "--variant_name",
        default="",
        help="Experiment variant name, e.g. sft_full_repair",
    )

    ap.add_argument(
        "--initial_cache",
        default="",
        help=(
            "Path to a shared initial-output cache file (JSONL). "
            "When set, the first-pass output of each row is reused from "
            "the cache when an entry with the same row id exists, and is "
            "appended to the cache otherwise, so that all ablation "
            "variants share the same initial y0."
        ),
    )


    # =========================================================
    # API model
    # =========================================================

    ap.add_argument(
        "--model_id",
        default="cascade",
    )

    ap.add_argument(
        "--api_base_url",
        default="",
    )

    ap.add_argument(
        "--api_key",
        default="EMPTY",
    )


    # =========================================================
    # Original local-model interface
    # =========================================================

    ap.add_argument(
        "--local_model",
        default="",
        help="Base model path",
    )

    ap.add_argument(
        "--adapter",
        default="",
        help="LoRA adapter path",
    )


    # =========================================================
    # NEW: Main Results interface
    # =========================================================

    ap.add_argument(
        "--model_type",
        choices=["base", "sft"],
        default=None,
        help=(
            "Main-results model type. "
            "base = base model only; "
            "sft = base model + LoRA adapter"
        ),
    )

    ap.add_argument(
        "--base_model_path",
        default="",
        help="Path to pretrained/base LLM",
    )

    ap.add_argument(
        "--sft_adapter_path",
        default="",
        help="Path to fine-tuned LoRA adapter",
    )


    # =========================================================
    # Repair
    # =========================================================

    ap.add_argument(
        "--repair",
        action="store_true",
    )

    ap.add_argument(
        "--repair_trigger",
        choices=[
            "syntax",
            "syntax_schema",
            "full",
        ],
        default="full",
        help=(
            "full = syntax + schema + geometry "
            "can trigger repair"
        ),
    )

    ap.add_argument(
        "--feedback_mode",
        choices=[
            "structured",
            "generic",
            "none",
        ],
        default="structured",
    )

    ap.add_argument(
        "--allow_no_effect_boolean",
        action="store_true",
        help=(
            "Treat cut/fuse no-effect as warning "
            "instead of repair-triggering error"
        ),
    )

    ap.add_argument(
        "--max_repair_rounds",
        type=int,
        default=3,
    )


    # =========================================================
    # Generation
    # =========================================================

    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
    )

    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    ap.add_argument(
        "--top_p",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=0,
    )


    args = ap.parse_args()


    # =========================================================
    # Output directory
    # =========================================================

    out_dir = Path(args.output_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # =========================================================
    # Dataset
    # =========================================================

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():

        raise SystemExit(
            f"\n[ERROR] Dataset not found:\n"
            f"{dataset_path.resolve()}\n"
        )


    rows = [
        json.loads(line)
        for line in dataset_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


    if args.limit:

        rows = rows[: args.limit]


    print(
        f"[Dataset] {dataset_path}"
    )

    print(
        f"[Dataset] Number of samples: {len(rows)}"
    )


    # =========================================================
    # Shared initial-output cache (reuse_initial_outputs)
    # =========================================================

    initial_cache_path = (
        Path(args.initial_cache)
        if args.initial_cache
        else None
    )

    initial_cache: Dict[str, str] = {}

    if initial_cache_path is not None:

        initial_cache = load_initial_cache(
            initial_cache_path
        )

        print(
            f"[Cache] {len(initial_cache)} initial-output "
            f"entries loaded from {initial_cache_path}"
        )


    # =========================================================
    # Resolve model
    # =========================================================

    model_path, adapter_path = resolve_local_model_paths(
        args
    )


    client = None

    local: Optional[LocalModel] = None


    # Local mode
    if model_path:

        local = LocalModel(
            model_path,
            adapter_path,
        )


    # API mode
    else:

        if not args.api_base_url:

            raise SystemExit(
                "\n[ERROR] No usable model is configured.\n\n"
                "Local mode requires:\n"
                "--local_model MODEL_PATH\n\n"
                "Main Results mode requires:\n"
                "--model_type base/sft "
                "--base_model_path MODEL_PATH\n\n"
                "API mode requires:\n"
                "--api_base_url URL\n"
            )

        client = load_client(
            args.api_base_url,
            args.api_key,
        )


    # =========================================================
    # Print experiment information
    # =========================================================

    print("\n" + "=" * 60)

    print(
        f"Variant       : "
        f"{args.variant_name or 'unnamed'}"
    )

    print(
        f"Model type    : "
        f"{args.model_type or 'legacy/API'}"
    )

    print(
        f"Repair        : "
        f"{args.repair}"
    )

    print(
        f"Repair trigger: "
        f"{args.repair_trigger}"
    )

    print(
        f"Feedback mode : "
        f"{args.feedback_mode}"
    )

    print(
        f"Max rounds    : "
        f"{args.max_repair_rounds}"
    )

    print(
        f"Output        : "
        f"{out_dir.resolve()}"
    )

    print("=" * 60 + "\n")


    # =========================================================
    # Repair-trigger validation
    # =========================================================

    enable_schema, enable_geometry = _validation_flags(
        args.repair_trigger
    )


    def validate_text(text: str):

        return validate_for_repair(
            text,
            enable_schema=enable_schema,
            enable_geometry=enable_geometry,
            geometry_no_effect_is_error=(
                not args.allow_no_effect_boolean
            ),
        )


    # =========================================================
    # Generation functions
    # =========================================================

    def generate_initial(
        messages: List[Dict[str, str]]
    ) -> str:

        if local:

            return local.generate(
                messages,
                args.max_new_tokens,
                args.temperature,
            )

        return call_api(
            client,
            args.model_id,
            messages,
            args.temperature,
            args.top_p,
            args.max_new_tokens,
        )


    def generate_repair(
        messages: List[Dict[str, str]]
    ) -> str:

        # Repair is deterministic by design.

        if local:

            return local.generate(
                messages,
                args.max_new_tokens,
                0.0,
            )

        return call_api(
            client,
            args.model_id,
            messages,
            0.0,
            0.1,
            args.max_new_tokens,
        )


    # =========================================================
    # Main evaluation loop
    # =========================================================

    out_jsonl = out_dir / "rows.jsonl"


    with out_jsonl.open(
        "w",
        encoding="utf-8",
    ) as f:

        for i, row in enumerate(rows):


            # -------------------------------------------------
            # Reference
            # -------------------------------------------------

            reference_path = Path(
                row["reference_path"]
            )

            if not reference_path.is_absolute():

                # Resolve relative paths against the dataset directory first
                candidate = (
                    dataset_path.parent
                    / reference_path
                )

                if candidate.exists():

                    reference_path = candidate


            if not reference_path.exists():

                raise FileNotFoundError(
                    f"\nReference JSON not found:\n"
                    f"{row['reference_path']}\n"
                    f"Resolved as:\n"
                    f"{reference_path.resolve()}\n"
                )


            reference = json.loads(
                reference_path.read_text(
                    encoding="utf-8"
                )
            )


            # -------------------------------------------------
            # Initial generation
            # -------------------------------------------------

            base_msgs = [

                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },

                {
                    "role": "user",
                    "content": row["description"],
                },

            ]


            cache_key = str(
                row.get("id")
                or row["description"]
            )

            cache_hit = cache_key in initial_cache

            if cache_hit:

                raw = initial_cache[cache_key]

                print(
                    f"[{i + 1}/{len(rows)}] "
                    f"{row['id']} initial output reused "
                    f"from cache"
                )

            else:

                raw = generate_initial(
                    base_msgs
                )

                if initial_cache_path is not None:

                    append_initial_cache_entry(
                        initial_cache_path,
                        str(row.get("id") or ""),
                        row["description"],
                        raw,
                    )

                    initial_cache[cache_key] = raw


            # -------------------------------------------------
            # Raw evaluation
            # -------------------------------------------------

            raw_score = score_text_against_reference(
                raw,
                reference,
            )

            raw_val = validate_text(
                raw
            )


            # -------------------------------------------------
            # Repair
            # -------------------------------------------------

            if (
                args.repair
                and not raw_val["valid"]
            ):

                (
                    current,
                    current_val,
                    repair_history,
                    repair_rounds,
                ) = iterative_repair(

                    requirement=row["description"],

                    initial_text=raw,

                    initial_validation=raw_val,

                    generate_fn=generate_repair,

                    validate_fn=validate_text,

                    max_rounds=args.max_repair_rounds,

                    feedback_mode=args.feedback_mode,

                )

            else:

                current = (
                    raw_val.get("normalized_json")
                    or raw
                )

                current_val = raw_val

                repair_history = []

                repair_rounds = 0


            # -------------------------------------------------
            # Final score
            # -------------------------------------------------

            final_score = score_text_against_reference(
                current,
                reference,
            )


            # -------------------------------------------------
            # Record
            # -------------------------------------------------

            rec = {

                "id":
                    row["id"],

                "variant":
                    args.variant_name,

                "model_type":
                    args.model_type,

                "repair_enabled":
                    bool(args.repair),

                "initial_cache_hit":
                    cache_hit,

                "description":
                    row["description"],

                "reference_path":
                    row["reference_path"],

                "raw_text":
                    raw,

                "final_text":
                    current,


                # Repair
                "repair_rounds":
                    repair_rounds,

                "repair_history":
                    repair_history,

                "repair_trigger":
                    args.repair_trigger,

                "feedback_mode":
                    args.feedback_mode,


                # Reference score
                "raw_score":
                    raw_score.get(
                        "reference_compare_score",
                        0.0,
                    ),

                "optimized_score":
                    final_score.get(
                        "reference_compare_score",
                        0.0,
                    ),


                # JSON
                "json_valid_raw":
                    bool(
                        raw_val.get("json_valid")
                    ),

                "json_valid_final":
                    bool(
                        current_val.get(
                            "json_valid"
                        )
                    ),


                # Schema
                "schema_valid_raw":
                    raw_val.get(
                        "schema_valid"
                    ),

                "schema_valid_final":
                    current_val.get(
                        "schema_valid"
                    ),


                # Geometry
                "geometry_valid_raw":
                    raw_val.get(
                        "geometry_valid"
                    ),

                "geometry_valid_final":
                    current_val.get(
                        "geometry_valid"
                    ),


                # Overall
                "validation_valid_raw":
                    bool(
                        raw_val.get("valid")
                    ),

                "validation_valid_final":
                    bool(
                        current_val.get("valid")
                    ),


                # Errors
                "validation_errors_raw":
                    raw_val.get(
                        "errors",
                        [],
                    ),

                "validation_errors_final":
                    current_val.get(
                        "errors",
                        [],
                    ),


                # Geometry details
                "geometry_checks_raw":
                    raw_val.get(
                        "geometry",
                        {},
                    ).get(
                        "checks",
                        [],
                    ),

                "geometry_checks_final":
                    current_val.get(
                        "geometry",
                        {},
                    ).get(
                        "checks",
                        [],
                    ),

                "geometry_warnings_raw":
                    raw_val.get(
                        "geometry",
                        {},
                    ).get(
                        "warnings",
                        [],
                    ),

                "geometry_warnings_final":
                    current_val.get(
                        "geometry",
                        {},
                    ).get(
                        "warnings",
                        [],
                    ),

                "geometry_volume_final":
                    current_val.get(
                        "geometry",
                        {},
                    ).get(
                        "volume",
                        0.0,
                    ),


                # Detailed score
                "detail_raw":
                    raw_score,

                "detail_final":
                    final_score,

            }


            f.write(
                json.dumps(
                    rec,
                    ensure_ascii=False,
                )
                + "\n"
            )

            f.flush()


            print(

                f"[{i + 1}/{len(rows)}] "
                f"{row['id']} "

                f"raw={rec['raw_score']:.2f} "

                f"final="
                f"{rec['optimized_score']:.2f} "

                f"valid="
                f"{rec['validation_valid_final']} "

                f"repair="
                f"{repair_rounds}"

            )


    print(
        "\nsaved",
        out_jsonl,
    )


if __name__ == "__main__":
    main()