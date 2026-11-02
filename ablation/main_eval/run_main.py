import os
import subprocess
from pathlib import Path


# ============================================================
# 1. Replace these with your own paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

def _p(v):
    p = Path(v)
    return str(p if p.is_absolute() else (REPO_ROOT / p))


DATASET = _p(os.getenv("CASCODE_DATASET", "ablation/splits/simple/simple.jsonl"))

BASE_MODEL = _p(os.getenv("CASCODE_BASE_MODEL", "saves/base/Qwen3.6-27B"))

SFT_ADAPTER = _p(os.getenv("CASCODE_MERGED_MODEL", "saves/qwen3_6_27b/lora+qwen3_6_27b_fixed"))


# ============================================================
# 2. Experiment configurations
# ============================================================

configs = [

    {
        "name": "base_no_repair",
        "model": "base",
        "repair": False,
    },

    {
        "name": "base_full_repair",
        "model": "base",
        "repair": True,
    },

    {
        "name": "sft_no_repair",
        "model": "sft",
        "repair": False,
    },

    {
        "name": "sft_full_repair",
        "model": "sft",
        "repair": True,
    },

]


# ============================================================
# 3. Path checks
# ============================================================

if not Path(DATASET).exists():

    raise FileNotFoundError(
        f"Dataset not found: {DATASET}"
    )


if not Path(BASE_MODEL).exists():

    raise FileNotFoundError(
        f"Base model not found: {BASE_MODEL}"
    )


if not Path(SFT_ADAPTER).exists():

    raise FileNotFoundError(
        f"SFT adapter not found: {SFT_ADAPTER}"
    )


# ============================================================
# 4. Run the four experiments in sequence
# ============================================================

for c in configs:

    output = str(
        REPO_ROOT / "ablation" / "main_eval" / "outputs" / "main_results" / c["name"]
    )


    os.makedirs(
        output,
        exist_ok=True,
    )


    cmd = [

        "python",

        str(REPO_ROOT / "ablation" / "generate.py"),

        "--dataset",
        DATASET,

        "--output_dir",
        output,

        "--variant_name",
        c["name"],

        "--model_type",
        c["model"],

        "--base_model_path",
        BASE_MODEL,

        "--sft_adapter_path",
        SFT_ADAPTER,

        "--temperature",
        "0",

        "--max_repair_rounds",
        "3",

    ]


    if c["repair"]:

        cmd.append(
            "--repair"
        )


    print("\n" + "=" * 80)

    print(
        "Running:"
    )

    print(
        " ".join(cmd)
    )

    print("=" * 80 + "\n")


    result = subprocess.run(
        cmd
    )


    if result.returncode != 0:

        raise RuntimeError(
            f"Experiment failed: "
            f"{c['name']}"
        )


print(
    "\nAll main-result experiments finished."
)