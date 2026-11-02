#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

SCRIPT_VERSION = "build_batch_config.v2_fixed"

# =========================
# Edit area
# =========================
DATA_ROOT = Path(
    "data/prompt_data"
).resolve()
EXPECTED_PNG_COUNT = 7
CONTINUE_ON_ERROR = True

OUTPUT_PATH = Path(
    "configs/batch_generated.yaml"
)

REQUIRED_VIEWS = {"front", "back", "left", "right", "top", "bottom", "iso"}


def find_samples(data_root: Path, expected_png_count: int = 7) -> List[Dict[str, str]]:
    samples = []

    for sample_dir in sorted(data_root.iterdir()):
        if not sample_dir.is_dir():
            continue

        uid = sample_dir.name
        json_path = sample_dir / f"{uid}_lite.json"
        if not json_path.exists():
            print(f"[SKIP] JSON missing: {json_path}")
            continue

        views: Dict[str, str] = {}
        for png in sample_dir.glob("*.png"):
            name = png.stem.lower()
            if name.endswith("_front"):
                views["front"] = str(png)
            elif name.endswith("_back"):
                views["back"] = str(png)
            elif name.endswith("_left"):
                views["left"] = str(png)
            elif name.endswith("_right"):
                views["right"] = str(png)
            elif name.endswith("_top"):
                views["top"] = str(png)
            elif name.endswith("_bottom"):
                views["bottom"] = str(png)
            elif name.endswith("_iso"):
                views["iso"] = str(png)

        # Check whether any view is missing
        missing = REQUIRED_VIEWS - set(views.keys())
        if missing:
            print(f"[SKIP] missing views in {uid}: {missing}")
            continue

        samples.append({
            "uid": uid,
            "minimal_json": str(json_path),
            "images": views
        })

    return samples


def dump_batch_yaml(samples: List[Dict[str, str]], continue_on_error: bool = True) -> str:
    lines = []
    lines.append("batch:")
    lines.append("  enabled: true")
    lines.append(f"  continue_on_error: {'true' if continue_on_error else 'false'}")
    lines.append("  items:")

    for s in samples:
        lines.append(f'    - uid: "{s["uid"]}"')
        lines.append(f'      minimal_json: "{s["minimal_json"]}"')
        lines.append("      images:")
        for view in ["front", "back", "left", "right", "top", "bottom", "iso"]:
            lines.append(f'        {view}: "{s["images"][view]}"')

    return "\n".join(lines) + "\n"


def main() -> None:
    print(f"SCRIPT_VERSION = {SCRIPT_VERSION}")
    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"EXPECTED_PNG_COUNT = {EXPECTED_PNG_COUNT}")
    print(f"OUTPUT_PATH = {OUTPUT_PATH}")

    samples = find_samples(
        data_root=DATA_ROOT,
        expected_png_count=EXPECTED_PNG_COUNT,
    )

    if not samples:
        raise RuntimeError("No valid samples found.")

    yaml_text = dump_batch_yaml(
        samples=samples,
        continue_on_error=CONTINUE_ON_ERROR,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(yaml_text, encoding="utf-8")

    print(f"[OK] wrote: {OUTPUT_PATH}")
    print(f"[OK] samples: {len(samples)}")


if __name__ == "__main__":
    main()