# -*- coding: utf-8 -*-
"""Offline --help checks for entry scripts (only entries without torch/OCC imports)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

ENTRIES = [
    "ablation/run_ablation.py",
    "ablation/generate.py",
    "ablation/analyze.py",
    "ablation/build_dataset.py",
    "scripts/make_prompts_jsonl.py",
]


def test_entrypoints_help():
    for rel in ENTRIES:
        proc = subprocess.run(
            [PY, str(ROOT / rel), "--help"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT),
        )
        assert proc.returncode == 0, f"{rel} --help failed: {proc.stderr[:200]}"
        assert "usage" in proc.stdout.lower()
