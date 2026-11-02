# -*- coding: utf-8 -*-
"""Offline syntax check: compile all Python files in the repository (no torch/OCC)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXCLUDE = {"third_party", "cad-chat-demo-java-qwen-integrated"}


def _iter_py():
    for p in sorted(ROOT.rglob("*.py")):
        rel = p.relative_to(ROOT)
        if rel.parts and rel.parts[0] in EXCLUDE:
            continue
        if "__pycache__" in rel.parts:
            continue
        yield p


def test_all_python_files_compile():
    failed = []
    checked = 0
    for p in _iter_py():
        checked += 1
        try:
            compile(p.read_text(encoding="utf-8"), str(p), "exec")
        except SyntaxError as exc:
            failed.append(f"{p.relative_to(ROOT)}: {exc}")
    assert checked > 0, "no python files found"
    assert not failed, "Syntax errors:\n" + "\n".join(failed)
