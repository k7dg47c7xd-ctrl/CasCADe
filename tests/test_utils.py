# -*- coding: utf-8 -*-
"""Tests for the dependency-free parts of the shared utilities (utils.config / utils.logger)."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import load_config  # noqa: E402


def test_load_config_ok(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("a: 1\nb:\n  - 2\n  - 3\n", encoding="utf-8")
    data = load_config(str(cfg))
    assert data["a"] == 1
    assert data["b"] == [2, 3]


def test_load_config_missing():
    with pytest.raises(FileNotFoundError):
        load_config(str(ROOT / "definitely" / "missing.yaml"))
