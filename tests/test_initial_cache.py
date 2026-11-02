# -*- coding: utf-8 -*-
"""Offline tests for the shared initial-output cache helpers in ablation/generate.py.

The helpers implement the ``reuse_initial_outputs: true`` semantics of
``ablation/config.yaml``: all ablation variants share the same first-pass
model output (initial y0) per dataset row.
"""
import json
import pathlib
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ablation"))

from generate import append_initial_cache_entry, load_initial_cache  # noqa: E402

_TMP_DIRS = []


def _tmp_dir():
    try:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="cascade_initial_cache_test_"))
        probe = tmp / ".probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        tmp = ROOT / "tests" / ("_tmp_" + next(tempfile._get_candidate_names()))
        tmp.mkdir(parents=True, exist_ok=True)
    _TMP_DIRS.append(tmp)
    return tmp


def _cleanup():
    for tmp in _TMP_DIRS:
        shutil.rmtree(tmp, ignore_errors=True)
    _TMP_DIRS.clear()


def test_load_missing_cache_is_empty():
    cache = load_initial_cache(_tmp_dir() / "missing.jsonl")
    assert cache == {}


def test_append_and_load_roundtrip():
    tmp = _tmp_dir()
    path = tmp / "cache.jsonl"
    append_initial_cache_entry(path, "id_1", "make a box", '{"type": "Group"}')
    append_initial_cache_entry(path, "", "make a cylinder", '{"type": "Prism"}')
    cache = load_initial_cache(path)
    assert cache["id_1"] == '{"type": "Group"}'
    assert cache["make a cylinder"] == '{"type": "Prism"}'


def test_load_skips_malformed_lines():
    tmp = _tmp_dir()
    path = tmp / "cache.jsonl"
    path.write_text(
        "not json\n"
        + json.dumps({"id": "ok", "description": "d", "text": "t"})
        + "\n",
        encoding="utf-8",
    )
    cache = load_initial_cache(path)
    assert cache == {"ok": "t"}


def test_cache_key_falls_back_to_description():
    tmp = _tmp_dir()
    path = tmp / "cache.jsonl"
    append_initial_cache_entry(path, "", "build a plate", "{}")
    cache = load_initial_cache(path)
    assert cache["build a plate"] == "{}"


if __name__ == "__main__":
    try:
        for fn in [
            test_load_missing_cache_is_empty,
            test_append_and_load_roundtrip,
            test_load_skips_malformed_lines,
            test_cache_key_falls_back_to_description,
        ]:
            fn()
            print("PASS", fn.__name__)
        print("ALL PASS")
    finally:
        _cleanup()
