# -*- coding: utf-8 -*-
"""
CAD Chat Demo backend: Qwen -> Cadmium-lite CAD JSON -> step splitting -> STEP/STL model display.

Main flow:
1. POST /api/chat: receives a natural-language modeling request and calls Qwen to generate the CAD JSON.
2. The normalized JSON is saved to CAD_JSON_OUTPUT_DIR.
3. Optionally call CAD_CONVERT_API_URL; otherwise wait for an external converter that watches the directory and writes a same-named model file.
4. POST /api/generate: one-click generation calls split_cad_json_dataset_ollama_html_steps.py to output per-step JSON and HTML explanations.
5. POST /api/select-json: after a JSON card is clicked, show the corresponding STL/STEP; if the model file does not exist, show a JSON summary.
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from uuid import uuid4

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
# .env must be loaded before qwen_cad_client is imported; otherwise QWEN_MODEL and other settings may not take effect.
load_dotenv(BASE_DIR / ".env", override=True)

from qwen_cad_client import generate_cad_json  # noqa: E402
from split_cad_json_dataset_ollama_html_steps import split_file  # noqa: E402

DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_dir(env_name: str, default: Path) -> Path:
    """Read a directory setting. Note: this must be a directory, not a concrete file."""
    value = os.getenv(env_name, "").strip().strip('"')
    path = Path(value) if value else default
    if not path.is_absolute():
        path = BASE_DIR / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


CAD_JSON_OUTPUT_DIR = _resolve_dir("CAD_JSON_OUTPUT_DIR", STATIC_DIR / "cad_json_in")
# The historical name CAD_STEP_OUTPUT_DIR is kept; it now means "model output directory" and may contain .step/.stp/.stl.
CAD_STEP_OUTPUT_DIR = _resolve_dir("CAD_STEP_OUTPUT_DIR", STATIC_DIR / "cad_step_out")
CAD_SPLIT_OUTPUT_DIR = _resolve_dir("CAD_SPLIT_OUTPUT_DIR", STATIC_DIR / "one_click")

CAD_CONVERT_API_URL = os.getenv("CAD_CONVERT_API_URL", "").strip()
CAD_CONVERT_WAIT_SECONDS = float(os.getenv("CAD_CONVERT_WAIT_SECONDS", "120"))
CAD_CONVERT_POLL_INTERVAL = float(os.getenv("CAD_CONVERT_POLL_INTERVAL", "0.5"))

CAD_OUTPUT_FORMAT = os.getenv("CAD_OUTPUT_FORMAT", "stl").strip().lower()
if CAD_OUTPUT_FORMAT == "slt":
    CAD_OUTPUT_FORMAT = "stl"
if CAD_OUTPUT_FORMAT not in {"step", "stp", "stl"}:
    raise RuntimeError("CAD_OUTPUT_FORMAT must be step, stp, or stl")

# Optional: fixed output file name. When empty, a unique name is generated from session + time.
CAD_JSON_OUTPUT_FILENAME = os.getenv("CAD_JSON_OUTPUT_FILENAME", "box_lite_demo.json").strip()
CAD_MODEL_OUTPUT_FILENAME = os.getenv("CAD_MODEL_OUTPUT_FILENAME", "").strip()

CAD_SPLIT_USE_LLM = _env_bool("CAD_SPLIT_USE_LLM", True)
CAD_SPLIT_MODEL = os.getenv("FINETUNED_MODEL", os.getenv("QWEN_STEP_MODEL", "")).strip()
# Whether per-step JSON files are also sent for STL/STEP conversion. Usually keep false to convert only the complete model.
CAD_CONVERT_SPLIT_JSONS = _env_bool("CAD_CONVERT_SPLIT_JSONS", False)

app = FastAPI(title="CAD Chat Demo Backend", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /static serves the project's internal static assets; if the fixed JSON/model directories are configured outside static, they are mounted separately.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if CAD_JSON_OUTPUT_DIR != STATIC_DIR and STATIC_DIR not in CAD_JSON_OUTPUT_DIR.parents:
    app.mount("/cad-json-files", StaticFiles(directory=str(CAD_JSON_OUTPUT_DIR)), name="cad-json-files")
if CAD_STEP_OUTPUT_DIR != STATIC_DIR and STATIC_DIR not in CAD_STEP_OUTPUT_DIR.parents:
    app.mount("/step-files", StaticFiles(directory=str(CAD_STEP_OUTPUT_DIR)), name="step-files")
if CAD_SPLIT_OUTPUT_DIR != STATIC_DIR and STATIC_DIR not in CAD_SPLIT_OUTPUT_DIR.parents:
    app.mount("/one-click-files", StaticFiles(directory=str(CAD_SPLIT_OUTPUT_DIR)), name="one-click-files")


class ChatRequest(BaseModel):
    sessionId: str
    message: str
    history: List[Dict[str, Any]] = Field(default_factory=list)


class SelectJsonRequest(BaseModel):
    name: str
    jsonData: Optional[Dict[str, Any]] = None
    stepUrl: Optional[str] = None


class GenerateRequest(BaseModel):
    sessionId: str
    finalJsonName: Optional[str] = None
    finalJsonData: Optional[Dict[str, Any]] = None


# ----------------------------- Multi-turn conversation memory -----------------------------

# Runtime session memory: keeps the complete CAD JSON last generated for each session.
# This lets Qwen see the previous model when the user says, in a second turn, "add a cube on top of this cylinder".
# Note: this is in-memory state and is cleared when the backend restarts; if the frontend history shows a JSON was already generated in this turn,
# the backend recovers once more from the fixed output JSON file as a fallback.
SESSION_MEMORY: Dict[str, Dict[str, Any]] = {}


def _history_has_generated_json(history: List[Dict[str, Any]]) -> bool:
    for item in history or []:
        if item.get("role") == "assistant" and "Generated CAD JSON" in str(item.get("content", "")):
            return True
    return False


def _latest_json_path() -> Optional[Path]:
    candidates: List[Path] = []

    if CAD_JSON_OUTPUT_FILENAME:
        name = Path(CAD_JSON_OUTPUT_FILENAME).name
        if not name.lower().endswith(".json"):
            name += ".json"
        candidates.append(CAD_JSON_OUTPUT_DIR / name)

    candidates.extend([p for p in CAD_JSON_OUTPUT_DIR.glob("*.json") if p.is_file()])
    candidates = [p for p in candidates if p.exists() and p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_latest_json_data() -> Optional[Dict[str, Any]]:
    path = _latest_json_path()
    if not path:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def build_contextual_requirement(req: ChatRequest) -> tuple[str, bool]:
    """
    Build the requirement actually sent to Qwen for /api/chat.
    First turn: use the user requirement directly.
    Later turns: put the complete CAD JSON from the previous turn into the prompt so the model modifies the old model,
    and require it to output the "modified complete JSON" rather than only the increment.
    """
    memory = SESSION_MEMORY.get(req.sessionId) or {}
    previous_json = memory.get("cad_json")
    previous_requirement = memory.get("requirement", "")

    if previous_json is None and _history_has_generated_json(req.history):
        previous_json = _load_latest_json_data()

    if not previous_json:
        return req.message, False

    previous_json_text = json.dumps(previous_json, ensure_ascii=False, indent=2)
    contextual_requirement = f"""This is a multi-turn CAD modeling conversation.

Previous user requirement:
{previous_requirement or "(no record)"}

The complete Cadmium-lite CAD JSON generated in the previous turn is:
{previous_json_text}

The user's new/modified requirement for this turn is:
{req.message}

Strictly modify the previous CAD JSON, preserving existing structures that the user did not ask to remove.
Phrases such as "on this model", "on this cylinder", "on the top", "beside it", or "add another" must be interpreted as continuing to edit the previous model.
Finally, output the complete modified Cadmium-lite CAD JSON, not only the newly added portion.
"""
    return contextual_requirement, True


# ----------------------------- File and URL utilities -----------------------------


def safe_file_stem(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_\-\.\u4e00-\u9fff]+", "_", text)
    text = text.strip("._-")
    return text or f"cad_{uuid4().hex[:8]}"


def _quote_parts(parts) -> str:
    return "/".join(quote(str(p)) for p in parts)


def _with_cache_buster(url: str, path: Path) -> str:
    """
    Model files often reuse the same name, e.g. box_lite_demo.stl.
    If the URL never changes, the browser / three.js loader may keep using the old cache.
    Append the file modification time to STL/STEP/STP URLs so the frontend reads the newest file.
    """
    if path.suffix.lower() in {".stl", ".step", ".stp"}:
        try:
            sep = "&" if "?" in url else "?"
            return f"{url}{sep}v={int(path.stat().st_mtime)}"
        except OSError:
            return url
    return url


def public_url_for_file(path: Optional[Path]) -> str:
    if not path:
        return ""
    path = Path(path).resolve()

    try:
        rel = path.relative_to(STATIC_DIR.resolve())
        return _with_cache_buster("/static/" + _quote_parts(rel.parts), path)
    except ValueError:
        pass

    try:
        rel = path.relative_to(CAD_JSON_OUTPUT_DIR.resolve())
        return _with_cache_buster("/cad-json-files/" + _quote_parts(rel.parts), path)
    except ValueError:
        pass

    try:
        rel = path.relative_to(CAD_STEP_OUTPUT_DIR.resolve())
        return _with_cache_buster("/step-files/" + _quote_parts(rel.parts), path)
    except ValueError:
        pass

    try:
        rel = path.relative_to(CAD_SPLIT_OUTPUT_DIR.resolve())
        return _with_cache_buster("/one-click-files/" + _quote_parts(rel.parts), path)
    except ValueError:
        pass

    raise HTTPException(status_code=500, detail=f"The file is not in an accessible static directory: {path}")


def save_json_file(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json_file(file_name: str) -> Dict[str, Any]:
    """Read a JSON file by name. The frontend usually passes jsonData directly; this is only a fallback."""
    candidates = [
        CAD_JSON_OUTPUT_DIR / file_name,
        DATA_DIR / file_name,
        STATIC_DIR / file_name,
        CAD_SPLIT_OUTPUT_DIR / file_name,
    ]
    # Per-step JSON files live in CAD_SPLIT_OUTPUT_DIR/<model-name>/xxx.json; fall back to a recursive same-name search.
    candidates.extend(CAD_SPLIT_OUTPUT_DIR.rglob(file_name))

    for path in candidates:
        if path.exists() and path.is_file():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    raise HTTPException(status_code=404, detail=f"JSON file not found: {file_name}")


def configured_json_stem() -> str:
    if not CAD_JSON_OUTPUT_FILENAME:
        return ""
    return safe_file_stem(Path(CAD_JSON_OUTPUT_FILENAME).stem)


def save_generation_record(
    session_id: str,
    requirement: str,
    cad_json: Dict[str, Any],
    raw_output: str,
    prompt_log: str,
) -> Path:
    """
    Save the complete CAD JSON generated by Qwen.
    - CAD_JSON_OUTPUT_FILENAME set: fixed file name, overwritten every time.
    - CAD_JSON_OUTPUT_FILENAME empty: generate a unique name from session + time.
    """
    if CAD_JSON_OUTPUT_FILENAME:
        file_name = Path(CAD_JSON_OUTPUT_FILENAME).name
        if not file_name.lower().endswith(".json"):
            file_name += ".json"
        stem = safe_file_stem(Path(file_name).stem)
        json_path = CAD_JSON_OUTPUT_DIR / f"{stem}.json"
    else:
        stem = safe_file_stem(f"{session_id}_{int(time.time())}_{uuid4().hex[:8]}")
        json_path = CAD_JSON_OUTPUT_DIR / f"{stem}.json"

    save_json_file(json_path, cad_json)

    debug_dir = CAD_JSON_OUTPUT_DIR / f"{json_path.stem}_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "requirement.txt").write_text(requirement, encoding="utf-8")
    (debug_dir / "prompt.txt").write_text(prompt_log, encoding="utf-8")
    (debug_dir / "raw_output.txt").write_text(raw_output, encoding="utf-8")

    return json_path


# ----------------------------- Model conversion integration: STEP / STL -----------------------------


def model_extensions() -> List[str]:
    if CAD_OUTPUT_FORMAT == "stl":
        return ["stl", "STL"]
    return ["step", "stp", "STEP", "STP"]


def default_model_ext() -> str:
    return "stl" if CAD_OUTPUT_FORMAT == "stl" else "step"


def expected_model_path_for_json(json_path_or_name: Path | str) -> Path:
    """
    Derive the expected model output path from the JSON file:
    - If CAD_MODEL_OUTPUT_FILENAME is set and the current JSON is the fixed complete-model JSON, prefer that file name.
    - Otherwise use the same-name rule: xxx.json -> xxx.stl / xxx.step.
    """
    json_path = Path(json_path_or_name)
    stem = json_path.stem
    fixed_json_stem = configured_json_stem()

    if CAD_MODEL_OUTPUT_FILENAME and fixed_json_stem and stem == fixed_json_stem:
        model_name = Path(CAD_MODEL_OUTPUT_FILENAME).name
        suffix = Path(model_name).suffix.lower().lstrip(".")
        if not suffix:
            model_name = f"{model_name}.{default_model_ext()}"
        return CAD_STEP_OUTPUT_DIR / model_name

    return CAD_STEP_OUTPUT_DIR / f"{stem}.{default_model_ext()}"


def _json_mtime_for_model_match(json_path_or_name: Path | str) -> Optional[float]:
    """
    Find the modification time of the current JSON.
    Use it to filter out stale model files so the previous model is not shown when box_lite_demo.stl was not overwritten by the converter.
    """
    raw = Path(json_path_or_name)
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([
            CAD_JSON_OUTPUT_DIR / raw.name,
            CAD_SPLIT_OUTPUT_DIR / raw.name,
            STATIC_DIR / raw.name,
        ])
        candidates.extend(CAD_SPLIT_OUTPUT_DIR.rglob(raw.name))

    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate.stat().st_mtime
        except OSError:
            continue

    return None


def find_model_file_for_json(json_path_or_name: Path | str) -> Optional[Path]:
    stem = Path(json_path_or_name).stem
    extensions = model_extensions()
    json_mtime = _json_mtime_for_model_match(json_path_or_name)

    def is_fresh_model(candidate: Path) -> bool:
        if json_mtime is None:
            return True
        try:
            # Allow a 1-second tolerance to avoid false mismatches from Windows filesystem time precision.
            return candidate.stat().st_mtime + 1 >= json_mtime
        except OSError:
            return False

    candidates: List[Path] = [expected_model_path_for_json(json_path_or_name)]
    candidates.extend(CAD_STEP_OUTPUT_DIR / f"{stem}.{ext}" for ext in extensions)

    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file() and is_fresh_model(candidate):
            return candidate.resolve()

    fuzzy: List[Path] = []
    for ext in extensions:
        fuzzy.extend(CAD_STEP_OUTPUT_DIR.glob(f"{stem}*.{ext}"))

    fuzzy = [p for p in fuzzy if p.is_file() and is_fresh_model(p)]
    if not fuzzy:
        return None

    fuzzy.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return fuzzy[0].resolve()


def trigger_json_to_model(json_path: Path) -> Optional[Path]:
    """
    Two modes:
    1. CAD_CONVERT_API_URL set: actively POST to your JSON->STL/STEP conversion endpoint.
    2. CAD_CONVERT_API_URL empty: assume a converter is watching the JSON directory; this function only waits for the model file to appear.
    """
    expected_model_path = expected_model_path_for_json(json_path)

    if CAD_CONVERT_API_URL:
        try:
            resp = requests.post(
                CAD_CONVERT_API_URL,
                json={
                    "jsonPath": str(json_path),
                    "jsonFileName": json_path.name,
                    "inputDir": str(json_path.parent),
                    "outputDir": str(CAD_STEP_OUTPUT_DIR),
                    "outputFormat": CAD_OUTPUT_FORMAT,
                    "expectedOutputPath": str(expected_model_path),
                    # Keep compatibility with older conversion-service field names.
                    "expectedStepPath": str(expected_model_path),
                    "expectedStlPath": str(expected_model_path),
                },
                timeout=CAD_CONVERT_WAIT_SECONDS,
            )
            resp.raise_for_status()

            try:
                payload = resp.json()
                returned = (
                    payload.get("outputPath")
                    or payload.get("modelPath")
                    or payload.get("stepPath")
                    or payload.get("step_path")
                    or payload.get("stlPath")
                    or payload.get("stl_path")
                )
                if returned and Path(returned).exists():
                    return Path(returned).resolve()
            except Exception:
                pass
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"Failed to call the JSON-to-model conversion API: {exc}")

    deadline = time.time() + CAD_CONVERT_WAIT_SECONDS
    while time.time() < deadline:
        model_path = find_model_file_for_json(json_path)
        if model_path:
            return model_path.resolve()
        time.sleep(CAD_CONVERT_POLL_INTERVAL)

    return None


# ----------------------------- Display fallback -----------------------------


def collect_json_stats(data: Any) -> Dict[str, Any]:
    type_count: Dict[str, int] = {}
    heights: List[float] = []
    radii: List[float] = []
    colors: List[List[Any]] = []

    def walk(obj: Any):
        if isinstance(obj, dict):
            t = obj.get("type")
            if isinstance(t, str):
                type_count[t] = type_count.get(t, 0) + 1
            params = obj.get("parameters")
            if isinstance(params, dict):
                if isinstance(params.get("height"), (int, float)):
                    heights.append(float(params["height"]))
                if isinstance(params.get("r"), (int, float)):
                    radii.append(float(params["r"]))
            color = obj.get("color")
            if isinstance(color, list) and len(color) >= 3:
                colors.append(color[:4])
            geometry = obj.get("geometry")
            if isinstance(geometry, dict):
                gcolor = geometry.get("color")
                if isinstance(gcolor, list) and len(gcolor) >= 3:
                    colors.append(gcolor[:4])
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    # Deduplicate and keep the result JSON-serializable.
    unique_colors: List[List[Any]] = []
    seen = set()
    for c in colors:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_colors.append(c)

    return {
        "typeCount": type_count,
        "heights": sorted(set(round(x, 12) for x in heights)),
        "radii": sorted(set(round(x, 12) for x in radii)),
        "colors": unique_colors,
    }


def json_preview_viewer(json_name: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "json_preview",
        "title": json_name,
        "format": "json",
        "stats": collect_json_stats(json_data),
        "message": f"No corresponding {CAD_OUTPUT_FORMAT.upper()} file has been detected yet, so the CAD JSON structure summary is displayed first.",
    }


def model_viewer_for_path(json_name: str, json_data: Dict[str, Any], model_path_or_url: Path | str) -> Dict[str, Any]:
    url = str(model_path_or_url)
    if isinstance(model_path_or_url, Path):
        url = public_url_for_file(model_path_or_url)
    suffix = Path(url.split("?")[0]).suffix.lower()
    viewer_type = "stl" if suffix == ".stl" else "step"
    stats = collect_json_stats(json_data)
    colors = stats.get("colors") or []
    return {
        "type": viewer_type,
        "title": f"{json_name} model viewer",
        "format": viewer_type,
        "url": url,
        "message": f"Currently displaying the model converted from CAD JSON{viewer_type.upper()} model.",
        "stats": stats,
        "color": colors[0] if colors else None,
    }


# ----------------------------- API -----------------------------


@app.post("/api/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    effective_requirement, used_previous_json = build_contextual_requirement(req)

    try:
        cad_json, json_text, raw_output, prompt_log = generate_cad_json(
            requirement=effective_requirement,
            history=req.history,
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Qwen failed to generate CAD JSON: {exc}")

    json_path = save_generation_record(
        session_id=req.sessionId,
        requirement=req.message,
        cad_json=cad_json,
        raw_output=raw_output,
        prompt_log=prompt_log,
    )

    SESSION_MEMORY[req.sessionId] = {
        "requirement": req.message,
        "cad_json": cad_json,
        "json_path": str(json_path),
        "updated_at": time.time(),
    }

    model_path = trigger_json_to_model(json_path)
    json_url = public_url_for_file(json_path)
    model_url = public_url_for_file(model_path) if model_path else ""

    if used_previous_json:
        reply = f"Modifications have been completed based on the previous model, and the results have been generated. CAD JSON: {json_path.name}\nSaved to: {CAD_JSON_OUTPUT_DIR}."
    else:
        reply = f"Generated CAD JSON: {json_path.name}\nSaved to: {CAD_JSON_OUTPUT_DIR}."
    if model_path:
        reply += f"\nDetected {model_path.suffix.upper().lstrip('.')} File: {model_path.name}."
    else:
        reply += (
            f"\nNo corresponding {CAD_OUTPUT_FORMAT.upper()} file has been detected yet; "
            "If you are using an external conversion program, please make sure it is monitoring the JSON output directory and writing the model file with the same name into the model output directory."
        )

    return {
        "reply": reply,
        "jsonItem": {
            "id": json_path.stem,
            "name": json_path.name,
            "displayName": "Full CAD Model.json",
            "title": "Full CAD JSON",
            "data": cad_json,
            "url": json_url,
            "stepUrl": model_url,
            "stepName": model_path.name if model_path else "",
            "source": "qwen",
        },
    }


@app.post("/api/select-json")
def select_json(req: SelectJsonRequest) -> Dict[str, Any]:
    json_data = req.jsonData or read_json_file(req.name)

    # Items returned by one-click generation may already carry stepUrl/stlUrl; prefer those.
    if req.stepUrl:
        viewer = model_viewer_for_path(req.name, json_data, req.stepUrl)
        return {"selected": {"name": req.name, "data": json_data}, "viewer": viewer}

    model_path = find_model_file_for_json(req.name)
    if model_path:
        viewer = model_viewer_for_path(req.name, json_data, model_path)
    else:
        viewer = json_preview_viewer(req.name, json_data)

    return {"selected": {"name": req.name, "data": json_data}, "viewer": viewer}


@app.post("/api/generate")
def generate_assets(req: GenerateRequest) -> Dict[str, Any]:
    """
    One-click generation:
    1. Find the most recent complete CAD JSON generated by Qwen;
    2. Call split_cad_json_dataset_ollama_html_steps.py for step splitting;
    3. Return the complete JSON, per-step JSON, and detailed step-explanation HTML;
    4. Optional: trigger STL/STEP conversion.
    """
    if req.finalJsonData:
        stem = safe_file_stem(Path(req.finalJsonName or f"{req.sessionId}_manual.json").stem)
        json_path = CAD_JSON_OUTPUT_DIR / f"{stem}.json"
        save_json_file(json_path, req.finalJsonData)
    else:
        candidates = sorted(
            [p for p in CAD_JSON_OUTPUT_DIR.glob("*.json") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise HTTPException(status_code=404, detail="No CAD JSON is available for one-click generation yet. Please send the modeling requirements first.")
        json_path = candidates[0]

    try:
        result = split_file(
            json_path,
            CAD_SPLIT_OUTPUT_DIR,
            model=CAD_SPLIT_MODEL,
            ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate"),
            use_ollama=CAD_SPLIT_USE_LLM,
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"One-click generation step splitting failed: {exc}")

    full_json_path = Path(result["full_model_json"])
    with full_json_path.open("r", encoding="utf-8") as f:
        full_data = json.load(f)

    full_model_path = trigger_json_to_model(full_json_path)
    full_model_url = public_url_for_file(full_model_path) if full_model_path else ""

    step_json_items: List[Dict[str, Any]] = []
    output_dir = Path(result["output_dir"])

    for item in result.get("step_json_files", []):
        step_json_path = output_dir / item["file_name"]
        if not step_json_path.exists():
            continue

        with step_json_path.open("r", encoding="utf-8") as f:
            step_data = json.load(f)

        if CAD_CONVERT_SPLIT_JSONS:
            step_model_path = trigger_json_to_model(step_json_path)
        else:
            step_model_path = find_model_file_for_json(step_json_path)

        step_model_url = public_url_for_file(step_model_path) if step_model_path else ""

        step_json_items.append(
            {
                "id": f"{output_dir.name}_{step_json_path.stem}",
                "name": step_json_path.name,
                "displayName": item.get("step_title", step_json_path.stem) + ".json",
                "title": item.get("step_title", step_json_path.stem),
                "data": step_data,
                "url": public_url_for_file(step_json_path),
                "stepUrl": step_model_url,
                "stepName": step_model_path.name if step_model_path else "",
                "source": "split",
                "colorRgba": item.get("color_rgba"),
                "candidateIndices": item.get("candidate_indices", []),
            }
        )

    html_path = Path(result["step_explanation_html"])
    html_url = public_url_for_file(html_path) if html_path.exists() else ""

    return {
        "fullJson": {
            "id": full_json_path.stem,
            "name": full_json_path.name,
            "displayName": "Full CAD Model.json",
            "title": "Full CAD JSON",
            "data": full_data,
            "url": public_url_for_file(full_json_path),
            "stepUrl": full_model_url,
            "stepName": full_model_path.name if full_model_path else "",
            "source": "split_full",
        },
        "stepJsons": step_json_items,
        "stepImages": [],
        "htmlDocUrl": html_url,
        "stepFile": {
            "name": full_model_path.name,
            "url": full_model_url,
            "format": full_model_path.suffix.lower().lstrip("."),
        } if full_model_path else None,
    }


@app.get("/api/bootstrap")
def bootstrap() -> Dict[str, Any]:
    return {
        "ok": True,
        "mode": "qwen-json-to-model",
        "cadJsonOutputDir": str(CAD_JSON_OUTPUT_DIR),
        "cadModelOutputDir": str(CAD_STEP_OUTPUT_DIR),
        "cadOutputFormat": CAD_OUTPUT_FORMAT,
        "cadJsonOutputFilename": CAD_JSON_OUTPUT_FILENAME,
        "cadModelOutputFilename": CAD_MODEL_OUTPUT_FILENAME,
        "cadSplitOutputDir": str(CAD_SPLIT_OUTPUT_DIR),
        "convertApiConfigured": bool(CAD_CONVERT_API_URL),
        "splitUseLlm": CAD_SPLIT_USE_LLM,
        "splitModel": CAD_SPLIT_MODEL,
    }


@app.get("/api/list-files")
def list_existing_files() -> Dict[str, Any]:
    """List JSON files already generated in CAD_JSON_OUTPUT_DIR so the frontend can show them directly."""
    items: List[Dict[str, Any]] = []

    for index, json_path in enumerate(sorted(CAD_JSON_OUTPUT_DIR.glob("*.json")), start=1):
        if not json_path.is_file():
            continue

        if json_path.name.isascii():
            display_name = json_path.name
        else:
            display_name = f"Existing Model {index}.json"

        model_path = find_model_file_for_json(json_path)
        items.append({
            "id": f"existing-full-{json_path.stem}",
            "name": json_path.name,
            "displayName": display_name,
            "type": "json",
            "url": public_url_for_file(json_path),
            "stepUrl": public_url_for_file(model_path) if model_path else "",
            "source": "existing_full",
        })

    return {"files": items}

