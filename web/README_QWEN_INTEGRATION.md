# CAD Chat Demo: Qwen -> CAD JSON -> STEP Integration

This repository adapts the original demo flow to:

```text
natural-language modeling request
  -> backend calls the configured CAD-JSON provider
     (CAD_JSON_LLM_PROVIDER=finetuned by default; dashscope as an alternative)
  -> generate lite CAD JSON
  -> save to the configured JSON input directory
  -> call or wait for your JSON -> STEP/STL converter
  -> read the STEP/STL model from the configured model output directory
  -> frontend displays the model
```

## Scripts kept and removed

Kept (trimmed down):

- The dynamic prompt logic for lite JSON generation from `dataset_prompts.py`, reduced to `backend/qwen_prompts.py`.
- The Qwen OpenAI-compatible call flow from `chat_modified_qwen.py` (stream-first with non-streaming fallback, `enable_thinking=false`, JSON extraction, and the repair approach), reduced to `backend/qwen_cad_client.py`.
- Template field order and group/parameters/coordinates normalization from `cadjson_template_framework.py`, kept in `backend/cadjson_template_framework.py`.

Removed:

- Scoring flows from `Rating_modified.py` and `Rating_raw_structure.py`.
- The Gradio debug UI.
- Fixed simple/medium/complex folder batch evaluation.
- dataset.jsonl batch evaluation.
- Reference-JSON comparison logic.

## Backend configuration

Copy the environment template:

```bash
cd backend
cp .env.example .env
```

Edit `.env`:

```env
DASHSCOPE_API_KEY=<your API key>
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.7-max
QWEN_ENABLE_THINKING=false
QWEN_NORMALIZE_OUTPUT=true

CAD_JSON_OUTPUT_DIR=./static/cad_json_in
CAD_STEP_OUTPUT_DIR=./static/cad_step_out

# If your converter must be called actively, set its URL; if it watches the JSON directory on its own, leave this empty.
CAD_CONVERT_API_URL=
CAD_CONVERT_WAIT_SECONDS=120
```

`CAD_JSON_LLM_PROVIDER` defaults to `finetuned`, which routes `/api/chat` through `backend/finetuned_model_client.py` (an OpenAI-compatible inference service for the fine-tuned checkpoint). Set it to `dashscope` to call the DashScope OpenAI-compatible endpoint instead; the `QWEN_*` settings then take effect.

## Backend startup

The Vite dev server proxies `/api` and the static file routes to `http://localhost:8002`, so start the backend on port 8002:

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8002 --reload
```

## Frontend startup

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## JSON -> STEP/STL file-name convention

The backend writes:

```text
backend/static/cad_json_in/<stem>.json
```

Your converter should write a same-named model file:

```text
backend/static/cad_step_out/<stem>.step
```

Depending on `CAD_OUTPUT_FORMAT` (default `stl`), the lookup prefers `.stl` or `.step/.stp`, and also accepts a fuzzy match such as `<stem>*.step` / `<stem>*.stl`. If your converter uses a different naming rule, adjust `find_model_file_for_json()` in `backend/app.py`.

## Main endpoints

### POST `/api/chat`

Request:

```json
{
  "sessionId": "session_001",
  "message": "Generate a rectangular plate with a circular hole",
  "history": []
}
```

When the response contains a non-empty `jsonItem.stepUrl`, the frontend displays the STEP/STL model; otherwise it displays a JSON summary.

### POST `/api/select-json`

Called when a JSON card is clicked. The backend looks for the same-named model file and, when found, returns:

```json
{
  "viewer": {
    "type": "step",
    "url": "/static/cad_step_out/xxx.step"
  }
}
```

### POST `/api/generate`

One-click generation: splits the final JSON into step JSONs and generates the detailed step explanation HTML through `backend/split_cad_json_dataset_ollama_html_steps.py`, writing into `CAD_SPLIT_OUTPUT_DIR`.

### GET `/api/bootstrap` and GET `/api/list-files`

Return session state and the existing JSON/model file listings used to populate the frontend cards.

## Related guide

See `RUN_GUIDE.md` for a step-by-step walkthrough covering converter integration, one-click generation, and troubleshooting.

> Reproducibility note: this demo requires external services (a fine-tuned model inference service or DashScope access, plus a JSON-to-STEP/STL converter). The repository ships the integration code, not the model weights, datasets, or converter.
