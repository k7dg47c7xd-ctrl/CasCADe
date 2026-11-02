# CAD Chat Demo Usage Guide

## 1. Main flow

```text
user modeling request
|
POST /api/chat
|
configured CAD-JSON provider (CAD_JSON_LLM_PROVIDER; finetuned by default, dashscope as an alternative)
|
generate Cadmium-lite CAD JSON
|
normalize JSON
|
save to CAD_JSON_OUTPUT_DIR / CAD_JSON_OUTPUT_FILENAME
|
call CAD_CONVERT_API_URL actively, or wait for a converter that watches the JSON directory
|
look for the STL/STEP model in CAD_STEP_OUTPUT_DIR
|
return stepUrl to the frontend
|
frontend shows the STL/STEP when a JSON card is clicked; "One-click generation" outputs step JSONs and HTML explanations
```

## 2. Configure backend/.env

Open `backend/.env` (start from `backend/.env.example`) and at least set:

```env
DASHSCOPE_API_KEY=<your Bailian API key>
```

The template ships with:

```env
QWEN_MODEL=qwen3.7-max
CAD_OUTPUT_FORMAT=stl
CAD_JSON_OUTPUT_DIR=./static/cad_json_in
CAD_JSON_OUTPUT_FILENAME=box_lite_demo.json
CAD_STEP_OUTPUT_DIR=./static/cad_step_out
CAD_MODEL_OUTPUT_FILENAME=box_lite_demo.stl
CAD_SPLIT_OUTPUT_DIR=./static/one_click
CAD_SPLIT_LLM_PROVIDER=finetuned
FINETUNED_MODEL=<model id exposed by your inference service>
```

Notes:

- `CAD_JSON_OUTPUT_DIR` is the JSON output directory; do not point it at a concrete `xxx.json` file.
- `CAD_STEP_OUTPUT_DIR` is the model output directory; despite the STEP in its name it can also hold STL files.
- `CAD_JSON_OUTPUT_FILENAME` is the complete-JSON file name. When empty, the backend generates a unique name from session + time.
- `CAD_MODEL_OUTPUT_FILENAME` is the complete STL/STEP file name. When empty, the backend falls back to the same-stem rule.
- For STEP output set `CAD_OUTPUT_FORMAT=step` and `CAD_MODEL_OUTPUT_FILENAME=box_lite_demo.step`.
- `CAD_JSON_LLM_PROVIDER` defaults to `finetuned` (through `backend/finetuned_model_client.py`); set it to `dashscope` to use the DashScope OpenAI-compatible endpoint with `QWEN_MODEL`.
- `CAD_SPLIT_LLM_PROVIDER` controls the teaching-text model used by the one-click splitting script: `finetuned`, `qwen`, or legacy `ollama`.

## 3. Start the backend

The frontend dev server proxies to `http://localhost:8002`, so run the backend on port 8002:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8002 --reload
```

Successful startup shows:

```text
Application startup complete.
Uvicorn running on http://0.0.0.0:8002
```

## 4. Start the frontend

In a second PowerShell window:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## 5. Connecting a JSON -> STL/STEP converter

### Option A: backend actively calls the converter

If your converter endpoint is `http://127.0.0.1:9000/convert`, set:

```env
CAD_CONVERT_API_URL=http://127.0.0.1:9000/convert
```

The backend POSTs a JSON payload containing `jsonPath`, `jsonFileName`, `inputDir`, `outputDir`, `outputFormat`, `expectedOutputPath`, `expectedStepPath`, and `expectedStlPath`. The converter can write the model to `expectedOutputPath`, or return:

```json
{"outputPath": ".../static/cad_step_out/box_lite_demo.stl"}
```

### Option B: the converter watches the directory

Leave `CAD_CONVERT_API_URL=` empty; the backend then only waits for the model file to appear.

Default JSON output location:

```text
backend/static/cad_json_in/box_lite_demo.json
```

Default model output location:

```text
backend/static/cad_step_out/box_lite_demo.stl
```

The converter watches `backend/static/cad_json_in`, and after `box_lite_demo.json` appears it writes the same-named (or configured) model file into `backend/static/cad_step_out`.

## 6. How one-click generation works

Clicking "One-click generation" (`POST /api/generate`) invokes:

```text
backend/split_cad_json_dataset_ollama_html_steps.py
```

and writes into `backend/static/one_click/box_lite_demo/`:

```text
box_lite_demo.json
01_xxx.json
02_xxx.json
Detailed_Step_Explanation.html
```

The right panel shows the complete JSON, the step JSONs, and the HTML. Clicking the HTML opens the step explanation; clicking a JSON shows the matching STL/STEP, or a JSON summary when no model file exists.

## 7. FAQ

### The frontend shows "request failed"

Read the error details shown on the page or in the backend terminal; the frontend displays the backend `detail` field.

### The configured model call fails

Common causes:

- `DASHSCOPE_API_KEY` is missing (when using the DashScope provider);
- the API key is invalid, expired, or out of balance;
- the account lacks access to the configured model id;
- the machine cannot reach the DashScope OpenAI-compatible endpoint;
- `QWEN_ENABLE_THINKING=true`, so the model returns only reasoning content;
- `CAD_JSON_LLM_PROVIDER=finetuned` without a reachable fine-tuned inference service (check `FINETUNED_API_BASE_URL` and `FINETUNED_MODEL`).

Keep:

```env
QWEN_ENABLE_THINKING=false
```

### `npm run dev` only shows "vite --host ..."

The dev script is:

```json
"dev": "vite --host 127.0.0.1 --port 5173 --clearScreen false"
```

If it still hangs, delete `node_modules` and run `npm install` again.

## Avoiding stale models

When you reuse fixed names such as `box_lite_demo.json` / `box_lite_demo.stl`, the converter must overwrite the same-named model file after every JSON generation. The backend filters out model files whose modification time is older than the current JSON so the previous STL/STEP is not shown.

- STL: `CAD_OUTPUT_FORMAT=stl` with `CAD_MODEL_OUTPUT_FILENAME=box_lite_demo.stl`.
- STEP: `CAD_OUTPUT_FORMAT=step` with `CAD_MODEL_OUTPUT_FILENAME=box_lite_demo.step`.
- The model directory is `CAD_STEP_OUTPUT_DIR`; the name contains STEP, but STL files are fine there.
- When the model URL file name does not change, the backend appends `?v=<file mtime>` to prevent the browser from caching the old model.

## STEP load failure: "expected magic word"

If the page shows:

```txt
WebAssembly.instantiate(): expected magic word 00 61 73 6d, found 3c 21 64 6f
```

this is not a STEP parsing error. It means the `.wasm` file of the frontend STEP parser `occt-import-js` was not loaded correctly; `3c 21 64 6f` is the text `<!do`, i.e. the browser received an HTML page instead of the `.wasm` asset.

`frontend/src/components/StepViewer.jsx` imports the wasm explicitly through Vite's `?url`:

```js
import occtWasmUrl from "occt-import-js/dist/occt-import-js.wasm?url";
```

and passes it to `occtimportjs({ locateFile })`, so Vite bundles and serves the `.wasm` as a static asset. Restart the frontend and hard-refresh with `Ctrl + F5`.

## v4 fixes: multi-turn memory and frontend overwrite

This version adds two key fixes:

1. `/api/chat` remembers the last complete CAD JSON generated in the current browser session.
   - On a second message such as "add a 10x10x10 cube on top of this cylinder", the backend sends the previous complete JSON along with the request.
   - The model is expected to return the modified complete CAD JSON, not just the new part.
   - In-memory state is cleared on backend restart; if the frontend history already contains a generated CAD JSON, the backend tries to recover the previous JSON from the fixed JSON output file.

2. The right-side JSON list overwrites the previous conversation result.
   - Each "Send" keeps only the latest `box_lite_demo.json` in the list.
   - The center preview switches to the newest model/JSON instead of showing the previous preview.
   - "One-click generation" still shows the complete JSON, step JSONs, and the HTML explanation, and it also overwrites the previous one-click result.

If you use a fixed model file name such as `model.step` or `box_lite_demo.stl`, the converter must really overwrite that file after every JSON generation. The backend appends a cache parameter to the model URL so the browser does not keep reading the old model.
