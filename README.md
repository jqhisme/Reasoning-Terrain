# Reasoning Terrian

A live semantic visualization of a Codex run. The runner currently keeps four
natural-language event categories:

- `reasoning_summary`
- `plan`
- `progress_update`
- `function_result`

`tool_call` events are temporarily commented out in the runner and UI. The
completed results of web searches and other tools still appear as
`function_result` points.

The final answer is stored separately and displayed beside the visualization.

## Architecture

1. `agent_service.py` runs Codex inside the FastAPI process and normalizes completed items.
2. `event_formatter.py` converts structured tool calls/results into readable text.
3. `backend.py` splits text into sentence-sized chunks, embeds it with
   `sentence-transformers/all-MiniLM-L6-v2`, and refits a shared two-dimensional
   UMAP projection whenever new chunks arrive.
4. FastAPI broadcasts the current state over WebSocket.
5. `frontend/index.html` renders the points and D3 KDE contours in real time.

UMAP is provided by the separate `umap-learn` package; it is compatible with
scikit-learn but is not included in scikit-learn itself. With fewer than four
points, the backend uses a PCA warm-up projection because UMAP is not defined
reliably for such a small sample.

## Install

```powershell
conda run -n my_env python -m pip install -r reasoning-viz/requirements.txt
```

The first backend start downloads the lightweight MiniLM model. It uses CUDA
automatically when PyTorch detects it and otherwise falls back to CPU.

## Run

Start the backend from the repository root:

```powershell
conda run -n my_env python -m uvicorn backend:app `
  --app-dir reasoning-viz `
  --host 127.0.0.1 `
  --port 8000
```

Open the application:

```text
http://127.0.0.1:8000/
```

Enter a question in the prompt box and choose **Run**. The backend returns
a task ID immediately, runs Codex in a worker thread, and streams events to the
page while keeping the API responsive. Each run still writes its normalized
event log and final answer under `reasoning-viz/runs/`.

You can also start a run directly through the API:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/runs `
  -ContentType application/json `
  -Body '{"prompt":"Read Cities.txt and summarize it."}'
```

The original CLI remains available for batch runs and can run without the backend:

```powershell
conda run -n my_env python reasoning-viz/run_codex_agent.py `
  --no-backend `
  "Your task"
```

## Visualization

- Color identifies the four active event categories.
- Hovering previews a point; clicking it opens the full text panel.
- At eight total points, KDE contours appear.
- KDE can be grouped across all points or separately by task, with outline,
  fill, and smooth-distribution display styles.
- Dragging pans the plot and scrolling zooms it.
- The final answer appears in the right-hand panel.

The backend stores data in memory for the life of the server. Restarting it
clears all tasks. Coordinates are recomputed as points arrive, so existing
points can move; that movement reflects the newly refitted shared UMAP space.
