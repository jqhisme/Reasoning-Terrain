"""FastAPI backend for live embedding and UMAP projection of Codex events."""

from __future__ import annotations

import asyncio
import inspect
import re
import threading
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

from agent_service import SANDBOXES, run_codex_task


HERE = Path(__file__).resolve().parent
FRONTEND = HERE / "frontend" / "index.html"
DEFAULT_WORKSPACE = HERE.parent
OUTPUT_DIR = HERE / "runs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CATEGORIES = {
    "reasoning_summary",
    "plan",
    "progress_update",
    # Temporarily disabled. Uncomment when tool-call points return to the UI.
    # "tool_call",
    "function_result",
}


class TaskCreate(BaseModel):
    prompt: str
    task_id: str | None = None


class EventCreate(BaseModel):
    task_id: str
    category: str
    text: str = Field(min_length=1)


class FinalAnswer(BaseModel):
    task_id: str
    text: str


class RunCreate(BaseModel):
    prompt: str = Field(min_length=1)
    task_id: str | None = None
    model: str | None = None
    provider: str = "openai"
    sandbox: str = "workspace-write"
    network_enabled: bool = True


class Embedder:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        output = self.model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
        pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)


class Store:
    def __init__(self) -> None:
        self.embedder: Embedder | None = None
        self.tasks: dict[str, dict[str, Any]] = {}
        self.clients: dict[str, set[WebSocket]] = defaultdict(set)
        self.lock = threading.Lock()
        self.umap_class: Any = None
        self.umap_status = "warming"

    def public_state(self, task_id: str) -> dict[str, Any]:
        task = self.tasks[task_id]
        return {
            "task_id": task_id,
            "prompt": task["prompt"],
            "final_answer": task["final_answer"],
            "projection": task["projection"],
            "status": task["status"],
            "error": task["error"],
            "turn_count": task["turn_count"],
            "points": [
                {key: value for key, value in point.items() if key != "embedding"}
                for point in task["points"]
            ],
        }

    def public_all_state(self) -> dict[str, Any]:
        points = []
        latest_answer = None
        projection = "none"
        for task in self.tasks.values():
            points.extend(
                {key: value for key, value in point.items() if key != "embedding"}
                for point in task["points"]
            )
            latest_answer = task["final_answer"] or latest_answer
            if task["projection"] != "none":
                projection = task["projection"]
        return {
            "task_id": "all",
            "prompt": "All tasks",
            "final_answer": latest_answer,
            "projection": projection,
            "status": "aggregate",
            "error": None,
            "turn_count": sum(task["turn_count"] for task in self.tasks.values()),
            "points": points,
        }


store = Store()


def split_text(text: str, max_chars: int = 420) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            while len(sentence) > max_chars:
                chunks.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def project(points: list[dict[str, Any]]) -> str:
    count = len(points)
    if count == 0:
        return "none"
    if count == 1:
        coordinates = np.array([[0.0, 0.0]], dtype=np.float32)
        method = "single-point"
    else:
        vectors = np.vstack([point["embedding"] for point in points])
        if count < 4:
            components = min(2, count, vectors.shape[1])
            coordinates = PCA(n_components=components).fit_transform(vectors)
            if coordinates.shape[1] == 1:
                coordinates = np.column_stack([coordinates[:, 0], np.zeros(count)])
            method = "pca-warmup"
        elif store.umap_class is None:
            components = min(2, count, vectors.shape[1])
            coordinates = PCA(n_components=components).fit_transform(vectors)
            if coordinates.shape[1] == 1:
                coordinates = np.column_stack([coordinates[:, 0], np.zeros(count)])
            method = "pca-while-umap-warms"
        else:
            reducer = store.umap_class(
                n_components=2,
                n_neighbors=min(10, count - 1),
                min_dist=0.18,
                metric="cosine",
                random_state=42,
                n_jobs=1,
                init="random",
            )
            coordinates = reducer.fit_transform(vectors)
            method = "umap"
    for point, (x, y) in zip(points, coordinates):
        point["x"] = float(x)
        point["y"] = float(y)
    return method


async def broadcast(task_id: str) -> None:
    for channel, message in (
        (task_id, store.public_state(task_id)),
        ("all", store.public_all_state()),
    ):
        stale: list[WebSocket] = []
        for websocket in list(store.clients[channel]):
            try:
                await websocket.send_json(message)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            store.clients[channel].discard(websocket)


def _initialize_umap() -> Any:
    """Import and JIT-warm UMAP off the request path (slow once on Windows)."""
    import umap

    # umap-learn < 0.5.10 passes the removed ``force_all_finite`` keyword to
    # scikit-learn >= 1.8. Keep the app working in an existing environment even
    # before its pinned umap-learn version is upgraded.
    sklearn_check_array = umap.umap_.check_array
    if "force_all_finite" not in inspect.signature(sklearn_check_array).parameters:
        def compatible_check_array(
            *args: Any,
            force_all_finite: bool | str | None = None,
            **kwargs: Any,
        ) -> Any:
            if force_all_finite is not None:
                kwargs["ensure_all_finite"] = force_all_finite
            return sklearn_check_array(*args, **kwargs)

        umap.umap_.check_array = compatible_check_array

    warmup = np.eye(8, dtype=np.float32)
    umap.UMAP(
        n_components=2,
        n_neighbors=5,
        min_dist=0.18,
        metric="cosine",
        random_state=42,
        n_jobs=1,
        init="random",
    ).fit_transform(warmup)
    return umap.UMAP


async def warm_umap() -> None:
    try:
        umap_class = await asyncio.to_thread(_initialize_umap)
        with store.lock:
            store.umap_class = umap_class
            store.umap_status = "ready"
            all_points = [
                point for task in store.tasks.values() for point in task["points"]
            ]
            if all_points:
                method = project(all_points)
                for task in store.tasks.values():
                    task["projection"] = method
        for task_id in list(store.tasks):
            await broadcast(task_id)
    except Exception as exc:
        store.umap_status = f"error: {exc}"


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_dotenv(HERE.parent / ".env")
    store.embedder = await asyncio.to_thread(Embedder)
    warmup_task = asyncio.create_task(warm_umap())
    yield
    warmup_task.cancel()


app = FastAPI(title="Reasoning Landscape", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "frontend"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": store.embedder is not None,
        "model": MODEL_NAME,
        "device": str(store.embedder.device) if store.embedder else "loading",
        "umap": store.umap_status,
    }


@app.post("/api/tasks")
async def create_task(request: TaskCreate) -> dict[str, Any]:
    task_id = request.task_id or uuid.uuid4().hex[:12]
    with store.lock:
        store.tasks[task_id] = {
            "prompt": request.prompt,
            "points": [],
            "final_answer": None,
            "projection": "none",
            "status": "created",
            "error": None,
            "turn_count": 0,
            "codex_thread_id": None,
        }
    await broadcast(task_id)
    return store.public_state(task_id)


async def ingest_event(task_id: str, category: str, text: str) -> None:
    """Embed, project, and broadcast one normalized agent event."""
    chunks = split_text(text)
    if not chunks:
        return
    assert store.embedder is not None
    embeddings = await asyncio.to_thread(store.embedder.encode, chunks)
    with store.lock:
        points = store.tasks[task_id]["points"]
        for chunk, embedding in zip(chunks, embeddings):
            points.append({
                "id": uuid.uuid4().hex,
                "task_id": task_id,
                "category": category,
                "text": chunk,
                "embedding": embedding,
                "x": 0.0,
                "y": 0.0,
            })
        all_points = [point for task in store.tasks.values() for point in task["points"]]
        method = project(all_points)
        for task in store.tasks.values():
            task["projection"] = method
    await broadcast(task_id)


async def execute_run(task_id: str, request: RunCreate) -> None:
    loop = asyncio.get_running_loop()
    store.tasks[task_id]["status"] = "running"
    await broadcast(task_id)

    def publish(category: str, text: str) -> None:
        future = asyncio.run_coroutine_threadsafe(
            ingest_event(task_id, category, text), loop
        )
        future.result()

    try:
        answer, thread_id = await asyncio.to_thread(
            run_codex_task,
            task_id=task_id,
            prompt=request.prompt,
            workspace=DEFAULT_WORKSPACE,
            output_dir=OUTPUT_DIR,
            publish=publish,
            model=request.model,
            provider=request.provider,
            sandbox=request.sandbox,
            network_enabled=request.network_enabled,
            thread_id=store.tasks[task_id]["codex_thread_id"],
        )
        store.tasks[task_id]["final_answer"] = answer
        store.tasks[task_id]["codex_thread_id"] = thread_id
        store.tasks[task_id]["turn_count"] += 1
        store.tasks[task_id]["status"] = "completed"
    except Exception as exc:
        store.tasks[task_id]["status"] = "failed"
        store.tasks[task_id]["error"] = str(exc)
    await broadcast(task_id)


@app.post("/api/runs", status_code=202)
async def start_run(request: RunCreate) -> dict[str, Any]:
    if request.provider not in {"openai", "ollama"}:
        raise HTTPException(422, "Provider must be openai or ollama")
    if request.sandbox not in SANDBOXES:
        raise HTTPException(422, f"Unknown sandbox: {request.sandbox}")
    task_id = request.task_id or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    with store.lock:
        if task_id not in store.tasks:
            store.tasks[task_id] = {
                "prompt": request.prompt.strip(),
                "points": [],
                "final_answer": None,
                "projection": "none",
                "status": "queued",
                "error": None,
                "turn_count": 0,
                "codex_thread_id": None,
            }
        elif store.tasks[task_id]["status"] in {"queued", "running"}:
            raise HTTPException(409, "This session already has a turn running")
        else:
            store.tasks[task_id]["status"] = "queued"
            store.tasks[task_id]["error"] = None
            store.tasks[task_id]["prompt"] = request.prompt.strip()
    asyncio.create_task(execute_run(task_id, request))
    return store.public_state(task_id)


@app.post("/api/sessions", status_code=201)
async def new_session() -> dict[str, Any]:
    task_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    with store.lock:
        if any(
            task["status"] in {"queued", "running"}
            for task in store.tasks.values()
        ):
            raise HTTPException(409, "Wait for the current turn to finish")
        store.tasks.clear()
        store.tasks[task_id] = {
            "prompt": "",
            "points": [],
            "final_answer": None,
            "projection": "none",
            "status": "created",
            "error": None,
            "turn_count": 0,
            "codex_thread_id": None,
        }
    await broadcast(task_id)
    return store.public_state(task_id)


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    if task_id == "all":
        return store.public_all_state()
    if task_id not in store.tasks:
        raise HTTPException(404, "Task not found")
    return store.public_state(task_id)


@app.post("/api/events")
async def add_event(event: EventCreate) -> dict[str, Any]:
    if event.category not in CATEGORIES:
        raise HTTPException(422, f"Unknown category: {event.category}")
    if event.task_id not in store.tasks:
        raise HTTPException(404, "Task not found")
    await ingest_event(event.task_id, event.category, event.text)
    return store.public_state(event.task_id)


@app.post("/api/final-answer")
async def set_final_answer(answer: FinalAnswer) -> dict[str, Any]:
    if answer.task_id not in store.tasks:
        raise HTTPException(404, "Task not found")
    store.tasks[answer.task_id]["final_answer"] = answer.text
    store.tasks[answer.task_id]["status"] = "completed"
    await broadcast(answer.task_id)
    return store.public_state(answer.task_id)


@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str) -> None:
    await websocket.accept()
    store.clients[task_id].add(websocket)
    if task_id == "all":
        await websocket.send_json(store.public_all_state())
    elif task_id in store.tasks:
        await websocket.send_json(store.public_state(task_id))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        store.clients[task_id].discard(websocket)
