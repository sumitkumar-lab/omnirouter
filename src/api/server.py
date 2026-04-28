from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from src.agent.graph import app as agent_app
from src.api.cache import check_cache, save_to_cache
from src.rag.pipeline import sync_corpus
from src.rag.settings import get_rag_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = PROJECT_ROOT / "web"

app = FastAPI(
    title="OmniRouter AI Research Scientist API",
    description="Streaming FastAPI gateway for the LangGraph research swarm.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class ChatRequest(BaseModel):
    query: str
    thread_id: str | None = None
    use_cache: bool = True


class UploadedDocument(BaseModel):
    filename: str
    path: str
    size_bytes: int


class UploadResponse(BaseModel):
    uploaded: list[UploadedDocument]
    rebuilt: bool
    version: str | None
    source_count: int
    chunk_count: int


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "omnirouter-research-agent"}


@app.get("/")
async def web_index():
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "OmniRouter API is running. Add web/index.html to enable the browser UI."}


@app.post("/chat/stream")
async def chat_streaming_endpoint(request: ChatRequest):
    return StreamingResponse(
        stream_generator(
            query=request.query,
            thread_id=request.thread_id,
            use_cache=request.use_cache,
        ),
        media_type="text/event-stream",
    )


@app.get("/documents")
async def list_documents():
    settings = get_rag_settings()
    settings.data_lake_dir.mkdir(parents=True, exist_ok=True)
    documents = []
    for path in sorted(settings.data_lake_dir.rglob("*")):
        if not path.is_file() or path.name == "README.md":
            continue
        if path.suffix.lower() not in settings.supported_extensions:
            continue
        documents.append(
            {
                "filename": path.name,
                "path": path.relative_to(settings.project_root).as_posix(),
                "size_bytes": path.stat().st_size,
            }
        )
    return {"documents": documents}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    sync: bool = Form(default=True),
):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one document.")

    settings = get_rag_settings()
    settings.data_lake_dir.mkdir(parents=True, exist_ok=True)
    uploaded: list[UploadedDocument] = []

    for file in files:
        safe_name = _safe_upload_filename(file.filename or "")
        extension = Path(safe_name).suffix.lower()
        if extension not in settings.supported_extensions:
            allowed = ", ".join(settings.supported_extensions)
            raise HTTPException(status_code=400, detail=f"Unsupported file type '{extension}'. Allowed: {allowed}")

        destination = (settings.data_lake_dir / safe_name).resolve()
        _ensure_inside_directory(destination, settings.data_lake_dir.resolve())

        size_bytes = 0
        with destination.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size_bytes += len(chunk)
                handle.write(chunk)
        await file.close()

        uploaded.append(
            UploadedDocument(
                filename=safe_name,
                path=destination.relative_to(settings.project_root).as_posix(),
                size_bytes=size_bytes,
            )
        )

    result = sync_corpus(settings=settings, force=True) if sync else None
    return UploadResponse(
        uploaded=uploaded,
        rebuilt=bool(result.rebuilt) if result is not None else False,
        version=result.version_label if result is not None else None,
        source_count=result.source_count if result is not None else len(uploaded),
        chunk_count=result.chunk_count if result is not None else 0,
    )


async def stream_generator(query: str, thread_id: str | None = None, use_cache: bool = True):
    if use_cache:
        cached_answer = check_cache(query)
        if cached_answer:
            yield _sse("meta", {"source": "cache"})
            for word in cached_answer.split(" "):
                yield _sse("token", {"token": word + " "})
                await asyncio.sleep(0.02)
            yield _sse("done", {"cached": True})
            return

    initial_state = {"messages": [HumanMessage(content=query)]}
    full_answer = ""
    config = {"configurable": {"thread_id": thread_id or f"web_{uuid4().hex}"}}

    yield _sse("meta", {"source": "agent", "thread_id": config["configurable"]["thread_id"]})

    async for event in agent_app.astream_events(initial_state, config, version="v1"):
        if event["event"] != "on_chat_model_stream":
            continue

        content = event["data"]["chunk"].content
        if content:
            full_answer += content
            yield _sse("token", {"token": content})

    failure_phrase = "I do not have enough information"
    if use_cache and full_answer and failure_phrase not in full_answer:
        save_to_cache(query, full_answer)
    yield _sse("done", {"cached": False})


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _safe_upload_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    safe_name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    if not safe_name or not Path(safe_name).suffix:
        raise HTTPException(status_code=400, detail="Uploaded file must have a supported extension.")
    return safe_name


def _ensure_inside_directory(path: Path, directory: Path) -> None:
    try:
        path.relative_to(directory)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid upload path.") from exc
