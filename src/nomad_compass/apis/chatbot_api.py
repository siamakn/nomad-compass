from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from nomad.config import config
from pydantic import BaseModel

from .rag_core import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_RDF_ZIP,
    OLLAMA_BASE_URL,
    SYSTEM_PROMPT,
    chunk_text,
    download_extract_rdf,
    jsonld_dir_to_markdown_str,
    ollama_embed,
    ollama_generate,
    ollama_generate_stream,
    strip_deepseek_think,
    top_k_cosine,
)

# Look up the plugin entry point to construct the root_path correctly
chatbot_api_entry_point = config.get_plugin_entry_point(
    "nomad_compass.apis:chatbot_api"
)

app = FastAPI(
    root_path=f"{config.services.api_base_path}/{chatbot_api_entry_point.prefix}",
    title="NOMAD Compass — Minimal Local RAG API",
)

app.add_middleware(  # type: ignore
    CORSMiddleware,  # type: ignore
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory RAG state
app.state.vectors = None
app.state.chunks = None
app.state.metas = None
app.state.config = {}  # extra debug info (incl. startup build status)


class BuildRequest(BaseModel):
    # Default: use local data directory (configured in rag_core.DEFAULT_RDF_ZIP)
    rdf_zip: str | None = DEFAULT_RDF_ZIP
    chunk_size: int = 1200
    chunk_overlap: int = 200
    embed_model: str = DEFAULT_EMBED_MODEL


class AskRequest(BaseModel):
    question: str
    k: int = 8
    chat_model: str = DEFAULT_CHAT_MODEL
    temperature: float = 0.2
    embed_model: str = DEFAULT_EMBED_MODEL


async def _build_index_internal(req: BuildRequest) -> dict[str, int]:
    """
    Shared helper that builds the in-memory RAG index and updates app.state.

    Supports:
    - Local directory containing JSON-LD/JSON files
    - Remote ZIP URL (fallback, using download_extract_rdf)
    """
    source = req.rdf_zip or DEFAULT_RDF_ZIP
    source_path = Path(source)

    if source_path.exists():
        # Local directory (or file) with RDF JSON-LD content
        rdf_dir = source_path
    else:
        # Remote ZIP (legacy behaviour)
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as td:
            tmp = Path(td)
            rdf_dir = await download_extract_rdf(source, tmp)

    md_text, count = jsonld_dir_to_markdown_str(rdf_dir)

    chunks = chunk_text(
        md_text, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap
    )
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No chunks produced from markdown; check your RDF input.",
        )

    embeddings = await ollama_embed(
        chunks, model=req.embed_model, base_url=OLLAMA_BASE_URL
    )
    metas = [{"source": "combined.md", "idx": i} for i in range(len(chunks))]

    app.state.vectors = embeddings
    app.state.chunks = chunks
    app.state.metas = metas
    app.state.config = {
        "ollama_base_url": OLLAMA_BASE_URL,
        "embed_model": req.embed_model,
        "chunk_size": req.chunk_size,
        "chunk_overlap": req.chunk_overlap,
        "doc_count": count,
        "chunks": len(chunks),
        "markdown": "in-memory",
    }
    return {"docs": count, "chunks": len(chunks)}


def initialize_index() -> None:
    """
    Build the default RAG index at process/plugin startup.

    - Safe to call multiple times (no-op if index already exists).
    - Safe to call when an event loop is already running: in that case we
      schedule the build as a background task instead of calling asyncio.run.
    """
    if getattr(app.state, "vectors", None) is not None:
        return

    req = BuildRequest()

    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We are already inside an event loop (e.g. async tests or
            # when NOMAD starts the app from an async context) – schedule
            # the build in the background.
            loop.create_task(_build_index_internal(req))
        else:
            # Normal synchronous startup: block until the index is ready.
            asyncio.run(_build_index_internal(req))

        app.state.config["startup_index_built"] = True
    except Exception as exc:  # pragma: no cover  (best-effort logging)
        # Do not crash NOMAD startup if the index build fails.
        app.state.config["startup_index_built"] = False
        app.state.config["startup_index_error"] = repr(exc)
        print(
            f"[nomad_compass] Warning: initial RAG index build failed: {exc!r}"
        )


@app.get("/health")
async def health():
    """
    Simple health check + introspection of startup build.
    """
    return {
        "status": "ok",
        "ollama": OLLAMA_BASE_URL,
        "index_ready": app.state.vectors is not None,
        "config": app.state.config,
    }


@app.post("/build")
async def build(req: BuildRequest | None = Body(default=None)):
    """
    Explicit rebuild of the index (e.g. when data changes).
    """
    if req is None:
        req = BuildRequest()

    result = await _build_index_internal(req)
    return {"ok": True, **result}


@app.post("/ask")
async def ask(req: AskRequest):
    if (
        app.state.vectors is None
        or app.state.chunks is None
        or app.state.metas is None
    ):
        raise HTTPException(
            status_code=400,
            detail="Index not found. Build first via POST /build.",
        )

    q_vec = await ollama_embed(
        [req.question], model=req.embed_model, base_url=OLLAMA_BASE_URL
    )
    if q_vec.shape[0] == 0:
        raise HTTPException(status_code=500, detail="Failed to embed question.")
    top_idx = top_k_cosine(q_vec[0], app.state.vectors, k=req.k)

    parts = []
    for i, idx in enumerate(top_idx, 1):
        src = app.state.metas[idx].get("source", "")
        parts.append(f"### DOC {i} — {src}\n{app.state.chunks[idx]}")
    context = "\n\n".join(parts)

    prompt = "\n\n".join(
        [
            SYSTEM_PROMPT,
            f"User question:\n{req.question}",
            "<docs>",
            context,
            "</docs>",
        ]
    )
    raw = await ollama_generate(
        prompt,
        model=req.chat_model,
        temperature=req.temperature,
        base_url=OLLAMA_BASE_URL,
    )
    answer = strip_deepseek_think(raw)

    sources, seen = [], set()
    for idx in top_idx:
        s = app.state.metas[idx].get("source", "")
        if s and s not in seen:
            seen.add(s)
            sources.append(s)

    return {"answer": answer, "sources": sources[:10]}


@app.post("/ask-stream")
async def ask_stream(req: AskRequest):
    if (
        app.state.vectors is None
        or app.state.chunks is None
        or app.state.metas is None
    ):
        raise HTTPException(
            status_code=400,
            detail="Index not found. Build first via POST /build.",
        )

    q_vec = await ollama_embed(
        [req.question], model=req.embed_model, base_url=OLLAMA_BASE_URL
    )
    if q_vec.shape[0] == 0:
        raise HTTPException(status_code=500, detail="Failed to embed question.")
    top_idx = top_k_cosine(q_vec[0], app.state.vectors, k=req.k)

    parts = []
    for i, idx in enumerate(top_idx, 1):
        src = app.state.metas[idx].get("source", "")
        parts.append(f"### DOC {i} — {src}\n{app.state.chunks[idx]}")
    context = "\n\n".join(parts)

    sources, seen = [], set()
    for idx in top_idx:
        s = app.state.metas[idx].get("source", "")
        if s and s not in seen:
            seen.add(s)
            sources.append(s)

    prompt = "\n\n".join(
        [
            SYSTEM_PROMPT,
            f"User question:\n{req.question}",
            "<docs>",
            context,
            "</docs>",
        ]
    )

    async def sse_emitter():
        meta = {"sources": sources[:10]}
        # Send sources once
        yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"
        # Then stream chunks
        async for delta in ollama_generate_stream(
            prompt,
            model=req.chat_model,
            temperature=req.temperature,
            base_url=OLLAMA_BASE_URL,
        ):
            payload = {"delta": delta}
            yield (
                "event: chunk\n"
                f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            )
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(sse_emitter(), media_type="text/event-stream")


@app.post("/wipe")
async def wipe():
    app.state.vectors = None
    app.state.chunks = None
    app.state.metas = None
    app.state.config = {}
    return {"ok": True, "removed": ["in-memory index"]}


@app.get("/starters")
async def starters():
    return {
        "starters": [
            "I'm new to NOMAD. Where should I begin?",
            "Can you find a tutorial on how to upload data?",
            "Explain the concept of the NOMAD Archive.",
            "Are there any videos on data visualization?",
        ]
    }


# Best-effort: build index as soon as this module is imported
# (works both in full NOMAD and when running stand-alone with
#  `uvicorn nomad_compass.apis.chatbot_api:app`).
initialize_index()
