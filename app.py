from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from config import AppConfig
from ingest import ingest_files
from logger_config import init_logging
from rag_graph import run_graph

init_logging()
_log = logging.getLogger(__name__)

config = AppConfig()

app = FastAPI(
    title = "Diplom Security RAG Agent",
    description = (
        "Simple RAG (pgvector) + LangGraph demo.\n\n"
        "This API intentionally exposes two execution modes:\n"
        "- baseline (vulnerable)\n"
        "- secure (guards on input/context/output)\n\n"
        "Use the `secure` flag in requests to switch."
    ),
    version = "0.1.0",
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length = 1, description = "User question")
    secure: bool = Field(default = False, description = "Use secure graph (guards enabled)")


# -----------------------------
# API: (a) embed files
# -----------------------------
@app.post("/api/ingest")
def api_ingest( files: list[UploadFile] = File(...) ) -> dict[str, Any]:
    if not files:
        _log.error("ingest: no files provided")
        raise HTTPException(status_code = 400, detail = "No files provided")

    _log.info("ingest: start files=%d", len(files))
    try:
        res = ingest_files(files)
        _log.info("ingest: done %s", res)
        return { "status": "ok", **res }
    except Exception as e:
        _log.exception("ingest: error")
        raise HTTPException(status_code = 500, detail = f"ingest_failed:{e}") from e


# -----------------------------
# API: (b) chat mode (RAG over already ingested KB)
# -----------------------------
@app.post("/api/chat")
def api_chat( req: ChatRequest ) -> dict[str, Any]:
    question = req.question.strip()
    secure = bool(req.secure)

    _log.info("chat: start secure=%s question=%d", secure, question)
    try:
        resp = run_graph(question = question, secure = secure)
        _log.info(
            "chat: done secure=%s answer_len=%d evidence=%d blocked=%s",
            secure,
            len(resp.get("answer", "")),
            len(resp.get("evidence", [])),
            resp.get("blocked", False),
        )
        return resp
    except Exception as e:
        _log.exception("chat: error")
        raise HTTPException(status_code = 500, detail = f"chat_failed:{e}") from e


# -----------------------------
# API: (a + b) ingest then chat
# -----------------------------
@app.post("/api/ask")
def api_ask(
        question: str = Form(...),
        secure: bool = Form(False),
        files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    q = question.strip()
    if not q:
        _log.error("ask: empty question")
        raise HTTPException(status_code = 400, detail = "Empty question")
    if not files:
        _log.error("ask: no files provided")
        raise HTTPException(status_code = 400, detail = "No files provided")

    _log.info("ask: start secure=%s top_k=%d files=%d q_len=%d", secure, config, len(files), len(q))

    # (a) ingest
    try:
        ingest_res = ingest_files(files)
    except Exception as e:
        _log.exception("ask: ingest error")
        raise HTTPException(status_code = 500, detail = f"ask_ingest_failed:{e}") from e

    # (b) chat
    try:
        chat_res = run_graph(question = q, secure = bool(secure))
    except Exception as e:
        _log.exception("ask: chat error")
        raise HTTPException(status_code = 500, detail = f"ask_chat_failed:{e}") from e

    _log.info("ask: done secure=%s", secure)
    return {
        "status": "ok",
        "ingest": ingest_res,
        "chat": chat_res,
    }
