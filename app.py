"""
Локальный режим (не реализуем в коде, только где заменить):
- build_llm(): заменить ChatOpenAI -> любой локальный chat-model (например, Ollama/llama.cpp backend)
- init_embeddings(): заменить OpenAIEmbeddings -> локальные embeddings
Остальной код не меняется.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from config import AppConfig
from ingest import ingest_files
from logger_config import init_logging
from rag_graph import RAGState, GRAPH

init_logging()

_log = logging.getLogger(__name__)

_log.info("boot: logger ready module=%s", __name__)

config = AppConfig()

# =========================
# FastAPI + Swagger
# =========================
app = FastAPI(title = "Minimal RAG Agent", version = "0.1.0")


class ChatReq(BaseModel):
    question: str


@app.post("/api/ingest")
def api_ingest(
        files: list[UploadFile] = File(...),
):
    try:
        _log.info("ingest: start files=%d", len(files))
        result = ingest_files(files)
        _log.info(
            "ingest: done collection=%s files=%d chunks=%d", result["collection"], len(result["files"]),
            result["chunks_added"],
        )
        return result
    except Exception as e:
        _log.error("ingest: error %s", e)
        raise


@app.post("/api/chat")
def api_chat( req: ChatReq ):
    try:

        if not req.question or len(req.question) > config.MAX_QUESTION_CHARS:
            _log.warning("chat: bad question length len=%d", len(req.question))
            raise HTTPException(400, "Bad question length")

        state: RAGState = {
            "question": req.question,
            "top_k": config.DEFAULT_TOP_K,
            "evidence": [],
            "answer": "",
        }
        _log.info("chat: start question = %s top_k=%d", req.question, config.DEFAULT_TOP_K)

        out = GRAPH.invoke(state)
        _log.info("chat: done answer_len=%d evidence=%d", len(out["answer"]), len(out.get("evidence", [])))

        return {
            "answer": out["answer"],
            "evidence": out.get("evidence", []),
        }
    except Exception as e:
        _log.error("chat: error %s", e)
        raise


@app.post("/api/ask")
def api_ask(
        question: str = Form(...),
        files: list[UploadFile] = File(...),
):
    try:
        if not question or len(question) > config.MAX_QUESTION_CHARS:
            _log.warning("chat: bad question length len=%d", len(question))
            raise HTTPException(400, "Bad question length")

        ingest_files(files)

        state: RAGState = {
            "question": question,
            "top_k": config.DEFAULT_TOP_K,
            "evidence": [],
            "answer": "",
        }
        _log.info("chat: start question = %s top_k=%d", question, config.DEFAULT_TOP_K)
        out = GRAPH.invoke(state)
        _log.info("chat: done answer_len=%d evidence=%d", len(out["answer"]), len(out.get("evidence", [])))
        return {
            "answer": out["answer"],
            "evidence": out.get("evidence", []),
        }
    except Exception as e:
        _log.error("chat: error %s", e)
        raise
