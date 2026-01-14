from __future__ import annotations

import logging

from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import AppConfig
from file_parser import check_file_limits, extract_text
from vector_store import pg_engine

_log = logging.getLogger(__name__)

config = AppConfig()


# =========================
# Ingest (embed files)
# =========================
def ingest_files( files: list[UploadFile] ) -> dict:
    _log.info(
        "ingest: start files=%d chunk_size=%d overlap=%d", len(files), config.CHUNK_SIZE,
        config.CHUNK_OVERLAP,
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size = config.CHUNK_SIZE, chunk_overlap = config.CHUNK_OVERLAP)

    total_chunks = 0
    sources: list[str] = []

    for f in files:
        check_file_limits(f)
        text = extract_text(f).strip()
        if not text:
            _log.warning("ingest: empty text -> skip filename=%s", f.filename)
            continue

        _log.info("ingest: file start filename=%s", f.filename)
        docs = splitter.create_documents(
            texts = [text],
            metadatas = [{ "source_file": f.filename }],
        )
        _log.info("ingest: extracted filename=%s chars=%d", f.filename, len(text))
        pg_engine.add_documents(docs)

        total_chunks += len(docs)
        sources.append(f.filename or "unknown")
        _log.info("ingest: chunked filename=%s chunks=%d", f.filename, len(docs))

    return { "files": sources, "chunks_added": total_chunks }
