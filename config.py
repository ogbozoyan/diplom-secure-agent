from __future__ import annotations

import logging
import os

from pydantic_settings import BaseSettings
from sqlalchemy import make_url

_log = logging.getLogger(__name__)


class AppConfig(BaseSettings):
    TABLE_NAME: str = os.getenv("PG_TABLE", "vectorstore")

    # text-embedding-3-small по умолчанию 1536 измерений
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "1536"))
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    OPENAI_EMB_MODEL: str = os.getenv("OPENAI_EMB_MODEL", "text-embedding-3-small")

    PG_DSN: str = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:6024/rag")
    PG_URL: str = os.getenv("PG_URL") or PG_DSN.replace("postgresql://", "postgresql+psycopg://", 1)

    MAX_QUESTION_CHARS: int = int(os.getenv("MAX_QUESTION_CHARS", "2000"))
    MAX_FILE_MB: int = int(os.getenv("MAX_FILE_MB", "15"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    DEFAULT_TOP_K: int = int(os.getenv("TOP_K", "5"))
    _log.info(
        "boot: config table=%s vector_size=%d chunk_size=%d overlap=%d top_k=%d max_file_mb=%d max_question_chars=%d pg_url=%s openai_model=%s emb_model=%s openai_key_set=%s",
        TABLE_NAME,
        VECTOR_SIZE,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        DEFAULT_TOP_K,
        MAX_FILE_MB,
        MAX_QUESTION_CHARS,
        make_url(PG_URL).render_as_string(hide_password = True),
        OPENAI_MODEL,
        OPENAI_EMB_MODEL,
        bool(OPENAI_API_KEY),
    )
