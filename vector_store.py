from __future__ import annotations

import logging

import psycopg
from langchain_postgres import PGEngine, PGVectorStore
from sqlalchemy import URL
from sqlalchemy.engine import make_url

from config import AppConfig
from providers import init_embeddings

_log = logging.getLogger(__name__)

config = AppConfig()


# =========================
# VectorStore factory
# =========================

def pg_table_exists( pg_dsn: str, schema: str, table: str ) -> bool:
    full = f"{schema}.{table}"
    with psycopg.connect(pg_dsn) as conn:
        _log.info(
            "pgvector: ensure extension vector dsn=%s", make_url(config.PG_DSN).render_as_string(hide_password = True),
        )
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (full,))
            return cur.fetchone()[0] is not None


def psycopg_dsn_from_sqlalchemy( sa_url: str ) -> str:
    """
    sa_url example: postgresql+psycopg://user:pass@host:port/db
    psycopg needs:  postgresql://user:pass@host:port/db
    """
    u: URL = make_url(sa_url)
    u2: URL = u.set(drivername = "postgresql")
    return u2.render_as_string(hide_password = False)


def ensure_pgvector_store(
        vector_size: int,
) -> PGVectorStore:
    table_name = "embeddings"

    pg_dsn: str = psycopg_dsn_from_sqlalchemy(config.PG_URL)
    exists = pg_table_exists(pg_dsn, "public", table_name)

    pg_engine: PGEngine = PGEngine.from_connection_string(config.PG_URL)
    if not exists:
        pg_engine.init_vectorstore_table(
            table_name = table_name,
            schema_name = "public",
            vector_size = vector_size,
            overwrite_existing = True,
            id_column = "langchain_id",
            store_metadata = True,
        )

    return PGVectorStore.create_sync(
        engine = pg_engine,
        embedding_service = init_embeddings(),
        table_name = table_name,
        schema_name = "public",
    )


pg_engine = ensure_pgvector_store(config.VECTOR_SIZE)
