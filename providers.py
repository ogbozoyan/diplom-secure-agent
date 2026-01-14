"""
Локальный режим (не реализуем в коде, только где заменить):
- build_llm(): заменить ChatOpenAI -> любой локальный chat-model (например, Ollama/llama.cpp backend)
- init_embeddings(): заменить OpenAIEmbeddings -> локальные embeddings
Остальной код не меняется.
"""
from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import AppConfig

_log = logging.getLogger(__name__)

config = AppConfig()


# =========================
# Providers (OpenAI now)
# =========================
def init_llm( ) -> ChatOpenAI:
    if not config.OPENAI_API_KEY:
        _log.error("openai: OPENAI_API_KEY is empty -> cannot build llm")
        raise RuntimeError("OPENAI_API_KEY is empty")
    ai = ChatOpenAI(model = config.OPENAI_MODEL, api_key = config.OPENAI_API_KEY, temperature = 0)
    _log.info("openai: build llm model=%s temperature=%s", config.OPENAI_MODEL, ai.temperature)
    return ai


def init_embeddings( ) -> OpenAIEmbeddings:
    if not config.OPENAI_API_KEY:
        _log.error("openai: OPENAI_API_KEY is empty -> cannot build embeddings")
        raise RuntimeError("OPENAI_API_KEY is empty")
    ai_embeddings = OpenAIEmbeddings(model = config.OPENAI_EMB_MODEL, api_key = config.OPENAI_API_KEY)
    _log.info("openai: build embeddings model=%s", config.OPENAI_EMB_MODEL)
    return ai_embeddings
