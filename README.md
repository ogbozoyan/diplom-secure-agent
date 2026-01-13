Структура проекта:

```text
app/
  config.py                 # настройки (env), выбор LLM режима, лимиты
  main.py                   # FastAPI, роуты, Swagger
  db.py                     # соединение с Postgres
  models.py                 # SQLAlchemy модели: Document, Chunk
  ingestion/
    parser.py               # извлечение текста из PDF/DOCX/TXT
    chunker.py              # разбиение на чанки
    service.py              # orchestration ingestion
  rag/
    providers.py            # LLM/Embeddings providers (OpenAI vs local)
    retrieval.py            # pgvector top-k поиск
    prompts.py              # шаблоны prompt
    graph.py                # LangGraph StateGraph
  security/
    limits.py               # квоты/лимиты/anti-DoS
    output_guard.py         # output handling, интеграция с guardrails
    sanitizer.py            # безопасный вывод (экранирование/markdown)
migrations/
  001_init.sql              # таблицы + индексы pgvector
pyproject.toml              # uv
```

Запуск

```bash
uv run uvicorn main:app --reload
```
