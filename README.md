```bash
docker run --name pgvector-container \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rag \
  -p 6024:5432 \
  -d pgvector/pgvector:pg16
```

```bash
export PG_DSN="postgresql://postgres:postgres@localhost:5432/rag"
export OPENAI_API_KEY="..."

1. pip install -r requirements.txt
2. uvicorn app:app --reload --port 8000 
```
