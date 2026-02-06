# User Guide

## Quick Start

```bash
# Start everything (detached)
docker compose up -d --build

# Stop everything
docker compose down

# View logs
docker compose logs -f          # all services
docker compose logs -f app      # app only
docker compose logs -f worker   # ARQ worker only
```

## API Endpoints

Base URL: `http://localhost:8000`

### Health & Info

```bash
# Health check
curl localhost:8000/health

# System stats (document count, memory count, status)
curl localhost:8000/stats
```

### Documents

```bash
# Upload a document (triggers background processing)
curl -X POST localhost:8000/upload -F "file=@path/to/file.txt"

# Check processing status
curl localhost:8000/status/1

# Get document details
curl localhost:8000/docs/1

# Semantic search across documents
curl "localhost:8000/search?q=your+query&limit=5"
```

### Chat (Moltbot Agent)

```bash
# Send a message (new session)
curl -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What documents do I have?"}'

# Continue a session
curl -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me more", "session_id": 1}'

# Custom system prompt
curl -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello", "system_prompt": "You are a pirate."}'

# List all chat sessions
curl localhost:8000/chat/sessions

# Get session with message history
curl localhost:8000/chat/sessions/1

# Delete a session
curl -X DELETE localhost:8000/chat/sessions/1
```

### Briefing

```bash
# Generate intelligence briefing from recent documents
curl -X POST localhost:8000/briefing/generate \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Interactive Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Database Migrations (Alembic)

```bash
# Apply all pending migrations
docker compose run --rm migrate alembic upgrade head

# Rollback one migration
docker compose run --rm migrate alembic downgrade -1

# Show current migration version
docker compose run --rm migrate alembic current

# Show migration history
docker compose run --rm migrate alembic history

# Auto-generate a new migration after changing models
docker compose run --rm migrate alembic revision --autogenerate -m "describe change"

# Stamp existing DB (skip running migrations, mark as current)
docker compose run --rm migrate alembic stamp head
```

## WebSocket

```bash
# Connect to real-time status feed (document processing events)
websocat ws://localhost:8000/ws/status
```

Events look like:
```json
{"type": "doc_status", "doc_id": 1, "status": "embedding", "progress": 60}
```

## Testing

```bash
# Run all tests
.venv/bin/python -m pytest tests/ -v

# Run specific test file
.venv/bin/python -m pytest tests/test_endpoints.py -v
.venv/bin/python -m pytest tests/test_services.py -v
```

## Environment

Copy `.env.example` to `.env` and adjust values. Key variables:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://user:password@db:5432/mydatabase` | Postgres connection |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama LLM server |
| `OLLAMA_LLM_MODEL` | `llama3.2:3b` | Chat/summarization model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model (768d) |
| `ARQ_MAX_JOBS` | `4` | Concurrent worker jobs |
| `ARQ_JOB_TIMEOUT` | `300` | Job timeout in seconds |
| `LOG_FORMAT` | `console` | `console` or `json` |
| `LOG_LEVEL` | `INFO` | Python log level |

## Architecture

```
docker compose up
  db (pgvector/pg16)       - PostgreSQL with vector extension
  redis (redis:6-alpine)   - Job queue + pubsub
  migrate                  - Runs alembic upgrade head, then exits
  app (4 gunicorn workers) - FastAPI API server
  worker (arq)             - Background document processing
```
