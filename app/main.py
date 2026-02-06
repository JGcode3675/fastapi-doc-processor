import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import select, text, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

import redis.asyncio as redis
from arq import create_pool
from arq.connections import RedisSettings

from app.config import get_settings
from app.logging_config import setup_logging, get_logger
from app.models import Base, Document, ChatSession, ChatMessage as ChatMessageModel
from app.schemas import (
    DocumentResponse, UploadResponse, SearchResponse, SearchResult,
    ChatRequest, ChatResponse, ChatSessionResponse, StatsResponse,
    BriefingRequest, BriefingResponse,
)
from app.services import generate_embedding
from app.agents import MoltbotAgent

setup_logging()
log = get_logger(__name__)

settings = get_settings()

# --- Database Setup ---
engine = create_async_engine(settings.database_url)
AsyncSessionLocal = async_sessionmaker(
    autocommit=False, autoflush=False, bind=engine, class_=AsyncSession
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


# --- Redis Setup ---
redis_client: redis.Redis = None

# --- WebSocket Connection Manager ---
ws_clients: set[WebSocket] = set()


async def broadcast_ws(event: dict) -> None:
    """Broadcast a JSON event to all connected WebSocket clients."""
    data = json.dumps(event)
    disconnected = set()
    for ws in ws_clients:
        try:
            await ws.send_text(data)
        except Exception:
            disconnected.add(ws)
    ws_clients.difference_update(disconnected)


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("app_startup", msg="Connecting to DB and Redis")

    global redis_client
    redis_client = redis.from_url(settings.redis_url)

    # Create ARQ pool for task enqueueing
    arq_settings = RedisSettings.from_dsn(settings.redis_url)
    app.state.arq_pool = await create_pool(arq_settings)

    log.info("app_ready", msg="DB, Redis, and ARQ pool connected")
    yield

    log.info("app_shutdown", msg="Closing connections")
    if redis_client:
        await redis_client.close()
    await app.state.arq_pool.close()
    log.info("app_shutdown_complete")


# --- FastAPI App ---
app = FastAPI(lifespan=lifespan, title="Document Processor API")


# --- Endpoints ---
@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document for processing.
    Saves record to Postgres with status 'processing' and enqueues ARQ job.
    """
    content = await file.read()
    file_size = len(content)

    async with AsyncSessionLocal() as session:
        document = Document(
            filename=file.filename,
            content_type=file.content_type,
            file_size=file_size,
            status="processing",
        )
        session.add(document)
        await session.commit()
        await session.refresh(document)

        doc_id = document.id
        log.info("doc_uploaded", doc_id=doc_id, filename=file.filename, size=file_size)

        # Enqueue ARQ job instead of fire-and-forget asyncio.create_task
        await app.state.arq_pool.enqueue_job(
            "process_document_task", doc_id, content, file.filename
        )

        # Publish initial status via Redis pubsub for WebSocket clients
        if redis_client:
            await redis_client.publish(
                "ws:events",
                json.dumps({
                    "type": "doc_status",
                    "doc_id": doc_id,
                    "status": "processing",
                    "progress": 0,
                }),
            )

        return UploadResponse(
            message="Document uploaded successfully",
            document=DocumentResponse.model_validate(document),
        )


@app.get("/search", response_model=SearchResponse)
async def search_documents(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=5, ge=1, le=20, description="Number of results"),
):
    """Semantic search across documents using vector similarity."""
    query_embedding = await generate_embedding(q)

    if not query_embedding:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service unavailable",
        )

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT
                    id,
                    filename,
                    summary,
                    1 - (embedding <=> :query_embedding) as similarity
                FROM documents
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> :query_embedding
                LIMIT :limit
            """),
            {"query_embedding": str(query_embedding), "limit": limit},
        )
        rows = result.fetchall()

        results = [
            SearchResult(
                id=row.id,
                filename=row.filename,
                summary=row.summary,
                similarity=float(row.similarity),
            )
            for row in rows
        ]

        return SearchResponse(query=q, results=results, count=len(results))


@app.get("/status/{doc_id}", response_model=DocumentResponse)
async def get_document_status(doc_id: int):
    """Get the status of a document by ID."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Document).where(Document.id == doc_id)
        )
        document = result.scalar_one_or_none()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {doc_id} not found",
            )
        return DocumentResponse.model_validate(document)


@app.get("/docs/{doc_id}", response_model=DocumentResponse)
async def get_document_info(doc_id: int):
    """Retrieve information about a document."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Document).where(Document.id == doc_id)
        )
        document = result.scalar_one_or_none()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {doc_id} not found",
            )
        return DocumentResponse.model_validate(document)


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get live system statistics."""
    documents = 0
    memories = 0
    sys_status = "ONLINE"

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(func.count(Document.id)))
            documents = result.scalar() or 0
    except Exception as e:
        log.error("stats_postgres_error", error=str(e))
        sys_status = "DEGRADED"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            chroma_url = (
                f"{settings.chromadb_url}/api/v2/tenants/default_tenant/"
                f"databases/default_database/collections/{settings.chromadb_collection_id}/count"
            )
            response = await client.get(chroma_url)
            if response.status_code == 200:
                memories = int(response.text)
    except Exception as e:
        log.warning("stats_chromadb_error", error=str(e))
        memories = -1

    return StatsResponse(documents=documents, memories=memories, status=sys_status)


# --- Chat Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with Moltbot - the AI Systems Architect.

    Supports session-based history, RAG, and tool use via MoltbotAgent.
    """
    agent = MoltbotAgent(
        session_factory=AsyncSessionLocal,
        settings=settings,
    )

    try:
        response_text, tokens = await agent.chat(
            user_message=request.query,
            session_id=request.session_id,
            system_prompt=request.system_prompt,
        )
        return ChatResponse(
            response=response_text,
            model=request.model,
            tokens_used=tokens,
            session_id=request.session_id,
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama service unavailable. Is it running?",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="LLM request timed out",
        )


@app.get("/chat/sessions", response_model=list[ChatSessionResponse])
async def list_chat_sessions():
    """List all chat sessions."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ChatSession).order_by(ChatSession.created_at.desc())
        )
        sessions = result.scalars().all()
        return [ChatSessionResponse.model_validate(s) for s in sessions]


@app.get("/chat/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(session_id: int):
    """Get a chat session with message history."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        chat_session = result.scalar_one_or_none()
        if not chat_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )
        return ChatSessionResponse.model_validate(chat_session)


@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: int):
    """Delete a chat session and its messages."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        chat_session = result.scalar_one_or_none()
        if not chat_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )
        await session.delete(chat_session)
        await session.commit()
        return {"message": f"Session {session_id} deleted"}


# --- Briefing ---

@app.post("/briefing/generate", response_model=BriefingResponse)
async def generate_briefing(request: BriefingRequest):
    """Generate a daily intelligence briefing from recent documents."""
    from datetime import timedelta

    documents_data = []

    async with AsyncSessionLocal() as session:
        cutoff = datetime.now() - timedelta(hours=24)
        result = await session.execute(
            text("""
                SELECT DISTINCT ON (filename) filename, summary, content, created_at
                FROM documents
                WHERE status = 'completed'
                  AND created_at > :cutoff
                ORDER BY filename, created_at DESC
                LIMIT 10
            """),
            {"cutoff": cutoff},
        )
        rows = result.fetchall()

        if not rows:
            result = await session.execute(
                text("""
                    SELECT DISTINCT ON (filename) filename, summary, content, created_at
                    FROM documents
                    WHERE status = 'completed'
                    ORDER BY filename, created_at DESC
                    LIMIT 5
                """)
            )
            rows = result.fetchall()

        for row in rows:
            content_excerpt = ""
            if row.summary:
                content_excerpt = row.summary[:1500]
            elif row.content:
                content_excerpt = row.content[:3000]

            documents_data.append({
                "filename": row.filename,
                "excerpt": content_excerpt,
            })

    if not documents_data:
        return BriefingResponse(
            markdown_report="## No Intelligence Available\n\nNo documents have been indexed yet.",
            timestamp=datetime.now().isoformat(),
            document_count=0,
        )

    doc_context = "\n\n".join([
        f"**{d['filename']}**:\n{d['excerpt']}" for d in documents_data
    ])

    system_prompt = """You are Moltbot, a tactical intelligence analyst for the AI Nervous System.
Summarize these document excerpts into a daily briefing.

Structure your response with these exact sections:
## Critical Risks
## Opportunities
## Tech Watch

Rules:
- Be concise. Use bullet points.
- If no relevant info for a section, state 'N/A'.
- Focus on actionable intelligence."""

    user_prompt = f"Generate a daily briefing from these {len(documents_data)} documents:\n\n{doc_context}"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            ollama_response = await client.post(
                settings.ollama_chat_url,
                json={
                    "model": settings.ollama_llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
            )

            if ollama_response.status_code == 200:
                data = ollama_response.json()
                markdown_report = data.get("message", {}).get("content", "")
                return BriefingResponse(
                    markdown_report=markdown_report,
                    timestamp=datetime.now().isoformat(),
                    document_count=len(documents_data),
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Ollama error: {ollama_response.status_code}",
                )

    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama service unavailable. Is it running?",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Briefing generation timed out",
        )


# --- WebSocket ---

@app.websocket("/ws/status")
async def websocket_status(ws: WebSocket):
    """
    WebSocket endpoint for real-time system events.

    Subscribes to Redis pubsub and pushes all events to connected clients.
    Protocol:
        {"type": "doc_status", "doc_id": 123, "status": "embedding", "progress": 50}
        {"type": "system", "event": "worker_ready", "workers": 4}
    """
    await ws.accept()
    ws_clients.add(ws)
    log.info("ws_client_connected", total=len(ws_clients))

    pubsub = redis_client.pubsub()
    await pubsub.subscribe("ws:events")

    try:
        while True:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=1.0
            )
            if message and message["type"] == "message":
                await ws.send_text(message["data"].decode() if isinstance(message["data"], bytes) else message["data"])

            # Also check if WebSocket client sent anything (ping/pong)
            try:
                data = await ws.receive_text()
                if data == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except WebSocketDisconnect:
                break
            except Exception:
                pass

    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)
        await pubsub.unsubscribe("ws:events")
        await pubsub.close()
        log.info("ws_client_disconnected", total=len(ws_clients))


# --- Health ---

@app.get("/health")
async def health_check():
    """Simple health check endpoint for monitoring."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Document Processor API"}
