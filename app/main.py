import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Query
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from sqlalchemy import select, text, func

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.models import Base, Document
from app.schemas import DocumentResponse, UploadResponse, SearchResponse, SearchResult
from app.services import process_document, generate_embedding


# --- Settings ---
class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://user:password@db:5432/mydatabase"
    redis_url: str = "redis://redis:6379/0"
    chromadb_url: str = "http://host.docker.internal:8100"
    chromadb_collection_id: str = "69b78eb9-e090-42c2-b76d-9bf474088204"

    class Config:
        env_file = ".env"


settings = Settings()


# --- Stats Response Model ---
class StatsResponse(BaseModel):
    documents: int
    memories: int
    status: str

# --- Database Setup ---
DATABASE_URL = settings.database_url
engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = async_sessionmaker(
    autocommit=False, autoflush=False, bind=engine, class_=AsyncSession
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


# --- Redis Setup ---
redis_client: redis.Redis = None

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to DB and Redis
    print("Application startup: Connecting to DB and Redis...")

    # Enable pgvector extension
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    global redis_client
    redis_client = redis.from_url(settings.redis_url)
    print("Connected to DB and Redis. pgvector extension enabled.")
    yield
    # Shutdown: Close DB and Redis connections
    print("Application shutdown: Closing DB and Redis connections...")
    if redis_client:
        await redis_client.close()
    print("DB and Redis connections closed.")


# --- FastAPI App ---
app = FastAPI(lifespan=lifespan, title="Document Processor API")


# --- Endpoints ---
@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document for processing.
    Saves record to Postgres with status 'processing' and triggers background task.
    """
    # Read file content to get size
    content = await file.read()
    file_size = len(content)

    # Create document record in database
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
        print(f"Created document record: id={doc_id}, filename={file.filename}")

        # Trigger background processing task with file content
        asyncio.create_task(
            process_document(doc_id, content, file.filename, AsyncSessionLocal)
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
    """
    Semantic search across documents using vector similarity.
    Embeds the query and finds most similar documents using cosine similarity.
    """
    # Generate embedding for the query
    query_embedding = await generate_embedding(q)

    if not query_embedding:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service unavailable",
        )

    # Search using cosine similarity (1 - cosine_distance)
    async with AsyncSessionLocal() as session:
        # Use pgvector's cosine distance operator (<=>)
        # Lower distance = more similar, so we order ascending
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

        return SearchResponse(
            query=q,
            results=results,
            count=len(results),
        )


@app.get("/status/{doc_id}", response_model=DocumentResponse)
async def get_document_status(doc_id: int):
    """
    Get the status of a document by ID.
    """
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
    """
    Retrieve information about a document.
    """
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
    """
    Get live system statistics.

    Returns document count from Postgres and memory count from ChromaDB.
    """
    documents = 0
    memories = 0
    status = "ONLINE"

    # Count documents in Postgres
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(func.count(Document.id)))
            documents = result.scalar() or 0
    except Exception as e:
        print(f"[STATS] Postgres error: {e}")
        status = "DEGRADED"

    # Count memories in ChromaDB
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
        print(f"[STATS] ChromaDB error: {e}")
        # ChromaDB being down doesn't mean full degradation
        memories = -1  # Indicate unavailable

    return StatsResponse(
        documents=documents,
        memories=memories,
        status=status
    )


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Document Processor API"}
