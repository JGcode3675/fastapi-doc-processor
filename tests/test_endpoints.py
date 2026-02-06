"""
Endpoint Tests
==============

Tests for all API endpoints: /health, /upload, /search, /chat, /stats.
Uses mocked MoltbotAgent for chat tests, mocked session for DB-dependent tests.
"""

from contextlib import asynccontextmanager
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime

import pytest
from httpx import AsyncClient


# --- Helper to mock AsyncSessionLocal ---

def make_mock_session(execute_result=None, scalar_result=None):
    """Create a mock async session that works as an async context manager."""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.delete = AsyncMock()

    if execute_result is not None:
        mock_session.execute = AsyncMock(return_value=execute_result)
    if scalar_result is not None:
        # Use MagicMock for the result since scalar_one_or_none is sync
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=scalar_result)
        mock_session.execute = AsyncMock(return_value=mock_result)

    @asynccontextmanager
    async def session_ctx():
        yield mock_session

    return session_ctx


# --- Health & Root ---

@pytest.mark.asyncio
async def test_health(async_client: AsyncClient):
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_root(async_client: AsyncClient):
    response = await async_client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


# --- Search ---

@pytest.mark.asyncio
async def test_search_requires_query(async_client: AsyncClient):
    """Test that search requires a query parameter."""
    response = await async_client.get("/search")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_with_unavailable_embedding(async_client: AsyncClient):
    """Test search when embedding service is unavailable."""
    with patch("app.main.generate_embedding", return_value=None):
        response = await async_client.get("/search", params={"q": "test query"})
        assert response.status_code == 503


# --- Chat ---

@pytest.mark.asyncio
async def test_chat_endpoint(async_client: AsyncClient, mock_ollama_response):
    """Test chat endpoint with mocked MoltbotAgent."""
    with patch("app.main.MoltbotAgent") as MockAgent:
        mock_agent = AsyncMock()
        mock_agent.chat = AsyncMock(return_value=("Test response", 42))
        MockAgent.return_value = mock_agent

        response = await async_client.post("/chat", json={
            "query": "Hello Moltbot",
            "model": "moltbot",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Test response"
        assert data["tokens_used"] == 42


@pytest.mark.asyncio
async def test_chat_with_session_id(async_client: AsyncClient):
    """Test chat endpoint with session_id for history persistence."""
    with patch("app.main.MoltbotAgent") as MockAgent:
        mock_agent = AsyncMock()
        mock_agent.chat = AsyncMock(return_value=("Persisted response", 10))
        MockAgent.return_value = mock_agent

        response = await async_client.post("/chat", json={
            "query": "Remember this",
            "session_id": 1,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == 1


@pytest.mark.asyncio
async def test_chat_ollama_unavailable(async_client: AsyncClient):
    """Test chat when Ollama is down."""
    import httpx as httpx_lib
    with patch("app.main.MoltbotAgent") as MockAgent:
        mock_agent = AsyncMock()
        mock_agent.chat = AsyncMock(side_effect=httpx_lib.ConnectError("Connection refused"))
        MockAgent.return_value = mock_agent

        response = await async_client.post("/chat", json={
            "query": "Hello",
        })
        assert response.status_code == 503


# --- Upload ---

@pytest.mark.asyncio
async def test_upload_document(async_client: AsyncClient):
    """Test document upload creates a record and enqueues an ARQ job."""
    import io

    mock_doc = MagicMock()
    mock_doc.id = 1
    mock_doc.filename = "test.txt"
    mock_doc.content_type = "text/plain"
    mock_doc.file_size = 12
    mock_doc.status = "processing"
    mock_doc.summary = None
    mock_doc.created_at = datetime.now()
    mock_doc.updated_at = datetime.now()

    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock(side_effect=lambda doc: None)

    @asynccontextmanager
    async def mock_session_ctx():
        yield mock_session

    with patch("app.main.AsyncSessionLocal", side_effect=lambda: mock_session_ctx()), \
         patch("app.main.Document", return_value=mock_doc), \
         patch("app.main.redis_client") as mock_redis:
        mock_redis.publish = AsyncMock()

        files = {"file": ("test.txt", io.BytesIO(b"test content"), "text/plain")}
        response = await async_client.post("/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Document uploaded successfully"
        assert data["document"]["filename"] == "test.txt"


# --- Stats ---

@pytest.mark.asyncio
async def test_stats_endpoint(async_client: AsyncClient):
    """Test stats returns expected structure."""
    mock_count_result = MagicMock()
    mock_count_result.scalar.return_value = 5

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_count_result)

    @asynccontextmanager
    async def mock_session_ctx():
        yield mock_session

    with patch("app.main.AsyncSessionLocal", side_effect=lambda: mock_session_ctx()), \
         patch("app.main.httpx.AsyncClient") as mock_httpx:
        # Mock ChromaDB response
        mock_chroma_response = MagicMock()
        mock_chroma_response.status_code = 200
        mock_chroma_response.text = "42"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_chroma_response)
        mock_httpx.return_value = mock_client

        response = await async_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == 5
        assert data["status"] == "ONLINE"


# --- Document Status ---

@pytest.mark.asyncio
async def test_status_not_found(async_client: AsyncClient):
    """Test status endpoint for non-existent document."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)

    mock_session = MagicMock()
    mock_session.execute = AsyncMock(return_value=mock_result)

    @asynccontextmanager
    async def mock_ctx():
        yield mock_session

    with patch("app.main.AsyncSessionLocal", return_value=mock_ctx()):
        response = await async_client.get("/status/99999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_docs_not_found(async_client: AsyncClient):
    """Test docs endpoint for non-existent document."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)

    mock_session = MagicMock()
    mock_session.execute = AsyncMock(return_value=mock_result)

    @asynccontextmanager
    async def mock_ctx():
        yield mock_session

    with patch("app.main.AsyncSessionLocal", return_value=mock_ctx()):
        response = await async_client.get("/docs/99999")
        assert response.status_code == 404
