"""
Test Configuration
==================

Provides async test client, mock Ollama fixture, and test database setup.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def mock_arq_pool():
    """Mock ARQ pool so tests don't need real Redis for enqueueing."""
    pool = AsyncMock()
    pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="test-job-1"))
    pool.close = AsyncMock()
    return pool


@pytest.fixture(autouse=True)
def mock_redis():
    """Mock Redis client so tests don't need a real Redis instance."""
    redis_mock = AsyncMock()
    redis_mock.publish = AsyncMock()
    redis_mock.close = AsyncMock()
    return redis_mock


@pytest_asyncio.fixture
async def async_client(mock_arq_pool, mock_redis):
    """
    Create an async test client with mocked external dependencies.

    Uses ASGI transport so no real server is started.
    Patches Redis, ARQ, and database setup to avoid needing real services.
    """
    # Patch the database engine to use SQLite for tests
    with patch("app.main.engine") as mock_engine, \
         patch("app.main.redis_client", mock_redis), \
         patch("app.main.setup_logging"):

        from app.main import app

        # Inject mock ARQ pool
        app.state.arq_pool = mock_arq_pool

        # Mock the lifespan to skip real DB/Redis connections
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_lifespan(app):
            app.state.arq_pool = mock_arq_pool
            yield

        app.router.lifespan_context = mock_lifespan

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture
def mock_ollama_response():
    """Mock successful Ollama API response."""
    return {
        "model": "llama3.2:3b",
        "message": {"role": "assistant", "content": "Test response from Moltbot."},
        "eval_count": 42,
    }


@pytest.fixture
def mock_embedding():
    """Mock embedding vector (768 dimensions)."""
    return [0.1] * 768
