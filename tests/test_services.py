"""
Service Tests
=============

Tests for services.py: embedding generation and summarization with mocked Ollama.
"""

from unittest.mock import patch, AsyncMock, MagicMock

import httpx
import pytest


@pytest.mark.asyncio
async def test_generate_embedding_success():
    """Test successful embedding generation."""
    mock_embedding = [0.1] * 768
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": [mock_embedding]}
    mock_response.raise_for_status = MagicMock()

    with patch("app.services.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        MockClient.return_value = mock_client

        from app.services import generate_embedding
        result = await generate_embedding("test text")

        assert result is not None
        assert len(result) == 768


@pytest.mark.asyncio
async def test_generate_embedding_connection_error():
    """Test embedding generation when Ollama is unavailable."""
    with patch("app.services.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        MockClient.return_value = mock_client

        from app.services import generate_embedding
        result = await generate_embedding("test text")

        assert result is None


@pytest.mark.asyncio
async def test_summarize_with_ollama_success():
    """Test successful summarization."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "- Point 1\n- Point 2\n- Point 3"}
    mock_response.raise_for_status = MagicMock()

    with patch("app.services.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        MockClient.return_value = mock_client

        from app.services import summarize_with_ollama
        result = await summarize_with_ollama("Some long document content here.")

        assert result is not None
        assert "Point 1" in result


@pytest.mark.asyncio
async def test_summarize_with_ollama_connection_error():
    """Test summarization when Ollama is unavailable."""
    with patch("app.services.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        MockClient.return_value = mock_client

        from app.services import summarize_with_ollama
        result = await summarize_with_ollama("Some content")

        assert result is None


@pytest.mark.asyncio
async def test_generate_embedding_empty_response():
    """Test embedding generation with empty response from Ollama."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": []}
    mock_response.raise_for_status = MagicMock()

    with patch("app.services.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        MockClient.return_value = mock_client

        from app.services import generate_embedding
        result = await generate_embedding("test text")

        assert result is None
