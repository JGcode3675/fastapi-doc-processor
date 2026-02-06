import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.logging_config import get_logger
from app.models import Document

log = get_logger(__name__)


async def summarize_with_ollama(content: str) -> str | None:
    """
    Send content to Ollama for summarization.
    Returns the summary or None if Ollama is unavailable.
    """
    settings = get_settings()
    prompt = f"""Summarize this document in 3 bullet points:

{content[:1000]}"""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                settings.ollama_generate_url,
                json={
                    "model": settings.ollama_llm_model,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
    except httpx.ConnectError:
        log.error("ollama_connection_failed", service="summarize")
        return None
    except httpx.HTTPStatusError as e:
        log.error("ollama_http_error", service="summarize", status=e.response.status_code)
        return None
    except Exception as e:
        log.error("ollama_unexpected_error", service="summarize", error=str(e))
        return None


async def generate_embedding(text: str) -> list[float] | None:
    """
    Generate embeddings using Ollama's nomic-embed-text model.
    Returns a 768-dimensional vector or None if unavailable.
    """
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                settings.ollama_embed_url,
                json={
                    "model": settings.ollama_embed_model,
                    "input": text[:8000],
                },
            )
            response.raise_for_status()
            result = response.json()
            embeddings = result.get("embeddings", [])
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            return None
    except httpx.ConnectError:
        log.error("ollama_connection_failed", service="embedding")
        return None
    except httpx.HTTPStatusError as e:
        log.error("ollama_http_error", service="embedding", status=e.response.status_code)
        return None
    except Exception as e:
        log.error("ollama_unexpected_error", service="embedding", error=str(e))
        return None


async def process_document(
    doc_id: int,
    file_content: bytes,
    filename: str,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """
    Background task that processes a document.
    For .txt and .md files:
    - Sends content to Ollama for summarization
    - Generates embeddings using nomic-embed-text
    """
    log.info("doc_processing_started", doc_id=doc_id, filename=filename)

    summary = None
    embedding = None
    text_content = None

    if filename.lower().endswith((".txt", ".md")):
        try:
            text_content = file_content.decode("utf-8")

            log.info("doc_summarizing", doc_id=doc_id)
            summary = await summarize_with_ollama(text_content)
            if summary:
                log.info("doc_summary_received", doc_id=doc_id)
            else:
                log.warning("doc_summary_empty", doc_id=doc_id)

            log.info("doc_embedding", doc_id=doc_id)
            embedding = await generate_embedding(text_content)
            if embedding:
                log.info("doc_embedding_generated", doc_id=doc_id, dimensions=len(embedding))
            else:
                log.warning("doc_embedding_failed", doc_id=doc_id)

        except UnicodeDecodeError:
            log.warning("doc_decode_error", doc_id=doc_id, filename=filename)

    async with session_factory() as session:
        result = await session.execute(
            select(Document).where(Document.id == doc_id)
        )
        document = result.scalar_one_or_none()

        if document:
            document.status = "completed"
            if summary:
                document.summary = summary
            if text_content:
                document.content = text_content
            if embedding:
                document.embedding = embedding
            await session.commit()
            log.info("doc_processing_completed", doc_id=doc_id)
        else:
            log.error("doc_not_found", doc_id=doc_id)
