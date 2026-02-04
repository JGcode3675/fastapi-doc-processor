import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import Document

OLLAMA_BASE_URL = "http://host.docker.internal:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"
OLLAMA_LLM_MODEL = "llama3.2:3b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"


async def summarize_with_ollama(content: str) -> str | None:
    """
    Send content to Ollama for summarization.
    Returns the summary or None if Ollama is unavailable.
    """
    prompt = f"""Summarize this document in 3 bullet points:

{content[:1000]}"""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OLLAMA_GENERATE_URL,
                json={
                    "model": OLLAMA_LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
    except httpx.ConnectError:
        print("[Ollama] Connection failed - is Ollama running?")
        return None
    except httpx.HTTPStatusError as e:
        print(f"[Ollama] HTTP error: {e}")
        return None
    except Exception as e:
        print(f"[Ollama] Unexpected error: {e}")
        return None


async def generate_embedding(text: str) -> list[float] | None:
    """
    Generate embeddings using Ollama's nomic-embed-text model.
    Returns a 768-dimensional vector or None if unavailable.
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OLLAMA_EMBED_URL,
                json={
                    "model": OLLAMA_EMBED_MODEL,
                    "input": text[:8000],  # Limit input length
                },
            )
            response.raise_for_status()
            result = response.json()
            # Ollama returns embeddings in result["embeddings"][0]
            embeddings = result.get("embeddings", [])
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            return None
    except httpx.ConnectError:
        print("[Embedding] Connection failed - is Ollama running?")
        return None
    except httpx.HTTPStatusError as e:
        print(f"[Embedding] HTTP error: {e}")
        return None
    except Exception as e:
        print(f"[Embedding] Unexpected error: {e}")
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
    print(f"[Processing] Starting processing for document {doc_id} ({filename})...")

    summary = None
    embedding = None
    text_content = None

    # Check if file is text-based (.txt or .md)
    if filename.lower().endswith((".txt", ".md")):
        try:
            text_content = file_content.decode("utf-8")

            # Generate summary
            print(f"[Processing] Sending to Ollama for summarization...")
            summary = await summarize_with_ollama(text_content)
            if summary:
                print(f"[Processing] Received summary from Ollama")
            else:
                print(f"[Processing] No summary received from Ollama")

            # Generate embeddings
            print(f"[Processing] Generating embeddings with nomic-embed-text...")
            embedding = await generate_embedding(text_content)
            if embedding:
                print(f"[Processing] Generated embedding with {len(embedding)} dimensions")
            else:
                print(f"[Processing] Failed to generate embedding")

        except UnicodeDecodeError:
            print(f"[Processing] Could not decode file as UTF-8")

    # Update document in database
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
            print(f"[Processing] Document {doc_id} processing completed!")
        else:
            print(f"[Processing] Document {doc_id} not found!")
