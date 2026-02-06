"""
ARQ Task Queue — Persistent background processing.
====================================================

Replaces fire-and-forget asyncio.create_task with Redis-backed job queue.
Jobs survive restarts, support retries, and publish progress via Redis pubsub.
"""

import json

import redis.asyncio as aioredis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.config import get_settings
from app.logging_config import setup_logging, get_logger
from app.models import Base, Document
from app.services import summarize_with_ollama, generate_embedding

setup_logging()
log = get_logger(__name__)


async def _publish_status(ctx: dict, doc_id: int, status: str, progress: int) -> None:
    """Publish document processing status to Redis pubsub for WebSocket clients."""
    redis: aioredis.Redis = ctx.get("redis")
    if redis:
        await redis.publish(
            "ws:events",
            json.dumps({
                "type": "doc_status",
                "doc_id": doc_id,
                "status": status,
                "progress": progress,
            }),
        )


async def process_document_task(
    ctx: dict,
    doc_id: int,
    file_content: bytes,
    filename: str,
) -> str:
    """
    ARQ task: process a document (summarize + embed).

    Publishes progress updates via Redis pubsub at each stage.
    """
    session_factory: async_sessionmaker[AsyncSession] = ctx["session_factory"]
    log.info("task_started", doc_id=doc_id, filename=filename)
    await _publish_status(ctx, doc_id, "processing", 10)

    summary = None
    embedding = None
    text_content = None

    if filename.lower().endswith((".txt", ".md")):
        try:
            text_content = file_content.decode("utf-8")

            # Summarize
            await _publish_status(ctx, doc_id, "summarizing", 30)
            log.info("task_summarizing", doc_id=doc_id)
            summary = await summarize_with_ollama(text_content)

            # Embed
            await _publish_status(ctx, doc_id, "embedding", 60)
            log.info("task_embedding", doc_id=doc_id)
            embedding = await generate_embedding(text_content)

        except UnicodeDecodeError:
            log.warning("task_decode_error", doc_id=doc_id, filename=filename)

    # Update DB
    await _publish_status(ctx, doc_id, "saving", 90)
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
            log.info("task_completed", doc_id=doc_id)
        else:
            log.error("task_doc_not_found", doc_id=doc_id)

    await _publish_status(ctx, doc_id, "completed", 100)
    return f"Document {doc_id} processed"


async def startup(ctx: dict) -> None:
    """ARQ worker startup: initialize DB engine and session factory."""
    settings = get_settings()

    engine = create_async_engine(settings.database_url)
    ctx["session_factory"] = async_sessionmaker(
        autocommit=False, autoflush=False, bind=engine, class_=AsyncSession
    )
    ctx["redis"] = aioredis.from_url(settings.redis_url)
    log.info("arq_worker_started")


async def shutdown(ctx: dict) -> None:
    """ARQ worker shutdown: cleanup connections."""
    redis_conn = ctx.get("redis")
    if redis_conn:
        await redis_conn.close()
    log.info("arq_worker_stopped")


class WorkerSettings:
    """ARQ worker configuration."""
    settings = get_settings()

    functions = [process_document_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = None  # Set dynamically below
    max_jobs = settings.arq_max_jobs
    job_timeout = settings.arq_job_timeout

    @classmethod
    def get_redis_settings(cls):
        from arq.connections import RedisSettings
        return RedisSettings.from_dsn(cls.settings.redis_url)


# ARQ needs redis_settings as a class attribute
from arq.connections import RedisSettings  # noqa: E402
WorkerSettings.redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
