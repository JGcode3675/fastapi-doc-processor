"""
Centralized Configuration
=========================

Single source of truth for all environment-driven settings.
Loaded from .env file or environment variables.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://user:password@db:5432/mydatabase"

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Ollama
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_llm_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"

    # ChromaDB
    chromadb_url: str = "http://host.docker.internal:8100"
    chromadb_collection_id: str = "69b78eb9-e090-42c2-b76d-9bf474088204"

    # ARQ Worker
    arq_max_jobs: int = 4
    arq_job_timeout: int = 300

    model_config = {"env_file": ".env"}

    @property
    def ollama_generate_url(self) -> str:
        return f"{self.ollama_base_url}/api/generate"

    @property
    def ollama_embed_url(self) -> str:
        return f"{self.ollama_base_url}/api/embed"

    @property
    def ollama_chat_url(self) -> str:
        return f"{self.ollama_base_url}/api/chat"


@lru_cache
def get_settings() -> Settings:
    return Settings()
