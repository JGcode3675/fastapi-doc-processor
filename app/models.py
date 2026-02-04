from datetime import datetime
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import DateTime, func, Text
from pgvector.sqlalchemy import Vector

Base = declarative_base()

# Embedding dimension for nomic-embed-text model
EMBEDDING_DIM = 768


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    filename: Mapped[str]
    content_type: Mapped[str | None] = mapped_column(default=None)
    file_size: Mapped[int | None] = mapped_column(default=None)
    status: Mapped[str] = mapped_column(default="pending")
    summary: Mapped[str | None] = mapped_column(Text, default=None)
    content: Mapped[str | None] = mapped_column(Text, default=None)
    embedding = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status}')>"
