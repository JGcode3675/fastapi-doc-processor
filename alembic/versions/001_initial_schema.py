"""initial schema

Revision ID: 8a303ac66b2f
Revises:
Create Date: 2026-02-06

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy

# revision identifiers, used by Alembic.
revision: str = "8a303ac66b2f"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("content_type", sa.String(), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(768), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_documents_id"), "documents", ["id"])

    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("system_prompt", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chat_sessions_id"), "chat_sessions", ["id"])

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tool_calls", sa.JSON(), nullable=True),
        sa.Column("tokens_used", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["chat_sessions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chat_messages_id"), "chat_messages", ["id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_chat_messages_id"), table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index(op.f("ix_chat_sessions_id"), table_name="chat_sessions")
    op.drop_table("chat_sessions")
    op.drop_index(op.f("ix_documents_id"), table_name="documents")
    op.drop_table("documents")
